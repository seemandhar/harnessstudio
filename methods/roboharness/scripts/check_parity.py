"""Parity of the HarnessStudio port against the authors' RoboHarness code (commit 716defc), on randomized inputs.

    .venv/bin/python methods/roboharness/scripts/check_parity.py --src /home/sejain/research/cvpr27/RoboHarness/src

Checks (original module vs port):
  A. control flow: RollbackState / KeyStepRetryState / TaskDecomposeState on random operation sequences
  B. PerceptionModule.process_frame (released source with MAX_RETRIES = 0 -> 3, the port's documented fix) vs
     Perception.process_frame: same scripted VLM replies and the same emulated SAM3 tool server, sequences of
     calls per episode (quota, Encore reset, verification retries); compares frames, refined task, flags,
     counters
  C. prompts: every message list the original sends (orchestrate, failure report, success summary, detect,
     verify) equals the port's, images replaced by their content hash; JSON / box / verification parsers
  D. MemoryBank.add_experience + retrieve on random vectors
Only transport is stubbed: cv2 (debug image saving) and requests (the HTTP post, replaced by the emulator).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from hs_roboharness import control_flow as PC  # noqa: E402
from hs_roboharness import memory as PM  # noqa: E402
from hs_roboharness import perception as PP  # noqa: E402
from hs_roboharness import vlm as PV  # noqa: E402

TOOLS = ["visual_overlay", "remove_distractor", "replace_texture", "replace_background", "encore", "key_step_retry",
         "task_decompose", "instruction_refine", "chaining_step", "bogus_tool"]
FAILS: list = []


def check(cond: bool, what: str) -> None:
    if not cond:
        FAILS.append(what)


def load(src: Path, name: str, patch: tuple | None = None):
    text = (src / f"{name}.py").read_text()
    if patch:
        assert patch[0] in text, f"{patch[0]!r} not in {name}.py"
        text = text.replace(patch[0], patch[1])
    mod = types.ModuleType(name)
    mod.__file__ = str(src / f"{name}.py")
    sys.modules[name] = mod
    exec(compile(text, mod.__file__, "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------ A. control flow
def check_control_flow(O, n: int, rng: random.Random) -> int:
    ops = 0
    for _ in range(n):
        o_rb, p_rb = O.RollbackState(), PC.RollbackState()
        o_rc, p_rc = O.KeyStepRetryState(), PC.KeyStepRetryState()
        o_tc, p_tc = O.TaskDecomposeState(), PC.TaskDecomposeState()
        for step in range(rng.randint(5, 120)):
            r = rng.random()
            if r < 0.08:
                acc = np.asarray([[rng.uniform(-3, 3) for _ in range(7)]], dtype=np.float32)
                kw = {"reverse_steps": rng.choice([0, 1, 5, 40, 50]), "buffer_steps": rng.choice([0, 3, 20, 25])}
                o_rb.start_from_accumulated(acc, **kw)
                p_rb.start_from_accumulated(acc, **kw)
            if o_rb.active:
                a, b = o_rb.step_action(1), p_rb.step_action(1)
                check(np.array_equal(a, b) and a.dtype == b.dtype, f"rollback action {step}")
            check((o_rb.active, o_rb.phase) == (p_rb.active, p_rb.phase), f"rollback state {step}")
            if r > 0.9:
                flags = rng.choice([{"key_steps": {"start": 0, "end": rng.randint(0, 30), "timeout_grace": rng.randint(0, 5)}},
                                    {"key_steps": [1, 2]}, {"key_step_retry": True}, {"key_steps": {"end": 4}}, {}])
                o_rc.update_from_control_flags(flags)
                p_rc.update_from_control_flags(flags)
            succ = rng.random() < 0.1
            check(o_rc.should_trigger(step=step, success_any=succ) == p_rc.should_trigger(step=step, success_any=succ),
                  f"retry trigger {step}")
            check((o_rc.armed, o_rc.triggered, o_rc.key_steps) == (p_rc.armed, p_rc.triggered, p_rc.key_steps), "retry state")
            if 0.8 < r < 0.9:
                flags = {"subtasks": rng.choice([["a", "b", "c"], ["x"], [], ["a", 3], "notalist", ["a", "b", "c"]])}
                o_tc.update_from_control_flags(flags)
                p_tc.update_from_control_flags(flags)
            o_tc.maybe_advance(success_any=succ)
            p_tc.maybe_advance(success_any=succ)
            check(o_tc.current_task("base") == p_tc.current_task("base"), "subtask")
            ops += 1
    return ops


# ------------------------------------------------------------------ tool server emulator (both sides)
def emulate_tool(path: str, image: np.ndarray, prompt: str) -> tuple[bool, np.ndarray]:
    h = int(hashlib.sha256(f"{path}|{prompt}".encode()).hexdigest(), 16)
    if h % 4 == 0:   # "mask not found": original image back, success False
        return False, image
    out = image.copy()
    out[(h >> 8) % image.shape[0], :, (h >> 16) % 3] = (h >> 24) % 256
    return True, out


def b64png(img: np.ndarray) -> str:
    return PP.png_b64(img)


def img_tag(url_or_img) -> str:
    if isinstance(url_or_img, str):
        return "IMG"
    return "IMG:" + hashlib.sha256(np.ascontiguousarray(url_or_img).tobytes()).hexdigest()[:12]


def norm_messages(messages: list) -> list:
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), list):
            for part in m["content"]:
                if part.get("type") == "image_url":
                    part["image_url"] = {"url": "IMG"}
    return out


# ------------------------------------------------------------------ B + C. perception
def random_plan(rng: random.Random) -> str:
    chain = rng.sample(TOOLS, rng.randint(0, 4))
    params: dict = {}
    for t in chain:
        if t == "visual_overlay":
            params[t] = {"target_object": rng.choice(["the red bowl", "", None, "bowl in the center"])}
        elif t == "remove_distractor":
            params[t] = rng.choice([{"objects_to_remove": ["bowl left", "bowl right", ""]}, {"object_to_remove": "plate"},
                                    {"objects_to_remove": "not a list"}, {}])
        elif t == "replace_texture":
            params[t] = {"target_object": "bowl", "source": rng.choice(["rag_success", "rag_failure"])}
        elif t == "replace_background":
            params[t] = {"region_prompt": "floor", "alpha": 0.5}
        elif t == "key_step_retry":
            params[t] = rng.choice([{"key_steps": {"start": 0, "end": 30, "timeout_grace": 5}}, {}])
        elif t == "task_decompose":
            params[t] = rng.choice([{"subtasks": ["pick a", "place a"]}, {"subtasks": []}, {}])
    plan = {"reasoning": "r", "tool_chain": chain, "params": params}
    if rng.random() < 0.8:
        plan["refined_task"] = rng.choice(["pick up the highlighted bowl", "", "pick the ketchup"])
    if rng.random() < 0.2:
        plan["task_plan"] = {"key_steps": {"start": 0, "end": 12, "timeout_grace": 2}, "subtasks": ["s1", "s2"]}
    if rng.random() < 0.15:
        plan["params"] = rng.choice([[], "x"])
    text = json.dumps(plan)
    return rng.choice([text, f"```json\n{text}\n```", f"Sure. {text} done", "no json here"]) if rng.random() < 0.3 else text


def random_verify(rng: random.Random) -> str:
    return rng.choice(['{"valid": true, "reason": "ok"}', '{"valid": false, "reason": "wrong object"}',
                       "{'valid': false, 'reason': 'single quotes'}", "nothing", '{"reason": "missing"}'])


def check_perception(src: Path, n_eps: int, rng: random.Random) -> int:
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    req = types.ModuleType("requests")
    sys.modules["requests"] = req
    OV = load(src, "soma_vlm")
    OT = load(src, "soma_tools")
    OP = load(src, "soma_perception", ("MAX_RETRIES = 0", "MAX_RETRIES = 3"))
    calls = 0
    for ep in range(n_eps):
        script_plans = [random_plan(rng) for _ in range(40)]
        script_verify = [random_verify(rng) for _ in range(40)]
        orig_msgs: list = []

        # --- original side
        vlm = OV.Qwen3VLAPIClient(api_key="x", base_url="https://example.invalid", model_id="m")
        state = {"p": 0, "v": 0}

        def _generate(messages, max_tokens=1024, temperature=0.01, _s=state):
            orig_msgs.append(("brain", max_tokens, temperature, norm_messages(messages)))
            t = script_plans[_s["p"] % len(script_plans)]
            _s["p"] += 1
            return t
        vlm._generate = _generate

        class _Completions:
            def create(self, model, messages, max_tokens, temperature, _s=state):
                orig_msgs.append(("verifier", max_tokens, temperature, norm_messages(messages)))
                t = script_verify[_s["v"] % len(script_verify)]
                _s["v"] += 1
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=t))])
        vlm.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))

        def _post(self, path, payload):
            img = PP.decode_b64_image(payload["image"])
            prompt = payload.get("target_object") or payload.get("object_to_remove") or payload.get("region_prompt") or ""
            ok, out = emulate_tool(path, img, prompt)
            return {"success": ok, "image": b64png(out)}
        OT.Sam3HttpClient._post = _post
        orig = OP.PerceptionModule(vlm_client=vlm)

        # --- port side
        pstate = {"p": 0, "v": 0}
        port_msgs: list = []

        async def llm(messages, operator, _s=pstate):
            if operator == PV.BRAIN:
                port_msgs.append(("brain", PV.MAX_TOKENS[operator], 0.01, norm_messages(messages)))
                t = script_plans[_s["p"] % len(script_plans)]
                _s["p"] += 1
                return t
            if operator == PV.VERIFIER:
                port_msgs.append(("verifier", PV.MAX_TOKENS[operator], 0.1, norm_messages(messages)))
                t = script_verify[_s["v"] % len(script_verify)]
                _s["v"] += 1
                return t
            return "[100, 100, 300, 300]"   # detector: original does this inside the service

        class FakeTools:
            async def image_tool(self, path, image, **fields):
                prompt = fields.get("target_object") or fields.get("object_to_remove") or fields.get("region_prompt") or ""
                ok, out = emulate_tool(path, image, prompt)
                return out, {"success": ok}
        port = PP.Perception(llm, FakeTools(), max_attempts=3)

        rag_variants = [{}, {"failure": [{"diagnosis": "Visual Ambiguity: wrong bowl"}, {"diagnosis": ""},
                                         {"diagnosis": "Control Error: slipped"}], "success": [{"info": {}}]},
                        {"failure": [], "success": []}]
        rag = rng.choice(rag_variants)
        image = (np.arange(64 * 64 * 3, dtype=np.int64).reshape(64, 64, 3) * (ep + 1) % 251).astype(np.uint8)
        for k in range(rng.randint(3, 9)):
            task = rng.choice(["put the bowl on the plate", "Hey umm grab the bottle"])
            o_img, o_task, o_flags = orig.process_frame(image, task, 10 * k, copy.deepcopy(rag))
            p_img, p_task, p_flags = asyncio.run(port.process_frame(image, task, 10 * k, copy.deepcopy(rag)))
            check(np.array_equal(o_img, p_img), f"ep{ep} call{k} frame")
            check(o_task == p_task, f"ep{ep} call{k} refined task {o_task!r} vs {p_task!r}")
            check(o_flags == p_flags, f"ep{ep} call{k} flags {o_flags} vs {p_flags}")
            check((orig.visual_tool_steps, orig.retry_count) == (port.visual_tool_steps, port.retry_count),
                  f"ep{ep} call{k} counters")
            calls += 1
        check(orig_msgs == port_msgs, f"ep{ep} prompt/message sequence differs "
                                      f"({len(orig_msgs)} vs {len(port_msgs)} calls)")

    # --- C. other prompts and parsers
    vlm = OV.Qwen3VLAPIClient(api_key="x", base_url="https://example.invalid", model_id="m")
    cap: list = []
    vlm._generate = lambda messages, max_tokens=1024, temperature=0.01: (cap.append((max_tokens, temperature, norm_messages(messages))), "{}")[1]
    from PIL import Image
    imgs = [Image.fromarray(np.full((8, 8, 3), i, np.uint8)) for i in range(5)]   # the logger passes PIL frames
    vlm.generate_failure_report(imgs, "task x")
    check(cap[-1] == (1024, 0.01, norm_messages(PV.failure_report_messages(imgs, "task x"))), "failure report prompt")
    vlm.generate_success_description(imgs, "task x")
    check(cap[-1] == (1024, 0.01, norm_messages(PV.success_description_messages(imgs, "task x"))), "success prompt")
    arr = np.full((8, 8, 3), 7, np.uint8)
    vlm.detect_object(arr, "the bowl")
    check(cap[-1] == (128, 0.0, norm_messages(PV.detect_messages(arr, "the bowl"))), "detect prompt")
    for _ in range(3000):
        s = rng.choice(['{"a": 1}', '```json\n{"b": 2}\n```', 'x {"c": [1,2]} y', "{bad", "", "```\n{\"d\": 4}\n```",
                        '[1, 2]', 'pre ```json {"e": 5}``` post', '{"f": {"g": 1}} and {"h": 2}'])
        s = s + rng.choice(["", " ", "}", "{"])
        check(OV._extract_json(s) == PV.extract_json(s), f"extract_json {s!r}")
        txt = " ".join(str(rng.randint(0, 1200)) for _ in range(rng.randint(0, 6))) + rng.choice(["", " ymin", "[", "]"])
        vlm._generate = lambda messages, max_tokens=1024, temperature=0.01, _t=txt: _t
        h, w = rng.choice([(256, 256), (128, 200)])
        check(vlm.detect_object(np.zeros((h, w, 3), np.uint8), "x") == PV.parse_detection(txt, h, w), f"detect parse {txt!r}")
    return calls


# ------------------------------------------------------------------ D. memory
def check_memory(src: Path, rng: random.Random) -> int:
    OM = load(src, "soma_memory")
    n = 0
    for trial in range(20):
        with tempfile.TemporaryDirectory() as d:
            ob = OM.MemoryBank(d, dimension=1168)
            pb = PM.MemoryBank()
            for i in range(rng.randint(0, 40)):
                v = np.random.default_rng(trial * 100 + i).standard_normal(1168).astype(np.float32)
                s = rng.random() < 0.6
                ob.add_experience(v.tolist(), f"t{i % 5}", s, "/tmp/v.mp4", "/tmp/k.jpg", diagnosis=f"d{i}", info={})
                pb.add_experience(v, f"t{i % 5}", s, diagnosis=f"d{i}", info={})
            for q in range(10):
                qv = np.random.default_rng(10_000 + trial * 10 + q).standard_normal(1168).astype(np.float32)
                if q % 2:   # near-duplicate of a stored vector so the threshold matters
                    qv = np.random.default_rng(trial * 100).standard_normal(1168).astype(np.float32) + 0.5 * qv
                for part, k, th in (("success", 1, 0.4), ("failure", 2, 0.4), ("all", 3, -1.0)):
                    a = ob.retrieve(qv, top_k=k, partition=part, threshold=th)
                    b = pb.retrieve(qv, top_k=k, partition=part, threshold=th)
                    check([(x["diagnosis"], x["type"], round(x["score"], 5)) for x in a]
                          == [(x["diagnosis"], x["type"], round(x["score"], 5)) for x in b], f"retrieve {trial}/{q}/{part}")
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/sejain/research/cvpr27/RoboHarness/src")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = Path(args.src)
    rng = random.Random(args.seed)
    O = load(src, "soma_control_flow")
    a = check_control_flow(O, 400, rng)
    b = check_perception(src, args.episodes, rng)
    d = check_memory(src, rng)
    print(json.dumps({"control_flow_steps": a, "perception_calls": b, "retrievals": d, "failures": len(FAILS),
                      "first_failures": FAILS[:8]}, indent=1))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
