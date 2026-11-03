"""reseat-reentry controller (Run X idea x-r2-reseat-reentry; review runs/scaleagent-codex-20261103/reviews/
proposal_r2_review.json merged_spec + conditions K1-K10).

Every decision is exactly ONE `await ctx.policy.sample(1, sampler=SCALE)` with the ORIGINAL instruction and the raw
observation (obs=None, instruction=None); SCALE is the sampler dict the plain `scale` method sends
(hs_scale.decoder:ScaleDecoder with hs_scale.decoder.DEFAULTS); the first `execute_prefix` (= 1) action is executed
with its original dtype. The framework seeds the request with crn_seed(ns, contract@case_namespace, task, episode,
decision, 0) and the decision counter advances once per request exactly as in the scale DecisionLoop, so an episode
in which no re-seat moves the arm is bitwise the plain `scale` method under the same key. Never ctx.stop().

On a post-release gap fire (triggers_gap.py; proprio only) the simulator waits (no env step): call 1 shows the
planner the step-0 and the current agentview frames with the original instruction and asks for the remaining
movable instruction object (target pixel) or 'none'; the harness back-projects it (one replan only after a failed
geometric/workspace check); call 2 is the same-frame marked-crop witness with the measured-size cue; on confirm the
open gripper is moved up, across and down to hover 0.10 m above the object's local top (<= 45 primitive steps, never
a close), and SCALE resumes with the original instruction. Abstain / 'none' / articulated part / witness reject /
transport failure / budget end -> hand back with NO motion; the proposal of that decision is then executed.

Modes: "candidate"; "respondoff" (SCALE-shadow null: same trigger, planner + witness calls and back-projection,
response suppressed, executed actions = SCALE). The Campaign null is the framework's shadow null (`null: shadow`).

Vendored from the witnessed-rescue method package (methods/witnessed_rescue, controller module after fix2b, sha256
de4577f9...): scale_sampler, _as_recorded, _clean, _argv_out / evidence_dirs, _step, _retain, _call (renamed
evidence variables, otherwise the same code).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from . import motion as M
from . import perception as P
from . import planner as PL
from . import triggers_gap as T

MODES = ("candidate", "respondoff")

DEFAULTS = {
    "mode": "candidate",
    "execute_prefix": 1,
    "max_perception_calls": 6,
    "retain_packets": True,
    # scope rule (review): remaining work that is an articulated part is never a re-seat target; a code guard on the
    # target label in addition to the planner's `kind` (generic fixture categories, not a per-task table)
    "articulated_words": ["drawer", "drawers", "door", "knob", "knobs", "handle", "lid", "stove", "burner",
                          "microwave", "oven", "cabinet", "faucet", "switch"],
}

_EPISODES = itertools.count()


def _argv_out(argv) -> Optional[str]:
    """The `--out` of the hs process this harness server runs in (hs eval / hs verify --tier 2)."""
    argv = list(argv or ())
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--out="):
            return a[len("--out="):]
    return None


def evidence_dirs(argv=None, env=None) -> dict:
    """Default (frozen in code): <hs --out>/reseat_reentry_evidence/{packets,planner}. Development-only overrides
    (never set for scored runs): HS_RR_PACKET_DIR / HS_RR_PLANNER_LOG_DIR. Neither: nothing is written."""
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    out = _argv_out(argv)
    base = (Path(out).resolve() / "reseat_reentry_evidence") if out else None
    res = {}
    for key, var in (("packets", "HS_RR_PACKET_DIR"), ("planner", "HS_RR_PLANNER_LOG_DIR")):
        if env.get(var):
            res[key], res[key + "_source"] = str(Path(env[var]).resolve()), f"override:{var}"
        elif base is not None:
            res[key], res[key + "_source"] = str(base / key), "argv:--out"
        else:
            res[key], res[key + "_source"] = None, "none"
    return res


def scale_sampler() -> dict:
    """The exact sampler dict the plain `scale` method sends (its DecisionLoop: config.slots['sampler'].to_json())."""
    from harnessstudio.core.method import ModuleRef
    from hs_scale.decoder import DEFAULTS as SCALE_DEFAULTS
    return ModuleRef("hs_scale.decoder:ScaleDecoder", dict(SCALE_DEFAULTS)).to_json()


def _state_of(obs: Mapping[str, Any]) -> Optional[np.ndarray]:
    v = obs.get("states", obs.get("state"))
    return None if v is None else np.asarray(v, dtype=np.float64).reshape(-1)


def _as_recorded_state(st: Optional[np.ndarray]) -> np.ndarray:
    """Post-step proprio as vla-eval's LIBERO recorder stores it (float32), so online == jsonl replay input."""
    return st.astype(np.float32).astype(np.float64) if st is not None else np.zeros(0)


def _clean(d: Any) -> Any:
    if isinstance(d, Mapping):
        return {k: _clean(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


class ReseatReentry:
    TIER1_LLM_RESPONSES: list = []    # no framework operator; scripted replies come from planner.scripted_responses

    def __init__(self, mode: str = "candidate", execute_prefix: int = 1, max_perception_calls: int = 6,
                 retain_packets: bool = True, articulated_words=None, triggers: Optional[Mapping[str, Any]] = None,
                 planner: Optional[Mapping[str, Any]] = None, perception: Optional[Mapping[str, Any]] = None,
                 motion: Optional[Mapping[str, Any]] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if type(execute_prefix) is not int or execute_prefix < 1:
            raise ValueError("execute_prefix must be a positive int")
        self.mode, self.execute_prefix = mode, execute_prefix
        self.max_perception_calls, self.retain_packets = int(max_perception_calls), bool(retain_packets)
        words = DEFAULTS["articulated_words"] if articulated_words is None else articulated_words
        self.articulated_words = frozenset(str(w).lower() for w in words)
        self.trigger_params = dict(triggers or {})
        self.planner_params = dict(planner or {})
        self.perception_params = {**P.DEFAULTS, **dict(perception or {})}
        self.motion_params = dict(motion or {})
        T.GapMonitor(self.trigger_params)
        M.Motion(lambda a: None, lambda: None, lambda: True, self.motion_params)
        if set(self.perception_params) - set(P.DEFAULTS):
            raise ValueError(f"unknown perception params {sorted(set(self.perception_params) - set(P.DEFAULTS))}")
        if set({**PL.DEFAULTS, **self.planner_params}) - set(PL.DEFAULTS):
            raise ValueError("unknown planner params")
        self.sampler = scale_sampler()
        self.size = 0

    # ------------------------------------------------------------------------------------------------ episode
    async def run(self, ctx) -> None:
        self.ctx = ctx
        self.S: list = []
        self.monitor = T.GapMonitor(self.trigger_params)
        self.perception_calls = 0
        first = ctx.observe()
        self.start_packet = P.Packet.capture(first, ctx.steps)      # step-0 frame, retained by the harness at reset
        st0 = _state_of(first)
        scene = hashlib.sha256((ctx.instruction + "|" + (P.sha(st0) if st0 is not None else "")).encode()).hexdigest()[:16]
        self.tag = f"{scene}_p{os.getpid()}_e{next(_EPISODES):04d}"
        dirs = evidence_dirs()
        log_path = str(Path(dirs["planner"]) / f"planner_calls_p{os.getpid()}.jsonl") if dirs["planner"] else None
        self.planner = PL.PlannerSession(self.planner_params, log_path=log_path, episode_id=self.tag)
        self.packet_dir = dirs["packets"] if self.retain_packets else None
        self.motion = M.Motion(self._primitive_step, lambda: _state_of(ctx.observe()), lambda: ctx.ended,
                               self.motion_params)
        self.acc = {"mode": self.mode, "episode_tag": self.tag, "scene_tag": scene, "evidence_dirs": dirs,
                    "proprio_dims": 0 if st0 is None else int(len(st0)), "policy_steps": 0, "primitive_steps": 0,
                    "pause_count": 0, "pause_wall_s": 0.0, "fires": [], "escalations": [], "reseats": [],
                    "planner": {}, "perception_calls": 0, "analytic_share": 0.0, "closes_commanded_by_primitive": 0,
                    "sampler": self.sampler["factory"], "planner_backend": self.planner.p["backend"],
                    "planner_model": self.planner.p["model"], "planner_effort": self.planner.p["effort"]}
        ctx.accounting.extra["reseat_reentry"] = self.acc
        try:
            while not ctx.ended:
                obs = ctx.observe()
                proposals = await ctx.policy.sample(1, sampler=self.sampler)
                chunk = np.array(proposals[0].actions)[: self.execute_prefix]
                fire = self.monitor.check(self.S, ctx.remaining)
                if fire is not None:
                    self.acc["fires"].append(fire.to_json())
                    ctx.record("rr_trigger", fire=fire.to_json())
                    step0, t0 = ctx.steps, time.monotonic()
                    self.acc["pause_count"] += 1
                    try:
                        await self._escalate(fire, obs)
                    except Exception as exc:  # noqa: BLE001 - a re-seat bug must never invalidate the episode
                        ctx.record("rr_error", where="escalation", error=f"{type(exc).__name__}: {str(exc)[:300]}")
                        self.acc.setdefault("errors", []).append(f"{type(exc).__name__}: {str(exc)[:200]}")
                    finally:
                        self.acc["pause_wall_s"] = round(self.acc["pause_wall_s"] + time.monotonic() - t0, 3)
                        if self.acc["escalations"]:
                            self.acc["escalations"][-1] = _clean(self.acc["escalations"][-1])
                            ctx.record("rr_escalation", escalation=self.acc["escalations"][-1])
                        self._update()
                    if ctx.steps != step0:
                        continue      # the arm moved: the proposal is stale, decide again
                for a in chunk:
                    if ctx.ended:
                        break
                    await self._step(a, "policy")
        finally:
            self.planner.close()
            self._update()

    async def _step(self, action: np.ndarray, source: str) -> int:
        a = np.asarray(action).reshape(-1)     # policy actions keep their dtype: the trace hashes dtype + bytes
        if source == "primitive" and float(a[-1]) >= 0:
            self.acc["closes_commanded_by_primitive"] += 1     # must stay 0 (tested); vla-eval: >= 0 closes
        n = await self.ctx.execute(a[None, :])
        if n:
            self.S.append(_as_recorded_state(_state_of(self.ctx.observe())))
            self.acc["policy_steps" if source == "policy" else "primitive_steps"] += n
        return n

    async def _primitive_step(self, action: np.ndarray) -> int:
        return await self._step(action, "primitive")

    def _update(self) -> None:
        total = self.acc["policy_steps"] + self.acc["primitive_steps"]
        self.acc["analytic_share"] = round(self.acc["primitive_steps"] / total, 4) if total else 0.0
        self.acc["planner"] = self.planner.summary()
        self.acc["perception_calls"] = self.perception_calls

    # ------------------------------------------------------------------------------------------------ helpers
    def _retain(self, packet: P.Packet, label: str, images: Optional[dict] = None) -> dict:
        info = {"step": packet.step, "hashes": packet.hashes}
        if not self.packet_dir:
            return info
        try:
            root = Path(self.packet_dir) / self.tag
            root.mkdir(parents=True, exist_ok=True)
            base = f"s{packet.step:04d}_{label}"
            arrays = {}
            for k, v in (packet.obs.get("images") or {}).items():
                arrays[f"images__{k}"] = np.asarray(v)
            for k, v in (packet.obs.get("depth") or {}).items():
                arrays[f"depth__{k}"] = np.asarray(v)
            st = packet.state()
            if st is not None:
                arrays["states"] = st
            np.savez_compressed(root / f"{base}.npz", **arrays)
            meta = {"step": packet.step, "hashes": packet.hashes, "instruction": self.ctx.instruction,
                    "camera": _clean(packet.obs.get("camera") or {})}
            (root / f"{base}.json").write_text(json.dumps(meta, sort_keys=True))
            for name, img in (images or {}).items():
                from harnessstudio.tools.llm import png_bytes
                (root / f"{base}_{name}.png").write_bytes(png_bytes(img))
            info["files"] = str(root / base)
        except Exception as exc:  # noqa: BLE001 - retention is evidence, never control
            info["retain_error"] = f"{type(exc).__name__}: {exc}"
        return info

    def _agentview(self, packet: P.Packet) -> Optional[np.ndarray]:
        s = int(self.perception_params["image_scale"])
        av = P.rgb_view(packet.obs, "agentview")
        if av is None:
            return None
        self.size = int(av.shape[0] * s)
        return P.upsample(av, s)

    async def _call(self, esc: dict, role: str, prompt: str, images: list, schema: dict, validate=None):
        if esc["calls"] >= int(self.planner.p["max_calls_escalation"]):
            res = PL.CallResult(role, "budget_refused", record={"error": "per-escalation planner budget spent"})
        else:
            esc["calls"] += 1
            res = await self.planner.ask(role, prompt, images, schema, validate)
        esc["planner_calls"].append(res.to_json())
        return res

    def _handback(self, esc: dict, why: str) -> None:
        esc["handback"] = why
        esc["end_step"] = self.ctx.steps

    def _hover(self, esc: dict, packet: P.Packet, target: Mapping[str, Any]) -> dict:
        if self.perception_calls >= self.max_perception_calls:
            return {"ok": False, "reason": "perception budget spent"}
        self.perception_calls += 1
        loc = P.hover_target(packet, target["row"], target["col"], self.perception_params)
        esc["hover_target"].append(_clean(loc))
        return loc

    async def _point(self, esc: dict, packet: P.Packet, images: list, note: str = "") -> Optional[dict]:
        size = self.size or 512
        res = await self._call(esc, "point", PL.reseat_prompt(self.ctx.instruction, size, note), images,
                               PL.RESEAT_SCHEMA, validate=lambda o: PL.validate_reseat(o, size))
        if res.status != "ok":
            return None
        obj = res.obj
        if obj["action"] == "none" or obj.get("target") is None:
            res.outcome = "abstain"
        elif not PL.label_ok(obj["target"]["label"], self.ctx.instruction):
            res.outcome = "invalid"
        elif PL.target_kind(obj) == "articulated_part" or (
                self.articulated_words & set(PL.content_words(obj["target"]["label"]))):
            res.outcome = "articulated"          # scope rule: articulated remaining work is never re-seated
        else:
            res.outcome = "accept"
        esc["planner_calls"][-1]["outcome"] = res.outcome
        return obj if res.outcome == "accept" else None

    # ------------------------------------------------------------------------------------------------ re-seat
    async def _escalate(self, fire: T.Fire, obs: Mapping[str, Any]) -> None:
        ctx = self.ctx
        esc = {"kind": "post_release_gap", "step": ctx.steps, "steps_left": ctx.remaining, "calls": 0,
               "planner_calls": [], "hover_target": [], "retained": [], "reseat": None, "handback": None}
        self.acc["escalations"].append(esc)
        packet = P.Packet.capture(obs, ctx.steps)
        start_img, now_img = self._agentview(self.start_packet), self._agentview(packet)
        if start_img is None or now_img is None:
            return self._handback(esc, "no_agentview_image")
        esc["retained"].append(self._retain(self.start_packet, "start", {"agentview": start_img}))
        esc["retained"].append(self._retain(packet, "fire", {"agentview": now_img}))
        images = [start_img, now_img]
        plan = await self._point(esc, packet, images)
        if plan is None:
            return self._handback(esc, "point_not_accepted")
        loc = self._hover(esc, packet, plan["target"])
        if not loc.get("ok"):
            esc["planner_calls"][-1]["geometry_reject"] = loc.get("reason")
            note = (f"Your previous point ({plan['target']['row']}, {plan['target']['col']}) was rejected by the depth "
                    f"check ({loc.get('reason')}). Point again at the object itself, or answer 'none'.")
            plan = await self._point(esc, packet, images, note)
            if plan is None:
                return self._handback(esc, "replan_not_accepted")
            loc = self._hover(esc, packet, plan["target"])
            if not loc.get("ok"):
                esc["planner_calls"][-1]["geometry_reject"] = loc.get("reason")
                return self._handback(esc, f"workspace_check_failed: {loc.get('reason')}")
        try:
            full, crop = P.marked_views(packet, *loc["pixel_native"], self.perception_params)
        except P.PerceptionUnavailable as exc:
            return self._handback(esc, f"perception_unavailable: {exc}")
        esc["retained"].append(self._retain(packet, f"witness{len(esc['retained'])}", {"marked": full, "crop": crop}))
        label = plan["target"]["label"]
        res = await self._call(esc, "witness", PL.witness_prompt(ctx.instruction, label, False, loc), [full, crop],
                               PL.VERDICT_SCHEMA)
        if res.status != "ok":
            return self._handback(esc, "witness_transport")
        res.outcome = {"confirm": "accept", "wrong_object": "wrong_object"}.get(res.obj.get("verdict"), "abstain")
        esc["planner_calls"][-1]["outcome"] = res.outcome
        if res.outcome != "accept":
            return self._handback(esc, f"witness_{res.outcome}")
        esc["target"] = {"label": label, "hover_xyz": loc["hover_xyz"], "transit_z": loc["transit_z"]}
        if self.mode == "respondoff":
            return self._handback(esc, "respondoff")    # response suppressed: the SCALE proposal is executed
        if ctx.steps != packet.step:
            return self._handback(esc, "stale_packet")
        budget = M.reseat_budget(ctx.remaining, self.motion.p)       # ruling 2: steps_left at the fire (no env step yet)
        rs = await self.motion.reseat(loc["hover_xyz"], loc["transit_z"], budget)
        esc["reseat"] = _clean(rs)
        # C4: a re-seat that stops short is 'not delivered' (executed motion without a hover arrival), never relabelled
        self.acc["reseats"].append({"step": esc["step"], "label": label, "steps": rs["steps"], "budget": budget,
                                    "reached": rs["reached"], "final_error": rs["final_error"],
                                    "budget_exhausted": rs["budget_exhausted"],
                                    "delivered": bool(rs["reached"]),
                                    "legs": [{k: leg.get(k) for k in ("leg", "steps", "reached", "error", "stalled")}
                                             for leg in rs["legs"]]})
        return self._handback(esc, "after_reseat" if rs["reached"] else "reseat_stopped")
