"""witnessed-rescue controller (Run X idea x-r1-witnessed-rescue; review runs/scaleagent-codex-20261103/reviews/
proposal_r1_review.json).

Every decision is exactly ONE `await ctx.policy.sample(1, sampler=SCALE)` with the ORIGINAL instruction and the raw
observation (obs=None, instruction=None), where SCALE is the very sampler ModuleRef the plain `scale` method sends
(hs_scale.decoder:ScaleDecoder with hs_scale.decoder.DEFAULTS); the first `execute_prefix` (= OpenVLA's native 1)
actions are executed. The framework seeds that request with crn_seed(ns, contract@case_namespace, task, episode,
decision, 0), and the decision counter advances once per request exactly as in the scale DecisionLoop, so an
episode in which no escalation moves the arm is bitwise the plain `scale` method under the same case key (review
C1, A3). The controller never calls ctx.stop() and loops until ctx.ended.

At each decision the public trigger monitor (triggers.py) sees the executed actions, the post-step proprio and the
proposed action. On an eligible fire the controller pauses (no env step), retains the public packet and runs the
rescue: call 1 (planner points at the instruction's target and destination), geometric witness gate on metric
depth, call 2 (same-frame marked-crop confirmation), analytic grasp + measured custody test (or, for H1, lift +
custody), call 3 (wrist-crop custody confirmation) before an analytic place into an open-top destination with a
free depth column, then hand-back. Any abstention, rejection, timeout, transport error or spent budget hands back
without motion; when no env step was taken the proposed SCALE action is executed as if nothing happened.

Modes: "candidate" (respond), "respondoff" (the SCALE-shadow null: same triggers, planner and perception calls,
responses suppressed, executed actions = SCALE), "analytic" (primitives-only ablation, descriptive). The Campaign
null is the framework's shadow null (paper card `null: shadow`): this controller runs unchanged in that arm, pays
its SCALE samples on the shadow session and its planner calls, while the framework executes the baseline's native
(greedy) actions.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from . import motion as M
from . import perception as P
from . import planner as PL
from . import triggers as T

MODES = ("candidate", "respondoff", "analytic")

DEFAULTS = {
    "mode": "candidate",
    "place_enabled": True,             # physical place rule B4 may freeze it to False
    "execute_prefix": 1,               # OpenVLA executes one action per request (profile execute_horizon 1)
    "min_steps_left_place": 170,
    "max_grasp_attempts": 2,
    "max_perception_calls": 30,
    "witness_route": "marked_crop",    # SAM3 route not built (adopt only if qualification shows +10 pp)
    "retain_packets": True,            # write packets when HS_WR_PACKET_DIR is set (deployment detail)
    "custody_abstain_policy": "measured",   # planner abstains on custody: "measured" = trust the measured test
    # fix2b/K1: an analytic place is never made into a destination whose label names a furniture/appliance
    # compartment or enclosed container (generic categories from the approved spec clause, not a per-task table).
    "closed_destination_words": ["drawer", "drawers", "cabinet", "cupboard", "shelf", "compartment", "microwave",
                                 "oven", "fridge", "refrigerator", "caddy", "lid", "lidded"],
}


_EPISODES = itertools.count()   # per-process episode counter: evidence paths never collide within a process


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
    """Where retained packets and the planner call log go. Default (deterministic, frozen in code): under the hs
    run's own output dir, <--out>/witnessed_rescue_evidence/{packets,planner}. Override (development only; the
    evaluator never sets it for scored runs): HS_WR_PACKET_DIR / HS_WR_PLANNER_LOG_DIR. Neither: nothing is written
    (the trace keeps every hash). Paths never change actions."""
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    out = _argv_out(argv)
    base = (Path(out).resolve() / "witnessed_rescue_evidence") if out else None
    res = {}
    for key, var in (("packets", "HS_WR_PACKET_DIR"), ("planner", "HS_WR_PLANNER_LOG_DIR")):
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


def _as_recorded(action: np.ndarray) -> np.ndarray:
    """The action row as vla-eval's LIBERO recorder stores it (gripper discretised: < 0 -> -1 else +1; float32),
    so the online trigger input equals the offline jsonl replay input bit for bit."""
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    a[-1] = -1.0 if a[-1] < 0 else 1.0
    return a.astype(np.float64)


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


class WitnessedRescue:
    # No framework operator is declared (codex is a method-local isolated CLI operator), so tier 1 needs no script;
    # scripted planner replies come from planner.scripted_responses.
    TIER1_LLM_RESPONSES: list = []

    def __init__(self, mode: str = "candidate", place_enabled: bool = True, execute_prefix: int = 1,
                 min_steps_left_place: int = 170, max_grasp_attempts: int = 2, max_perception_calls: int = 30,
                 witness_route: str = "marked_crop", retain_packets: bool = True,
                 custody_abstain_policy: str = "measured", closed_destination_words=None,
                 triggers: Optional[Mapping[str, Any]] = None, planner: Optional[Mapping[str, Any]] = None,
                 perception: Optional[Mapping[str, Any]] = None, motion: Optional[Mapping[str, Any]] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if witness_route != "marked_crop":
            raise ValueError("only the marked_crop witness route is implemented (SAM3 not qualified)")
        if type(execute_prefix) is not int or execute_prefix < 1:
            raise ValueError("execute_prefix must be a positive int")
        self.mode, self.place_enabled, self.execute_prefix = mode, bool(place_enabled), execute_prefix
        self.min_steps_left_place = int(min_steps_left_place)
        self.max_grasp_attempts, self.max_perception_calls = int(max_grasp_attempts), int(max_perception_calls)
        self.retain_packets = bool(retain_packets)
        if custody_abstain_policy not in ("measured", "handback"):
            raise ValueError("custody_abstain_policy must be 'measured' or 'handback'")
        self.custody_abstain_policy = custody_abstain_policy
        words = DEFAULTS["closed_destination_words"] if closed_destination_words is None else closed_destination_words
        self.closed_destination_words = frozenset(str(w).lower() for w in words)
        self.trigger_params = dict(triggers or {})
        self.planner_params = dict(planner or {})
        self.perception_params = {**P.DEFAULTS, **dict(perception or {})}
        self.motion_params = dict(motion or {})
        # Validate eagerly (tier 0 builds the controller).
        T.TriggerMonitor(self.trigger_params)
        M.Motion(lambda a: None, lambda: None, lambda: True, self.motion_params)
        unknown = set(self.perception_params) - set(P.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown perception params {sorted(unknown)}")
        pp = {**PL.DEFAULTS, **self.planner_params}
        if set(pp) - set(PL.DEFAULTS):
            raise ValueError(f"unknown planner params {sorted(set(pp) - set(PL.DEFAULTS))}")
        self.sampler = scale_sampler()
        self.size = 0

    # ------------------------------------------------------------------------------------------------ episode
    async def run(self, ctx) -> None:
        self.ctx = ctx
        self.A: list = []
        self.S: list = []
        self.monitor = T.TriggerMonitor(self.trigger_params)
        self.perception_calls = 0
        first = ctx.observe()
        st0 = _state_of(first)
        scene = hashlib.sha256((ctx.instruction + "|" + (P.sha(st0) if st0 is not None else "")).encode()).hexdigest()[:16]
        # Unique per process and episode: arms / seeds of one scene never overwrite each other's evidence (P2).
        tag = f"{scene}_p{os.getpid()}_e{next(_EPISODES):04d}"
        self.tag = tag
        dirs = evidence_dirs()
        log_path = str(Path(dirs["planner"]) / f"planner_calls_p{os.getpid()}.jsonl") if dirs["planner"] else None
        self.planner = PL.PlannerSession(self.planner_params, log_path=log_path, episode_id=tag)
        self.packet_dir = dirs["packets"] if self.retain_packets else None
        self.motion = M.Motion(self._primitive_step, lambda: _state_of(ctx.observe()), lambda: ctx.ended,
                               self.motion_params)
        self.acc = {"mode": self.mode, "episode_tag": tag, "scene_tag": scene, "evidence_dirs": dirs,
                    "policy_steps": 0, "primitive_steps": 0,
                    "pause_count": 0, "pause_wall_s": 0.0, "escalations": [], "fires": [], "planner": {},
                    "perception_calls": 0, "custody": [], "places": [], "analytic_share": 0.0,
                    "sampler": self.sampler["factory"], "planner_backend": self.planner.p["backend"],
                    "planner_model": self.planner.p["model"], "planner_effort": self.planner.p["effort"]}
        ctx.accounting.extra["witnessed_rescue"] = self.acc
        try:
            while not ctx.ended:
                obs = ctx.observe()
                proposals = await ctx.policy.sample(1, sampler=self.sampler)
                chunk = np.array(proposals[0].actions)[: self.execute_prefix]
                fire = self.monitor.check(self.A, self.S, _as_recorded(chunk[0]), ctx.remaining)
                if fire is not None:
                    self.acc["fires"].append(fire.to_json())
                    ctx.record("wr_trigger", fire=fire.to_json())
                if fire is not None and fire.eligible:
                    step0 = ctx.steps
                    t0 = time.monotonic()
                    self.acc["pause_count"] += 1
                    try:
                        await self._escalate(fire, obs, chunk)
                    except Exception as exc:  # noqa: BLE001 - a rescue bug must never invalidate the episode
                        ctx.record("wr_error", where="escalation", error=f"{type(exc).__name__}: {str(exc)[:300]}")
                        self.acc.setdefault("errors", []).append(f"{type(exc).__name__}: {str(exc)[:200]}")
                    finally:
                        self.acc["pause_wall_s"] = round(self.acc["pause_wall_s"] + time.monotonic() - t0, 3)
                        if self.acc["escalations"]:
                            self.acc["escalations"][-1] = _clean(self.acc["escalations"][-1])
                            ctx.record("wr_escalation", escalation=self.acc["escalations"][-1])
                        self.monitor.handback(ctx.steps)
                        self._update()
                    if ctx.steps != step0:
                        if self.mode == "analytic" and any(c.get("custody") for c in self.acc["custody"]):
                            await self._analytic_completion()
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
        n = await self.ctx.execute(a[None, :])
        if n:
            self.A.append(_as_recorded(a))
            st = _state_of(self.ctx.observe())
            self.S.append(st.astype(np.float32).astype(np.float64) if st is not None else np.zeros(0))
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

    def _scene_images(self, packet: P.Packet) -> list:
        s = int(self.perception_params["image_scale"])
        out = []
        av = P.rgb_view(packet.obs, "agentview")
        if av is not None:
            self.size = int(av.shape[0] * s)
            out.append(P.upsample(av, s))
        wr = P.rgb_view(packet.obs, "wrist")
        if wr is not None:
            out.append(P.upsample(wr, s))
        return out

    async def _call(self, esc: dict, role: str, prompt: str, images: list, schema: dict, validate=None):
        if esc["calls"] >= int(self.planner.p["max_calls_escalation"]):
            res = PL.CallResult(role, "budget_refused", record={"error": "per-escalation planner budget spent"})
        else:
            esc["calls"] += 1
            res = await self.planner.ask(role, prompt, images, schema, validate)
        esc["planner_calls"].append(res.to_json())
        return res

    def _fresh(self, packet: P.Packet) -> bool:
        return self.ctx.steps == packet.step

    def _locate(self, esc: dict, packet: P.Packet, row, col, role: str) -> dict:
        if self.perception_calls >= self.max_perception_calls:
            return {"ok": False, "reason": "perception budget spent", "role": role}
        self.perception_calls += 1
        loc = P.locate(packet, row, col, self.perception_params, role=role)
        esc["locate"].append(_clean(loc))
        return loc

    def _handback(self, esc: dict, why: str) -> None:
        esc["handback"] = why
        esc["end_step"] = self.ctx.steps

    # ------------------------------------------------------------------------------------------------ rescue
    async def _plan(self, esc: dict, kind: str, packet: P.Packet, images: list) -> Optional[dict]:
        instr = self.ctx.instruction
        size = self.size or 512
        res = await self._call(esc, "plan", PL.plan_prompt(instr, kind, size), images, PL.PLAN_SCHEMA,
                               validate=lambda o: PL.validate_plan(o, kind, instr, size))
        if res.status != "ok":
            return None
        plan = res.obj
        if plan["action"] == "handback":
            res.outcome = "abstain"
        elif plan.get("target") and not PL.label_ok(plan["target"]["label"], instr):
            res.outcome = "invalid"
        elif kind == "H1_settled_clamp" and not (plan.get("clamped_task_object") is True
                                                  and plan.get("clamped_at_destination") is False):
            res.outcome = "abstain"
        else:
            res.outcome = "accept"
        esc["planner_calls"][-1]["outcome"] = res.outcome
        return plan if res.outcome == "accept" else None

    async def _witness(self, esc: dict, packet: P.Packet, label: str, row_n: float, col_n: float, held: bool,
                       size: Optional[Mapping[str, Any]] = None) -> bool:
        try:
            full, crop = P.marked_views(packet, row_n, col_n, self.perception_params)
        except P.PerceptionUnavailable as exc:
            esc["witness"] = f"unavailable: {exc}"
            return False
        esc["retained"].append(self._retain(packet, f"witness{len(esc['retained'])}", {"marked": full, "crop": crop}))
        res = await self._call(esc, "witness", PL.witness_prompt(self.ctx.instruction, label, held, size),
                               [full, crop], PL.VERDICT_SCHEMA)
        if res.status != "ok":
            return False
        verdict = res.obj.get("verdict")
        res.outcome = {"confirm": "accept", "wrong_object": "wrong_object"}.get(verdict, "abstain")
        esc["planner_calls"][-1]["outcome"] = res.outcome
        return res.outcome == "accept"

    async def _confirm_custody(self, esc: dict, label: str, before: P.Packet, row_n: float, col_n: float) -> bool:
        """Called only after the MEASURED custody test passed. Shows the planner the same front-camera zoom window
        before the grasp (the packet in which the target was witnessed) and now (after the lift), plus the wrist
        image only when the benchmark sends one; the prompt describes exactly the images sent.
        confirm -> proceed; not_held -> hand back; abstain -> proceed on the measured test when
        custody_abstain_policy == 'measured' (else hand back); transport/budget failures -> hand back (C5)."""
        packet = P.Packet.capture(self.ctx.observe(), self.ctx.steps)
        images, views = [], []
        try:
            images.append(P.zoom_view(before, row_n, col_n, self.perception_params))
            views.append("before")
            images.append(P.zoom_view(packet, row_n, col_n, self.perception_params))
            views.append("after")
        except P.PerceptionUnavailable:
            images, views = [], []
        wr = P.rgb_view(packet.obs, "wrist")
        if wr is not None:
            images.append(P.upsample(wr, int(self.perception_params["image_scale"])))
            views.append("wrist")
        esc["retained"].append(self._retain(packet, "custody", {f"img{i}_{v}": im for i, (im, v)
                                                                 in enumerate(zip(images, views))}))
        esc["custody_views"] = views
        if "after" not in views:
            esc["custody_confirm"] = "no_views"
            return False
        res = await self._call(esc, "custody", PL.custody_prompt(self.ctx.instruction, label, views), images,
                               PL.CUSTODY_SCHEMA)
        if res.status != "ok":
            esc["custody_confirm"] = f"transport:{res.status}"
            return False
        verdict = res.obj.get("verdict")
        res.outcome = {"confirm": "accept", "not_held": "not_held"}.get(verdict, "abstain")
        esc["planner_calls"][-1]["outcome"] = res.outcome
        if res.outcome == "abstain" and self.custody_abstain_policy == "measured":
            esc["custody_confirm"] = "abstain_proceed_on_measured"
            return True
        esc["custody_confirm"] = res.outcome
        return res.outcome == "accept"

    async def _escalate(self, fire: T.Fire, obs: Mapping[str, Any], chunk: np.ndarray) -> None:
        ctx = self.ctx
        kind = fire.kind
        esc = {"kind": kind, "step": ctx.steps, "steps_left": ctx.remaining, "calls": 0, "planner_calls": [],
               "locate": [], "retained": [], "grasps": [], "custody": None, "place": None, "handback": None}
        self.acc["escalations"].append(esc)
        place_time_ok = ctx.remaining >= self.min_steps_left_place
        packet = P.Packet.capture(obs, ctx.steps)
        images = self._scene_images(packet)
        esc["retained"].append(self._retain(packet, "plan", {f"img{i}": im for i, im in enumerate(images)}))
        plan = await self._plan(esc, kind, packet, images)
        if plan is None:
            return self._handback(esc, "plan_not_accepted")
        esc["plan_action"] = plan["action"]
        target = plan["target"]
        held = kind == "H1_settled_clamp"
        st = packet.state()
        if held:
            try:
                _, _, cam = P.views(packet.obs, "agentview")
            except P.PerceptionUnavailable as exc:
                return self._handback(esc, f"perception_unavailable: {exc}")
            row_n, col_n = P.project(st[:3], cam)
            loc = None
            z_table = P.packet_table_z(packet, self.perception_params)
            esc["z_table"] = z_table
        else:
            loc = self._locate(esc, packet, target["row"], target["col"], "target")
            if not loc.get("ok"):
                esc["planner_calls"][-1]["outcome"] = "wrong_object"   # geometric witness rejection of the point
                esc["planner_calls"][-1]["geometry_reject"] = loc.get("reason")
                if self.mode == "respondoff":
                    return self._handback(esc, "respondoff")
                return self._handback(esc, f"witness_geometry: {loc.get('reason')}")
            row_n, col_n = loc["pixel_native"]
        if not await self._witness(esc, packet, target["label"], row_n, col_n, held, size=loc):
            return self._handback(esc, "witness_not_confirmed")
        dest_loc = None
        dest = plan.get("destination")
        if dest is not None:
            dest_loc = self._locate(esc, packet, dest["row"], dest["col"], "destination")
        if self.mode == "respondoff":
            return self._handback(esc, "respondoff")      # responses suppressed: the SCALE proposal is executed
        if not self._fresh(packet):
            return self._handback(esc, "stale_packet")

        # ---- physical response -------------------------------------------------------------------------------
        if held:
            cust = await self.motion.lift_and_test(self.motion.width())
            cust["attempt"] = 1
            esc["grasps"].append(_clean(cust))
            # The clamp holds the object at (eef z - table z) above its base; with h_obj = 2 x that offset the
            # reviewed place height z_surface + h_obj/2 + 0.02 puts the base 2 cm above the destination support.
            h_obj = 2.0 * (float(st[2]) - z_table) if z_table is not None else None
        else:
            cust = None
            for attempt in range(1, self.max_grasp_attempts + 1):
                g = await self.motion.grasp(loc)
                g["attempt"] = attempt
                esc["grasps"].append(_clean(g))
                cust = g
                if g["custody"] or ctx.ended or attempt == self.max_grasp_attempts:
                    break
                # one re-witnessed retry on a fresh packet
                packet = P.Packet.capture(ctx.observe(), ctx.steps)
                images = self._scene_images(packet)
                esc["retained"].append(self._retain(packet, "replan", {f"img{i}": im for i, im in enumerate(images)}))
                plan2 = await self._plan(esc, "H2_empty_close", packet, images)
                if plan2 is None or plan2["action"] == "handback":
                    break
                target = plan2["target"]
                loc = self._locate(esc, packet, target["row"], target["col"], "target")
                if not loc.get("ok"):
                    esc["planner_calls"][-1]["outcome"] = "wrong_object"
                    esc["planner_calls"][-1]["geometry_reject"] = loc.get("reason")
                    break
                if not await self._witness(esc, packet, target["label"], *loc["pixel_native"], held=False, size=loc):
                    break
                if plan2.get("destination") is not None:
                    dest = plan2["destination"]
                    dest_loc = self._locate(esc, packet, dest["row"], dest["col"], "destination")
                plan = plan2
            h_obj = float(loc["height"]) if loc and loc.get("height") is not None else None
        esc["custody"] = bool(cust and cust.get("custody"))
        self.acc["custody"].append({"step": esc["step"], "kind": kind, "custody": esc["custody"],
                                    "measured": {k: cust.get(k) for k in ("width_after_close", "width_change_lift",
                                                                           "rise")} if cust else None})
        if not esc["custody"] or ctx.ended:
            return self._handback(esc, "no_custody" if not ctx.ended else "episode_ended")

        wants_place = plan["action"] in ("grasp_and_place", "lift_and_place")
        if not wants_place:
            return self._handback(esc, "handback_after_custody")
        why = None
        if not self.place_enabled:
            why = "place_disabled"
        elif not place_time_ok:
            why = "too_few_steps_for_place"
        elif dest is None or dest_loc is None or not dest_loc.get("ok"):
            why = "no_destination"
        elif not dest.get("open_top"):
            why = "destination_not_open_top"
        elif self.closed_destination_words & set(re.findall(r"[a-z]+", str(dest.get("label", "")).lower())):
            why = "destination_enclosed"     # K1: code guard independent of the planner's open_top
        if why:
            return self._handback(esc, why)
        if h_obj is None:
            return self._handback(esc, "object_height_unknown")
        before = (packet, row_n, col_n) if held else (packet, *loc["pixel_native"])
        if not await self._confirm_custody(esc, target["label"], *before):
            return self._handback(esc, "custody_not_confirmed")
        now = P.Packet.capture(ctx.observe(), ctx.steps)
        col = P.free_column(now, dest_loc, h_obj, now.state()[:3], self.perception_params)
        esc["free_column"] = col
        if not col["free"]:
            return self._handback(esc, "column_unobserved" if col["reason"] == "column_unobserved"
                                  else "destination_column_occupied")
        pl = await self.motion.place(dest_loc, h_obj)
        esc["place"] = _clean(pl)
        self.acc["places"].append({"step": esc["step"], "dest_label": dest.get("label"),
                                   "width_before_release": pl.get("width_before_release"),
                                   "column_points": col["column_points"], "occupied_points": col["occupied_points"]})
        return self._handback(esc, "after_place")

    async def _analytic_completion(self) -> None:
        """Primitives-only ablation: after the first custody, keep asking for the next object to move and move it
        analytically (open-top only); otherwise hold still. No hand-back to the policy."""
        ctx = self.ctx
        while not ctx.ended:
            if self.planner.remaining <= 0:
                break
            obs = ctx.observe()
            fire = T.Fire("analytic_continue", ctx.steps, ctx.steps, {}, True, ctx.remaining)
            step0 = ctx.steps
            try:
                await self._escalate(fire, obs, np.zeros((1, 7)))
            finally:
                self.acc["escalations"][-1] = _clean(self.acc["escalations"][-1])
                ctx.record("wr_escalation", escalation=self.acc["escalations"][-1])
                self._update()
            if ctx.steps == step0:
                break
        while not ctx.ended:
            await self.motion.hold(-1.0, 1)
