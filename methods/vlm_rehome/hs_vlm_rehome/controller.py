"""VLM-flagged post-boundary home re-entry (VPR; r4-perception P1; review r4-proposal-review.json C-V1..C-V10; A4).

A HarnessStudio v2 controller for openpi pi0_fast_libero on vla-eval LIBERO (null: shadow).

Per loop iteration at a decision start (step t, public observation obs, state s = obs['states']):
  1. If a scheduled post-boundary check is due (t >= boundary_step + offset, offsets 10/30/50/80, all overdue offsets of
     that boundary consumed by one check), the gripper is empty or open (width > empty_width or the last executed
     gripper command <= 0), the boundary has not been re-homed, its checks were not stopped by a new grasp, and the
     episode still has VLM calls (max_calls) and re-homes (max_rehomes) left:
       - gate 'vlm' (candidate): ask the frozen judge (hs_vlm_rehome.judge) about obs['images']['agentview'] with the
         ORIGINAL instruction; re-home iff the verdict is 'abnormal' (reply within 60 s, strict JSON). Anything else
         (normal, unsure, invalid JSON, timeout, HTTP/other error) abstains.
       - gate 'tilt' (admission control 'vlm-rehome-tiltgate', C-V5): the same judge call is made and recorded but
         ignored; re-home iff tilt(s) > tilt_deg (55, inherited from B).
       - gate 'forced' (dev-only diagnostic 'vlm-rehome-forced', C-V6): no VLM call; re-home at the first eligible
         check of the first boundary.
     Re-home: single-step ctx.execute of the deterministic proportional law (rehome.rehome_action) toward the episode's
     step-0 public pose (lift first to max(z, z0), then p0; world-frame log-map rotation; gripper open) until the pose
     is within 0.02 m / 10 deg or rehome_max_steps steps; then control returns to the native policy (original
     instruction, unmodified observations).
  2. Otherwise one native decision: `await ctx.policy.sample(1)` (seeded sample 0 = the baseline arm's call, no
     override) and `ctx.execute(chunk[:exec_rows])` (exec_rows = 5 = profile execute_horizon), feeding the boundary
     detector with the chunk's first gripper command, the decision-time finger width and the zero-chunk test.

Public inputs only (images.agentview, states, instruction); no simulator state, reward, success, seed, case id or arm.
No RNG. Every verdict is recorded with its input hashes and latency (C-V9); a per-episode reply cache keyed by the
prompt and image hashes avoids paying twice for an identical input. The served model id of a deployed reply is not
visible through ctx.tools['llm'] (framework limitation): the controller checks the pinned model against /models at
episode start and stops (infrastructure, episode finishes natively) when it is not listed; per-reply ids are
recorded by Q1 only. The framework operator's HTTP timeout is 120 s; a reply after 60 s is a timeout abstention.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import judge
from .rehome import (BoundaryDetector, home_reached, is_zero_chunk, rehome_action, state_vec, tilt_deg)

PKG = Path(__file__).resolve().parent
GATES = ("vlm", "tilt", "forced")


def _load_action_mean(path: str, sha256: Optional[str]) -> np.ndarray:
    p = Path(path)
    p = p if p.is_absolute() else PKG / p
    data = p.read_bytes()
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"norm stats {path}: sha256 mismatch")
    ns = json.loads(data)
    ns = ns.get("norm_stats", ns)
    return np.asarray(ns["actions"]["mean"], dtype=np.float64)


def _transport_class(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, TimeoutError) or "timed out" in text.lower() or "timeout" in type(exc).__name__.lower():
        return "timeout"
    if "HTTP" in text:
        return "http_error"
    return "error"


class VLMRehome:
    # tier 1 (no network): scripted replies cycle through every verdict class
    TIER1_LLM_RESPONSES = ['{"state": "normal"}', '{"state": "abnormal"}', "garbage", '{"state": "unsure"}']

    def __init__(self, gate: str = "vlm", boundary_min_close: int = 4, hold_width_m: float = 0.004,
                 empty_width_m: float = 0.06, max_boundaries: int = 2, check_offsets_steps: Sequence[int] = (10, 30, 50, 80),
                 close_stop_decisions: int = 2, max_calls: int = 8, max_rehomes: int = 2, rehome_tol_m: float = 0.02,
                 rehome_tol_deg: float = 10.0, rehome_max_steps: int = 60, vlm_timeout_s: float = 60.0,
                 kp: float = 1.0, kr: float = 1.0, pos_scale: float = 0.05, rot_scale: float = 0.5,
                 lift_first: bool = True, lift_tol_m: float = 0.02, gripper_open: float = -1.0, tilt_gate_deg: float = 55.0,
                 exec_rows: int = 5, zero_chunk_tol: float = 1e-4, image_key: str = "agentview", state_key: str = "states",
                 operator: str = "judge", models_check: bool = True,
                 norm_stats: str = "assets/pi0_fast_libero_norm_stats.json", norm_stats_sha256: Optional[str] = None,
                 judge_sha256: str = "") -> None:
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}")
        if float(vlm_timeout_s) != judge.TIMEOUT_S:
            raise ValueError("vlm_timeout_s must equal the frozen judge timeout")
        if judge_sha256:
            got = hashlib.sha256((PKG / "judge.py").read_bytes()).hexdigest()
            if got != judge_sha256:
                raise RuntimeError(f"judge.py sha256 {got} != frozen {judge_sha256} (Q1-qualified judge required)")
        self.gate = gate
        self.min_close, self.hold_width, self.empty_width = int(boundary_min_close), float(hold_width_m), float(empty_width_m)
        self.max_boundaries = int(max_boundaries)
        self.offsets = [int(o) for o in check_offsets_steps]
        self.close_stop = int(close_stop_decisions)
        self.max_calls, self.max_rehomes = int(max_calls), int(max_rehomes)
        self.tol_m, self.tol_deg, self.max_steps = float(rehome_tol_m), float(rehome_tol_deg), int(rehome_max_steps)
        self.kp, self.kr, self.pos_scale, self.rot_scale = float(kp), float(kr), float(pos_scale), float(rot_scale)
        self.lift_first, self.lift_tol, self.gripper_open = bool(lift_first), float(lift_tol_m), float(gripper_open)
        self.tilt_gate = float(tilt_gate_deg)
        self.exec_rows, self.zero_tol = int(exec_rows), float(zero_chunk_tol)
        self.image_key, self.state_key, self.operator = image_key, state_key, operator
        self.models_check = bool(models_check)
        self.action_mean = _load_action_mean(norm_stats, norm_stats_sha256)
        # per-episode state (components are built per episode)
        self.detector = BoundaryDetector(self.min_close, self.hold_width, self.max_boundaries)
        self.boundaries: list = []
        self.s0: Optional[np.ndarray] = None
        self.last_cmd: Optional[float] = None
        self.calls = 0
        self.rehomes = 0
        self.cache: dict = {}
        self.stats = {"decisions": 0, "boundaries": 0, "checks": 0, "llm_calls": 0, "cached": 0, "verdicts": {},
                      "rehomes": 0, "rehome_reached": 0, "rehome_steps": 0, "skipped_offsets": 0}

    # ---- helpers ------------------------------------------------------------------------------------------------
    def empty_or_open(self, s: np.ndarray) -> bool:
        width = float(s[6] - s[7])
        return width > self.empty_width or (self.last_cmd is not None and self.last_cmd <= 0)

    def _real_operator(self, ctx):
        tool = ctx.tools.get("llm")
        ops = getattr(tool, "_ops", None) or {}
        op = ops.get(self.operator)
        return op if op is not None and hasattr(op, "base_url") else None

    async def _check_models(self, ctx) -> bool:
        """Per-episode pin check (coordinator decision for framework gap 1): the pinned model must be listed."""
        op = self._real_operator(ctx)
        if op is None:
            return True     # scripted tier-1 operator or no operator: nothing to check
        import anyio

        def fetch():
            key = os.environ.get(op.api_key_env, "")
            req = urllib.request.Request(op.base_url + "/models", method="GET",
                                         headers={"Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=judge.TIMEOUT_S) as r:
                return json.loads(r.read().decode()).get("data", [])
        try:
            data = await anyio.to_thread.run_sync(fetch)
        except Exception as exc:  # noqa: BLE001
            ctx.record("vpr_infrastructure", what="models_check_failed", error=_transport_class(exc))
            ctx.stop("vpr infrastructure: /models check failed")
            return False
        ids = sorted(str(m.get("id")) for m in data)
        entry = next((m for m in data if m.get("id") == op.model), None)
        meta = {k: entry.get(k) for k in ("id", "root", "owned_by", "parent") if entry and k in entry} if entry else None
        ctx.record("vpr_models", ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(), pinned=op.model,
                   listed=entry is not None, entry=meta)
        if entry is None:
            ctx.stop("vpr infrastructure: pinned model not listed by /models")
            return False
        return True

    async def _ask(self, ctx, obs) -> dict:
        images = obs.get("images") or {}
        if self.image_key not in images:
            return {"verdict": "error", "detail": "no agent-view image", "called": False}
        try:
            png = judge.image_png(images[self.image_key])
        except ValueError as exc:
            return {"verdict": "error", "detail": str(exc), "called": False}
        msgs = judge.build_messages(ctx.instruction, png)
        psha, isha = judge.prompt_sha256(msgs), judge.sha256(png)
        key = psha + ":" + isha
        if key in self.cache:
            self.stats["cached"] += 1
            return {**self.cache[key], "cached": True, "called": False}
        self.calls += 1
        self.stats["llm_calls"] += 1
        text, err, t0 = None, None, time.monotonic()
        try:
            text = await ctx.tools["llm"](msgs, operator=self.operator, max_tokens=judge.MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001  (budget, HTTP, timeout, env): abstain
            err = _transport_class(exc)
        secs = round(time.monotonic() - t0, 3)
        verdict = judge.classify(text, secs, err)
        rec = {"verdict": verdict, "seconds": secs, "png_sha256": isha, "prompt_sha256": psha,
               "reply": text if text is None or len(text) <= 200 else text[:200], "cached": False, "called": True}
        self.cache[key] = {k: rec[k] for k in ("verdict", "seconds", "png_sha256", "prompt_sha256", "reply")}
        return rec

    async def _rehome(self, ctx, bidx: int) -> None:
        self.rehomes += 1
        self.stats["rehomes"] += 1
        p0, aa0 = self.s0[:3], self.s0[3:6]
        s = state_vec(ctx.observe(), self.state_key)
        start_err = (float(np.linalg.norm(p0 - s[:3])), None)
        from .rehome import geodesic_deg
        ctx.record("vpr_rehome_start", boundary=bidx, step=ctx.steps, pos_err_m=start_err[0],
                   rot_err_deg=geodesic_deg(aa0, s[3:6]), tilt_deg=tilt_deg(s[3:6]))
        steps, reached, errs = 0, False, []
        while not ctx.ended:
            s = state_vec(ctx.observe(), self.state_key)
            if s is None:
                break
            errs.append((round(float(np.linalg.norm(p0 - s[:3])), 4), round(geodesic_deg(aa0, s[3:6]), 2)))
            if home_reached(s[:3], s[3:6], p0, aa0, self.tol_m, self.tol_deg):
                reached = True
                break
            if steps >= self.max_steps:
                break
            a, lifting = rehome_action(s[:3], s[3:6], p0, aa0, z_first=self.lift_first, kp=self.kp, kr=self.kr,
                                       pos_scale=self.pos_scale, rot_scale=self.rot_scale,
                                       gripper_open=self.gripper_open, lift_tol=self.lift_tol)
            n = await ctx.execute(a[None, :])
            if n == 0:
                break
            steps += n
            self.last_cmd = self.gripper_open
        self.stats["rehome_steps"] += steps
        self.stats["rehome_reached"] += int(reached)
        ctx.record("vpr_rehome_end", boundary=bidx, step=ctx.steps, steps=steps, reached=reached,
                   errors=errs[:: max(1, len(errs) // 20)] + errs[-1:], stats=dict(self.stats))

    def _due(self, t: int):
        """(boundary index, consumed offsets) of the most recent boundary with an overdue check, else None."""
        for i in range(len(self.boundaries) - 1, -1, -1):
            b = self.boundaries[i]
            if b["stopped"] or b["rehomed"]:
                continue
            due = [o for o in b["offsets_left"] if t >= b["step"] + o]
            if due:
                b["offsets_left"] = [o for o in b["offsets_left"] if o not in due]
                self.stats["skipped_offsets"] += len(due) - 1
                return i, due
        return None

    # ---- episode ----------------------------------------------------------------------------------------------
    async def run(self, ctx) -> None:
        if self.gate != "forced" and self.models_check and not await self._check_models(ctx):
            return
        while not ctx.ended:
            obs = ctx.observe()
            s = state_vec(obs, self.state_key)
            t = ctx.steps
            if s is not None and self.s0 is None:
                self.s0 = s.copy()      # the episode's step-0 public pose
                ctx.record("vpr_home", step=t, pos=[float(x) for x in s[:3]], aa=[float(x) for x in s[3:6]])
            if s is not None and self.s0 is not None:
                due = self._due(t)
                if due is not None:
                    bidx, offs = due
                    acted = await self._maybe_check(ctx, obs, s, t, bidx, offs)
                    if acted:
                        continue
            await self._decision(ctx, s, t)

    async def _maybe_check(self, ctx, obs, s, t, bidx, offs) -> bool:
        b = self.boundaries[bidx]
        empty = self.empty_or_open(s)
        tilt = tilt_deg(s[3:6])
        rec = {"boundary": bidx, "step": t, "offsets": offs, "gate": self.gate, "empty_or_open": empty,
               "tilt_deg": round(tilt, 2), "calls_used": self.calls, "rehomes_used": self.rehomes}
        eligible = empty and self.rehomes < self.max_rehomes
        if self.gate != "forced":
            eligible = eligible and self.calls < self.max_calls
        if not eligible:
            ctx.record("vpr_check", **rec, checked=False)
            return False
        self.stats["checks"] += 1
        act = False
        if self.gate == "forced":
            act = bidx == 0 and not b.get("forced_done")
            b["forced_done"] = True
            rec["verdict"] = None
        else:
            r = await self._ask(ctx, obs)
            v = r["verdict"]
            self.stats["verdicts"][v] = self.stats["verdicts"].get(v, 0) + 1
            rec.update({k: r.get(k) for k in ("verdict", "seconds", "png_sha256", "prompt_sha256", "reply", "cached",
                                              "called")})
            act = judge.flags(v) if self.gate == "vlm" else tilt > self.tilt_gate
        rec["act"] = act
        rec["stats"] = {**self.stats, "verdicts": dict(self.stats["verdicts"])}
        ctx.record("vpr_check", **rec, checked=True)
        if act:
            b["rehomed"] = True
            ctx.record("vpr_flag", boundary=bidx, step=t, gate=self.gate)
            await self._rehome(ctx, bidx)
            return True
        return False

    async def _decision(self, ctx, s, t) -> int:
        self.stats["decisions"] += 1
        chunk = np.array((await ctx.policy.sample(1))[0].actions)
        if s is not None and chunk.ndim == 2 and chunk.shape[1] == self.action_mean.size:
            cmd = float(chunk[0, 6])
            width = float(s[6] - s[7])
            zc = is_zero_chunk(chunk, s, self.action_mean, self.zero_tol)
            for b in self.boundaries:     # a new grasp after a boundary stops its checks (a carry is never interrupted)
                if b["stopped"]:
                    continue
                b["close_run"] = b["close_run"] + 1 if (cmd > 0 and self.hold_width < width < self.empty_width) else 0
                if b["close_run"] >= self.close_stop:
                    b["stopped"] = True
            if self.detector.update(cmd, width, zc):
                self.stats["boundaries"] += 1
                self.boundaries.append({"step": t, "offsets_left": list(self.offsets), "rehomed": False,
                                        "stopped": False, "close_run": 0})
                ctx.record("vpr_boundary", index=len(self.boundaries) - 1, step=t, width=width)
        n = await ctx.execute(chunk[: self.exec_rows])
        if n > 0:
            self.last_cmd = float(chunk[n - 1, 6])
        return n
