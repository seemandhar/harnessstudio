"""PRR 1.1.0: Pose-Restore-and-Redecide with the C7-revised correction phase (rtcfpp-20261101 round-6 RECOVERY P1;
review r6-recovery-review.json C1-C11; ruling prr-c6-c7-ruling.json: the single W1 technical revision).

ONLY the correction phase differs from hs-prr 1.0.0 (trigger, checkpoint, custody, horizon formula, reversal, settle,
hand-back, re-arm, controls, coin and every other scalar are unchanged; planner.py and geometry.py are byte-identical):
  1.0.0  a_pos = clip((p_cp - p) / 0.0054), a_rot = clip(log(R_cp R^T) / 0.0324) per element, 2 decisions;
  1.1.0  u_pos = (Kp_pos (p_cp - p) - Kd_pos v) / 0.05, u_rot = (Kp_rot log(R_cp R^T) - Kd_rot w) / 0.5,
         v = p - p_prev and w = log(R R_prev^T) from the public states of consecutive steps, each group saturated
         direction-preservingly (u / max(1, max|u_i|)); Kp 1 (the OSC goal is the checkpoint pose: robosuite OSC_POSE
         goal = current + 0.05 a / 0.5 a rad), Kd 2 steps (pos) / 4 steps (rot); budget 10 decisions (50 steps).
  Stop tolerance (1 cm / 5 deg, checked at each correction-decision start) and W1 (2 cm / 10 deg) unchanged; the C4
  guard keeps its formula with the new budget (abstain if f + L + 5 * 10 + 5 > 460).

A HarnessStudio v2 controller (null: shadow) for openpi pi0_fast_libero on vla-eval LIBERO.

Native decisions: ``await ctx.policy.sample(1)`` (seeded sample 0 = the baseline arm's call; no override) and the first
``exec_rows`` (5 = profile execute_horizon) rows executed one row per ``ctx.execute`` call, so the controller sees the
pre-row public state of every executed row (custody window, C2). Before the first restoration the executed actions are
bitwise the baseline arm's (same payload, same seed index, same rows). The trigger/checkpoint/guard state machine is
``hs_prr.planner`` (also used by scripts/replay_table.py to generate the C3 tables).

Restoration (variant):
  prr   (i) reversal: the planner's rows (rows [cp, f) reversed, dims 0-5 negated, gripper = latch);
        (ii) correction (REVISED in 1.1.0): up to 10 decisions of 5 steps; at each correction-decision start the pose
             error to the checkpoint (||p_cp - p|| and the world-frame log-map angle of R_cp R^T) is checked and the
             phase ends when it is < 1 cm AND < 5 deg; otherwise 5 per-step damped rows recomputed from the current and
             previous public states (law above), gripper latch;
        (iii) settle: 5 rows of zero motion with the latched gripper; then control returns to native decisions.
  hold  zero-motion rows (dims 0-5 = 0, gripper latch) for the L reversal rows and for every correction step; same
        trigger, checkpoint, latch, correction stopping rule (on the actual pose) and settle.
  ortho reversal rows equal PRR's with dims 0-2 rotated by s*90 deg about world z, s = +1 if the public coin
        int(sha256(coin_input), 16) is even else -1, coin_input = 'prr-ortho|<instruction>|<initial public state
        digest>|<fire step>' (DISCLOSED DEVIATION D1: the review's 'task|init' uses the case index, which methods may not
        read; the instruction and a digest of the step-0 public state identify the scene from public inputs); the
        correction uses PRR's rotation law with position rows zero, same stopping rule; settle identical.
Restoration rows make no policy call and count against the official horizon; ctx.execute returning 0 ends the phase.
Public inputs only (states, instruction); no simulator state, reward, success, seed, case id or arm; no RNG.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from dataclasses import asdict

from .geometry import rot_error, rot_z, tilt_deg
from .planner import Params, Planner

PKG = Path(__file__).resolve().parent
VARIANTS = ("prr", "hold", "ortho")


def _state(obs, key: str) -> Optional[np.ndarray]:
    v = obs.get(key) if isinstance(obs, dict) else None
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return arr if arr.size >= 8 else None


def state_digest(s0: np.ndarray) -> str:
    return hashlib.sha256(",".join(f"{float(x):.6f}" for x in np.asarray(s0)[:8]).encode()).hexdigest()


def ortho_coin(instruction: str, s0: np.ndarray, fire_step: int):
    text = f"prr-ortho|{instruction}|{state_digest(s0)}|{int(fire_step)}"
    h = hashlib.sha256(text.encode()).hexdigest()
    return (1 if int(h, 16) % 2 == 0 else -1), text, h


def pose_error(cp_state, st) -> tuple:
    e_p = float(np.linalg.norm(np.asarray(cp_state[:3], float) - np.asarray(st[:3], float)))
    e_r = float(np.degrees(np.linalg.norm(rot_error(cp_state[3:6], st[3:6]))))
    return e_p, e_r


class PRR:
    def __init__(self, variant: str = "prr", exec_rows: int = 5, state_key: str = "states",
                 correction_kp_pos: float = 1.0, correction_kd_pos: float = 2.0, correction_kp_rot: float = 1.0,
                 correction_kd_rot: float = 4.0, correction_pos_scale: float = 0.05, correction_rot_scale: float = 0.5,
                 correction_tol_m: float = 0.01, correction_tol_deg: float = 5.0, record_states: bool = True,
                 norm_stats: str = "assets/pi0_fast_libero_norm_stats.json", norm_stats_sha256: Optional[str] = None,
                 **planner_params) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not (correction_pos_scale > 0 and correction_rot_scale > 0):
            raise ValueError("correction scales must be positive")
        self.variant = variant
        self.exec_rows, self.state_key = int(exec_rows), str(state_key)
        self.kp_pos, self.kd_pos = float(correction_kp_pos), float(correction_kd_pos)
        self.kp_rot, self.kd_rot = float(correction_kp_rot), float(correction_kd_rot)
        self.pos_scale, self.rot_scale = float(correction_pos_scale), float(correction_rot_scale)
        self._prev_state: Optional[np.ndarray] = None   # pre-row state of the most recently executed row
        self.tol_m, self.tol_deg = float(correction_tol_m), float(correction_tol_deg)
        self.record_states = bool(record_states)
        self.params = Params.from_mapping(planner_params)
        unknown = set(planner_params) - set(Params.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown PRR parameters {sorted(unknown)}")
        # pinned norm stats: the checkpoint rx bound is the served state q99 of dim 3 (3.2705)
        p = Path(norm_stats)
        p = p if p.is_absolute() else PKG / p
        data = p.read_bytes()
        if norm_stats_sha256 and hashlib.sha256(data).hexdigest() != norm_stats_sha256:
            raise RuntimeError("norm stats sha256 mismatch")
        ns = json.loads(data)
        q99 = float(ns.get("norm_stats", ns)["state"]["q99"][3])
        if abs(q99 - self.params.checkpoint_rx_max) > 1e-4:
            raise RuntimeError(f"checkpoint_rx_max {self.params.checkpoint_rx_max} != pinned state q99[3] {q99}")
        self.planner = Planner(self.params)
        self.s0: Optional[np.ndarray] = None
        self.stats = {"decisions": 0, "restorations": 0, "restoration_steps": 0, "abstains": 0}

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _flush(self, ctx) -> None:
        for name, fields in self.planner.drain():
            ctx.record(name, **fields)

    async def _row(self, ctx, row, dtype) -> int:
        """Execute one row, feeding the planner with its pre-row state. Returns 0 when the episode ended."""
        if ctx.ended:
            return 0
        pre = _state(ctx.observe(), self.state_key)
        step = ctx.steps
        r = np.asarray(row, dtype=dtype).reshape(1, -1)
        n = await ctx.execute(r)
        if n > 0:
            self.planner.note_row(step, pre, r[0])
            self._prev_state = pre
        return n

    async def _decision(self, ctx) -> None:
        self.stats["decisions"] += 1
        chunk = (await ctx.policy.sample(1))[0].actions
        chunk = np.asarray(chunk)
        self._dtype = chunk.dtype
        for i in range(min(self.exec_rows, len(chunk))):
            if await self._row(ctx, chunk[i], chunk.dtype) == 0:
                return

    @staticmethod
    def _sat(u: np.ndarray) -> np.ndarray:
        """Direction-preserving saturation into the OSC input box [-1, 1]^3."""
        m = float(np.max(np.abs(u)))
        return u / m if m > 1.0 else u

    def _correction_row(self, cp_state, st, prev, latch) -> np.ndarray:
        a = np.zeros(7)
        prev = st if prev is None else prev
        if self.variant == "prr":
            v = st[:3] - prev[:3]
            u = (self.kp_pos * (np.asarray(cp_state[:3]) - st[:3]) - self.kd_pos * v) / self.pos_scale
            a[:3] = self._sat(u)
        if self.variant in ("prr", "ortho"):
            w = rot_error(st[3:6], prev[3:6])           # log(R R_prev^T): last step's world-frame rotation
            u = (self.kp_rot * rot_error(cp_state[3:6], st[3:6]) - self.kd_rot * w) / self.rot_scale
            a[3:6] = self._sat(u)
        a[6] = latch
        return a

    async def _restore(self, ctx, plan) -> None:
        p = self.params
        dtype = getattr(self, "_dtype", np.float32)
        cp_state, latch, L = np.asarray(plan.checkpoint_state, float), float(plan.latch), int(plan.L)
        st = _state(ctx.observe(), self.state_key)
        e0 = pose_error(cp_state, st)
        rows = np.array(plan.rows, dtype=np.float64)
        coin = None
        if self.variant == "hold":
            rows = np.zeros((L, 7)); rows[:, 6] = latch
        elif self.variant == "ortho":
            sign, text, h = ortho_coin(ctx.instruction, self.s0, plan.fire)
            rows[:, :3] = rot_z(rows[:, :3], sign)
            coin = {"sign": sign, "input": text, "sha256": h}
        k = self.stats["restorations"]
        ctx.record("prr_restore_start", index=k, variant=self.variant, route=plan.route, fire=plan.fire,
                   checkpoint=plan.checkpoint, L=L, latch=latch, step=ctx.steps, pos_err_m=round(e0[0], 5),
                   rot_err_deg=round(e0[1], 3), tilt_deg=round(tilt_deg(st[3:6]), 3), coin=coin,
                   rows_sha256=hashlib.sha256(np.ascontiguousarray(rows.astype(dtype)).tobytes()).hexdigest())
        steps = 0
        for row in rows:
            n = await self._row(ctx, row, dtype)
            if n == 0:
                ctx.record("prr_restore_cut", index=k, step=ctx.steps, phase="reversal")
                return
            steps += n
        st = _state(ctx.observe(), self.state_key)
        er = pose_error(cp_state, st)
        ctx.record("prr_reversal_end", index=k, step=ctx.steps, pos_err_m=round(er[0], 5), rot_err_deg=round(er[1], 3))
        corr = []
        for j in range(p.correction_max_decisions):
            st = _state(ctx.observe(), self.state_key)
            ep, eg = pose_error(cp_state, st)
            corr.append([ctx.steps, round(ep, 5), round(eg, 3)])
            if ep < self.tol_m and eg < self.tol_deg:
                break
            for _ in range(p.decision_steps):
                st = _state(ctx.observe(), self.state_key)
                n = await self._row(ctx, self._correction_row(cp_state, st, self._prev_state, latch), dtype)
                if n == 0:
                    ctx.record("prr_restore_cut", index=k, step=ctx.steps, phase="correction")
                    return
                steps += n
        settle = np.zeros(7); settle[6] = latch
        for _ in range(p.settle_steps):
            n = await self._row(ctx, settle, dtype)
            if n == 0:
                ctx.record("prr_restore_cut", index=k, step=ctx.steps, phase="settle")
                return
            steps += n
        st = _state(ctx.observe(), self.state_key)
        eh = pose_error(cp_state, st)
        h = ctx.steps
        self.stats["restorations"] += 1
        self.stats["restoration_steps"] += steps
        ctx.record("prr_handback", index=k, variant=self.variant, route=plan.route, step=h, steps=steps,
                   correction_checks=corr, pos_err_m=round(eh[0], 5), rot_err_deg=round(eh[1], 3),
                   tilt_deg=round(tilt_deg(st[3:6]), 3), w1_within_2cm_10deg=bool(eh[0] <= 0.02 and eh[1] <= 10.0),
                   stats=dict(self.stats))
        self.planner.handback(h)
        self._flush(ctx)

    # ---- episode --------------------------------------------------------------------------------------------------
    async def run(self, ctx) -> None:
        ctx.record("prr_config", variant=self.variant, params=asdict(self.params), correction={"law": "pd_goal_vector_sat", "kp_pos": self.kp_pos,
                   "kd_pos": self.kd_pos, "kp_rot": self.kp_rot, "kd_rot": self.kd_rot, "pos_scale": self.pos_scale,
                   "rot_scale": self.rot_scale}, exec_rows=self.exec_rows)
        while not ctx.ended:
            obs = ctx.observe()
            st = _state(obs, self.state_key)
            d = ctx.steps
            if st is not None:
                if self.s0 is None:
                    self.s0 = st.copy()
                if self.record_states:
                    ctx.record("prr_d", step=d, state=[float(x) for x in st[:8]])
                plan = self.planner.decide(d, st)
                self._flush(ctx)
                if plan is not None and plan.abstain is not None:
                    self.stats["abstains"] += 1
                if plan is not None and plan.abstain is None:
                    await self._restore(ctx, plan)
                    continue
            await self._decision(ctx)
