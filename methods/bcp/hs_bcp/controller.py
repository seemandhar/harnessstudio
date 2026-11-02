"""H1-BCP controller (HarnessStudio v2): Boundary Clause Prompt for openpi pi0_fast_libero on vla-eval LIBERO-10.

Hypothesis: runs/rtcfpp-20261101/ideation/r8-task.json (H1-BCP, scientist afba071c). Approval (K1 only, with
conditions K1-c1..K1-c6; K2 conditional): runs/rtcfpp-20261101/reviews/r8-review.json (critic a182f0a2).

Every decision (cadence ``exec_rows`` = 5 = the pi0-FAST profile's execute_horizon) the controller makes ONE
native-style seeded call ``ctx.policy.sample(1)`` without overrides, executes its rows 0..4 one at a time with
``ctx.execute`` and feeds each executed row plus the post-step public state to the custody tracker (rules.py).
In the candidate arm that call is the baseline's own request (same raw observation, same CRN seed of the same
decision index), so the arm is bitwise native until the method changes the instruction. In the null arm (shadow)
the call goes to the controller's own shadow session and ``ctx.execute`` executes the native action instead, so the
call is a REPLAY of the native decode; its fingerprint is logged per decision and the K1 checker binds it to the
framework's ``native`` event of the same step (greedy FAST: equal on a deterministic server).

Modes (param ``mode``):
  shadow_probe   K1. After the first custody boundary, at each of the next ``window_decisions`` (10) decisions also
                 decode, on the same observation, the clause prompt and the critic's surface control prompt
                 (original + ' .'), log both chunks and their measures against the native decode, DISCARD both and
                 execute the native rows. After the window the controller returns (the episode finishes natively).
                 Run it ONLY as the NULL arm (``hs eval bcp --arms null``): there all calls go to the shadow session
                 and the native stream (decision indices, CRN seeds, actions) is untouched (K1-c2). In the
                 candidate arm the two extra calls would advance the framework decision counter and shift later
                 native seeds; the K1 plan never runs it there (tests/test_bcp.py shows the shift).
  switch         K2 candidate (NOT run now; only after a K1 pass and K2-c1 review). From the first decision after
                 the boundary every decision is ``ctx.policy.sample(1, instruction=clause)`` (no chunk cut). One
                 call per decision: decision indices and seeds stay aligned with the baseline. Its null arm is the
                 K2 shadow null (replays before the boundary, clause decodes after it, native actions executed).
  switch_control K2 churn control (K2-c2): as ``switch`` but the post-boundary instruction is the surface control
                 prompt (original + ' .').
No split (rules.split_clause is None) means the controller returns at once: fully native, no extra call.

Logged events (ctx.record): ``bcp_start`` (mode, instruction, clause, control), ``bcp_decision`` per decision
(d, step, role, chunk fingerprint; window decisions also carry the three chunks as base64 bytes with dtype/shape, the
decision-time state and the measures of rules.chunk_compare for clause and control), ``bcp_boundary`` (step,
segment), ``bcp_window_done`` / ``bcp_native``. No RNG, no simulator state, no case/seed/arm; public inputs only.
"""
from __future__ import annotations

import base64
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import rules

MODES = ("shadow_probe", "switch", "switch_control")


def _b64(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(np.asarray(a))
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "dtype": str(a.dtype), "shape": list(a.shape)}


class BCPController:
    def __init__(self, mode: str = "shadow_probe", window_decisions: int = 10, exec_rows: int = 5,
                 state_key: str = "states", hold_steps: int = rules.HOLD_STEPS, width_min: float = rules.WIDTH_MIN,
                 settle_rows: int = rules.SETTLE_ROWS, gripper_dim: int = rules.GRIPPER_DIM,
                 width_dims: Sequence[int] = rules.WIDTH_DIMS, log_rows: bool = False) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.window_decisions = int(window_decisions)
        self.exec_rows = int(exec_rows)
        if self.window_decisions < 1 or self.exec_rows < 1:
            raise ValueError("window_decisions and exec_rows must be positive")
        self.state_key = str(state_key)
        self.tracker_args = dict(hold_steps=hold_steps, width_min=width_min, settle_rows=settle_rows,
                                 gripper_dim=gripper_dim, width_dims=tuple(width_dims))
        self.log_rows = bool(log_rows)   # tests only: log every decision's chunk (proves executed rows are visible)

    def _state(self, obs: Mapping[str, Any]) -> Optional[np.ndarray]:
        st = obs.get(self.state_key) if isinstance(obs, Mapping) else None
        return None if st is None else np.asarray(st)

    async def run(self, ctx) -> None:
        instruction = ctx.instruction
        clause = rules.split_clause(instruction)
        control = rules.control_prompt(instruction)
        ctx.record("bcp_start", mode=self.mode, instruction=instruction, clause=clause, control=control)
        if clause is None:
            ctx.record("bcp_native", reason="no_split")
            return
        post_prompt = {"switch": clause, "switch_control": control}.get(self.mode)
        tracker = rules.CustodyTracker(**self.tracker_args)
        window_done = 0
        d = 0
        while not ctx.ended:
            step0 = ctx.steps
            obs = ctx.observe()
            switched = post_prompt is not None and tracker.boundary is not None
            if switched:
                chunk = np.array((await ctx.policy.sample(1, instruction=post_prompt))[0].actions)
            else:
                chunk = np.array((await ctx.policy.sample(1))[0].actions)
            rec: dict = {"d": d, "step": step0, "role": "switch" if switched else "native",
                         "chunk_sha": rules.array_fingerprint(chunk)}
            in_window = (self.mode == "shadow_probe" and tracker.boundary is not None
                         and window_done < self.window_decisions)
            if in_window:
                clause_chunk = np.array((await ctx.policy.sample(1, instruction=clause))[0].actions)
                control_chunk = np.array((await ctx.policy.sample(1, instruction=control))[0].actions)
                window_done += 1
                st = self._state(obs)
                rec.update(window=window_done, boundary=tracker.boundary,
                           native=_b64(chunk), clause=_b64(clause_chunk), control=_b64(control_chunk),
                           clause_sha=rules.array_fingerprint(clause_chunk),
                           control_sha=rules.array_fingerprint(control_chunk),
                           state=None if st is None else _b64(st),
                           clause_vs_native=rules.chunk_compare(chunk, clause_chunk),
                           control_vs_native=rules.chunk_compare(chunk, control_chunk))
            elif self.log_rows or switched:
                rec["chunk"] = _b64(chunk)
            ctx.record("bcp_decision", **rec)
            for i in range(min(self.exec_rows, len(chunk))):
                if ctx.ended:
                    break
                if await ctx.execute(chunk[i:i + 1]) == 0:
                    break
                if tracker.update(chunk[i], self._state(ctx.observe())):
                    ctx.record("bcp_boundary", step=tracker.boundary, decision=d, segment=tracker.segments[-1],
                               segments_before=len(tracker.segments) - 1)
            d += 1
            if self.mode == "shadow_probe" and window_done >= self.window_decisions:
                ctx.record("bcp_window_done", decisions=d, window=window_done)
                return
