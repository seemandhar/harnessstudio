"""Open-gripper re-seat primitive on public proprio (review proposal_r2_review.json merged_spec.response).

`states` = [eef xyz (grip site), eef axis-angle(3), finger qL, qR] (vla-eval LIBERO). Actions are 7-D OSC deltas:
0.05 m per unit position action; gripper -1 open / +1 close (vla-eval binarises it, so every action sets it).
Vendored and reduced from the witnessed-rescue method package (methods/witnessed_rescue, motion module after fix2b,
sha256 8afb975d...: ACTION_SCALE, the proportional servo with its stall early-return); the grasp/lift/place/custody
code is not carried over. This module NEVER commands a gripper close: every action it builds has gripper = -1.

Re-seat: vertical lift to the transit z (if below it), lateral transit at transit z above the target, descent to the
hover z; current orientation kept (zero rotation command); tolerance 0.02 m per waypoint; at most `budget` primitive
steps in total, else stop where it is (the caller hands back to the policy). fix1 (review r2_impl_review ruling 2):
budget = clip(steps_left_at_fire - reserve_policy_steps, min_steps, max_steps) = clip(steps_left - 105, 45, 90).
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, Optional

import numpy as np

ACTION_SCALE = 0.05
OPEN = -1.0

DEFAULTS = {
    "tol": 0.02,               # frozen (review)
    "max_steps": 90,           # ruling 2: per-re-seat budget cap
    "min_steps": 45,           # ruling 2: floor (the originally reviewed budget)
    "reserve_policy_steps": 105,   # ruling 2: policy steps kept after a re-seat (150 - 45)
    "gain": 2.5,               # action = clip(gain * err / 0.05, +-1): full speed until 2 cm, then proportional
    "stall_window": 10,
    "stall_progress": 0.004,
}


def reseat_budget(steps_left: int, p: Mapping[str, Any] = DEFAULTS) -> int:
    """clip(steps_left - reserve_policy_steps, min_steps, max_steps) (review r2_impl_review ruling 2)."""
    return int(min(p["max_steps"], max(p["min_steps"], int(steps_left) - p["reserve_policy_steps"])))


class Motion:
    """`step(action) -> executed (0/1)`, `state() -> np.ndarray | None`, `ended() -> bool`."""

    def __init__(self, step: Callable[[np.ndarray], Awaitable[int]], state: Callable[[], Optional[np.ndarray]],
                 ended: Callable[[], bool], params: Optional[Mapping[str, Any]] = None) -> None:
        p = dict(DEFAULTS)
        p.update(dict(params or {}))
        unknown = set(p) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown motion params {sorted(unknown)}")
        self.p, self._step, self._state, self._ended = p, step, state, ended
        self.steps = 0

    def pos(self) -> np.ndarray:
        s = self._state()
        if s is None or len(s) < 3:
            raise RuntimeError("no end-effector position in the observation")
        return np.asarray(s[:3], dtype=np.float64)

    @staticmethod
    def action(err) -> np.ndarray:
        a = np.zeros(7, dtype=np.float32)
        a[0:3] = np.clip(np.asarray(err, dtype=np.float64) / ACTION_SCALE, -1.0, 1.0)
        a[6] = OPEN
        return a

    async def servo(self, target, budget: int) -> dict:
        p = self.p
        target = np.asarray(target, dtype=np.float64)
        steps, stalled = 0, False
        window = [self.pos()]
        while steps < budget and not self._ended():
            err = target - self.pos()
            if float(np.linalg.norm(err)) < p["tol"]:
                break
            n = await self._step(self.action(p["gain"] * err))
            steps += n
            self.steps += n
            if n == 0:
                break
            window.append(self.pos())
            if len(window) > p["stall_window"] and steps % p["stall_window"] == 0:
                if float(np.linalg.norm(window[-1] - window[-1 - p["stall_window"]])) < p["stall_progress"]:
                    stalled = True
                    break
        final = self.pos()
        return {"target": [round(float(v), 4) for v in target], "final": [round(float(v), 4) for v in final],
                "error": round(float(np.linalg.norm(target - final)), 4), "steps": steps, "stalled": stalled,
                "reached": float(np.linalg.norm(target - final)) < p["tol"]}

    async def reseat(self, hover_xyz, transit_z: float, budget: Optional[int] = None) -> dict:
        p = self.p
        budget = int(p["max_steps"] if budget is None else budget)
        hx, hy, hz = (float(v) for v in hover_xyz)
        start = self.pos()
        used, legs = 0, []
        plan = []
        if start[2] < transit_z - p["tol"]:
            plan.append(("lift", [start[0], start[1], transit_z]))
        plan.append(("transit", [hx, hy, max(transit_z, float(start[2]))]))
        plan.append(("descend", [hx, hy, hz]))
        for name, wp in plan:
            if used >= budget or self._ended():
                break
            leg = await self.servo(wp, budget - used)
            leg["leg"] = name
            legs.append(leg)
            used += leg["steps"]
            if leg["stalled"]:
                break
        final = self.pos()
        err = float(np.linalg.norm(final - np.array([hx, hy, hz])))
        return {"primitive": "reseat", "legs": legs, "steps": used, "budget": budget,
                "reached": err < p["tol"], "final_error": round(err, 4),
                "budget_exhausted": used >= budget and err >= p["tol"],
                "start": [round(float(v), 4) for v in start], "final": [round(float(v), 4) for v in final]}
