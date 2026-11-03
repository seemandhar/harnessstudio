"""bdu (blocked-descent unload) monitor and window - pure Python + numpy, unit-tested.

Review runs/scalepp-20261102/reviews/proposals-r3-review.json BDU-RC1 (frozen), RC3 (precedence, applied by the
controller); proposal ideation/idea-r3a.json; offline replay idea-r3a-evidence/scripts/{bd_trigger,common}.py
(blocked_descent, first_run_at; parity reference bd_trigger_replay.json N20).

  b_k = (a_z[k] < -0.2) AND (executed gripper open, a[6] < 0 after LIBERO discretization) AND (z[k] - z[k-1] > -0.5 mm),
        b_0 = False (no previous post-step z). Fire after the action k at which b has been true for 20 consecutive steps.
  Window (<= 10 decisions, from decision k+1): the policy is queried every step as usual (one greedy decode); the executed
  action is the policy's with a_z := +0.15 (sham: also a_x = a_y := 0; rotation and gripper native). The window ends
  when the policy commands close (that close is executed natively and ends the window), when the EE has moved >= 2.5 cm
  laterally (xy) since the fire (post-step xy of action k), or after 10 window decisions. Then native actions resume;
  the counter resets; at most 4 fires per episode.
  Frozen counting rule (controller): every executed action feeds the counter (policy, scripted primitive and edited
  window actions); a GE window step resets it (bdu is not evaluated inside GE windows); a fire is evaluated only after
  a step with no active primitive / bdu window / GE window and no pending H2/H3 fire (precedence H2 > H1 > bdu > GE).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hs_gpp.rules import executed_action


class BduMonitor:
    def __init__(self, n: int = 20, az: float = -0.2, dz: float = -0.0005, max_fires: int = 4) -> None:
        if n < 1 or max_fires < 0:
            raise ValueError("bdu needs n >= 1 and max_fires >= 0")
        self.n, self.az, self.dz, self.max_fires = int(n), float(az), float(dz), int(max_fires)
        self.count = 0
        self.prev_z: Optional[float] = None
        self.k = -1
        self.fires = 0

    def feed(self, action: Any, z: Optional[float], reset: bool = False) -> bool:
        """After executed action k with post-step z. Returns b_k. `reset`: a GE-window step (not evaluated)."""
        a = executed_action(action)
        self.k += 1
        b = (self.k >= 1 and z is not None and self.prev_z is not None and float(a[2]) < self.az
             and float(a[6]) < 0 and (z - self.prev_z) > self.dz)
        self.prev_z = z
        self.count = 0 if (reset or not b) else self.count + 1
        return bool(b)

    def ready(self) -> bool:
        return self.count >= self.n and self.fires < self.max_fires

    def fired(self) -> None:
        self.fires += 1
        self.count = 0

    def window_ended(self) -> None:
        self.count = 0


class BduWindow:
    def __init__(self, xy0: Optional[np.ndarray], z0: Optional[float], sham: bool = False, max_steps: int = 10,
                 lift_az: float = 0.15, lateral: float = 0.025) -> None:
        self.xy0 = None if xy0 is None else np.asarray(xy0, dtype=np.float64)[:2].copy()
        self.z0, self.sham = z0, bool(sham)
        self.max_steps, self.lift_az, self.lateral = int(max_steps), float(lift_az), float(lateral)
        self.variant = "sham_lift_only" if sham else "candidate"
        self.steps = 0
        self.edited = 0
        self.outcome: Optional[str] = None
        self.commanded_xy: list = []
        self.realized_xy = 0.0
        self.realized_vec = np.zeros(2)

    def edit(self, proposed: Any) -> tuple:
        """(action to execute, edited?) for the policy's proposed action at a window decision."""
        a = np.asarray(proposed, dtype=np.float32).reshape(-1).copy()
        self.commanded_xy.append([float(a[0]), float(a[1])])
        if float(executed_action(a)[6]) > 0:
            return a, False                       # close: executed natively, ends the window
        a[2] = np.float32(self.lift_az)
        if self.sham:
            a[0] = np.float32(0.0)
            a[1] = np.float32(0.0)
        return a, True

    def after(self, edited: bool, xyz: Optional[np.ndarray]) -> None:
        self.steps += 1
        self.edited += int(edited)
        if xyz is not None and self.xy0 is not None:
            self.realized_vec = np.asarray(xyz, dtype=np.float64)[:2] - self.xy0
            self.realized_xy = float(np.linalg.norm(self.realized_vec))
        if not edited:
            self.outcome = "close_passthrough"
        elif self.realized_xy >= self.lateral:
            self.outcome = "lateral_done"
        elif self.steps >= self.max_steps:
            self.outcome = "max_steps"

    def info(self) -> dict:
        cmd = np.asarray(self.commanded_xy, dtype=np.float64) if self.commanded_xy else np.zeros((0, 2))
        mean_cmd = cmd.mean(0) if len(cmd) else np.zeros(2)
        denom = float(np.linalg.norm(mean_cmd) * np.linalg.norm(self.realized_vec))
        cos = float(np.dot(mean_cmd, self.realized_vec) / denom) if denom > 0 else None
        return {"steps": self.steps, "edited_steps": self.edited, "z_blocked": self.z0,
                "realized_lateral_m": self.realized_xy, "realized_vec": [float(x) for x in self.realized_vec],
                "mean_commanded_xy": [float(x) for x in mean_cmd], "cos_realized_vs_commanded": cos}
