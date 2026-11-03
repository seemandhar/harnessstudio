"""Public stall monitor of greedy-escape (pure Python + numpy; no framework imports, unit-tested).

Frozen rules (review runs/scalepp-20261102/reviews/proposals-r1-review.json, greedy-escape GE1). Step index k is the
index of the k-th EXECUTED action of the episode (0-based); "width after k" is the finger opening
states[6] - states[7] of the public observation that follows action k (vla-eval LIBERO `states` = post-step
[eef pos3, axis-angle3, gripper qpos2]).

  Executed action.  The benchmark executes dims 0-5 as sent and discretizes the gripper (vla-eval LIBEROBenchmark.step:
    -1 if a[6] < 0 else +1). The monitor uses the same discretization, so its action equals the recorded executed
    action row. close(k) := discretized gripper(k) > 0.
  (a) Fixed point.  A run of identical executed actions (all 7 dims, exact equality) of length >= W = 15 whose
    common action has max|a[0:6]| < eps = 0.01 fires after its 15th action (offline: trig.first_trigger fp).
  (b) Empty close.  A close segment starting at step i (close(i), not close(i-1)) that is held >= 10 steps is EMPTY when
    the width 10 steps after the close is <= 4 mm: width after action i+10 when the segment is longer than 10 steps,
    width after action i+9 when it lasts exactly 10 steps (offline: feat.grip_segments, k = min(i+10, j-1)). Either way
    the segment is classified after action i+10. The 3rd COUNTED empty close fires (offline el = close + 10).
  Window.  A fire at step k starts an escape window at decision k+1 that lasts M = 40 decisions (or to the episode end).
    No trigger is evaluated inside a window. At most 4 windows per episode; later fires are recorded as suppressed.
  Re-arming.  (a): the identical-action run restarts at the first action after a window (window actions never count).
    (b): the empty-close count is reset at each fire; a segment counts only if it starts outside every window and at
    or after the first step after the last window ("every later 3rd empty close since the last window re-triggers").
    A segment that started before a window and is classified during or after it is recorded but never counted.
  Missing proprio.  If the observation has no finger dims, the segment cannot be classified (held = None, not counted);
    the number of such cases is reported (no width is guessed).

The monitor never sees rewards, success, simulator state, seeds, case ids or the arm.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def executed_action(action: Any) -> np.ndarray:
    """The 7-D action the LIBERO benchmark executes and records: dims 0-5 as float32, gripper -> -1/+1."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size != 7:
        raise ValueError(f"greedy-escape expects 7-D LIBERO actions, got {a.size}")
    out = a.copy()
    out[6] = np.float32(-1.0) if float(a[6]) < 0 else np.float32(1.0)
    return out


class StallMonitor:
    def __init__(self, fixed_point_steps: int = 15, fixed_point_eps: float = 0.01, empty_close_steps: int = 10,
                 empty_width_m: float = 0.004, empty_closes: int = 3, window_decisions: int = 40,
                 max_windows: int = 4) -> None:
        if min(fixed_point_steps, empty_close_steps, empty_closes, window_decisions) < 1 or max_windows < 0:
            raise ValueError("greedy-escape thresholds must be positive")
        if fixed_point_eps <= 0 or empty_width_m <= 0:
            raise ValueError("eps and the empty width must be positive")
        self.W, self.eps = int(fixed_point_steps), float(fixed_point_eps)
        self.hold, self.empty_w, self.n_empty = int(empty_close_steps), float(empty_width_m), int(empty_closes)
        self.M, self.max_windows = int(window_decisions), int(max_windows)
        # state
        self.k = -1                      # index of the last executed action
        self.window_left = 0             # decisions left in the current window (0 = greedy)
        self.rearm_step = 0              # first step after the last window
        self.run_action: Optional[np.ndarray] = None
        self.run_len = 0
        self.prev_close = False
        self.prev_width: Optional[float] = None
        self.seg_start: Optional[int] = None
        self.seg_in_window = False
        self.seg_done = False
        self.empty_count = 0             # counted empty closes since the last fire
        # activation record
        self.fires: list[dict] = []      # every fire, incl. suppressed ones
        self.windows: list[dict] = []    # started windows: {start, trigger, end}
        self.empties: list[dict] = []    # every classified segment held >= hold steps
        self.unclassified = 0            # segments held >= hold steps without a finger width
        self.window_steps = 0            # decisions decoded by the escape operator
        self.width_missing = 0           # steps whose next observation had no finger width (send_state off?)

    # ---- decision-time query -----------------------------------------------------------------------------
    @property
    def in_window(self) -> bool:
        return self.window_left > 0

    # ---- after each executed action ------------------------------------------------------------------------
    def step(self, action: Any, width_after: Optional[float]) -> Optional[dict]:
        """Update with the executed action k and the finger width after it. Returns the fire event (dict) when a
        trigger fires after this action (a window then starts at the next decision, unless suppressed)."""
        a = executed_action(action)
        self.k += 1
        k = self.k
        was_window = self.in_window
        if was_window:
            self.window_left -= 1
            self.window_steps += 1
        close = bool(a[6] > 0)
        reasons: list[str] = []
        if width_after is None:
            self.width_missing += 1

        # (b) gripper segments (tracked in every step; counted only per the re-arming rule)
        if close and not self.prev_close:
            self.seg_start, self.seg_in_window, self.seg_done = k, was_window, False
        if self.seg_start is not None and not self.seg_done and k == self.seg_start + self.hold:
            self.seg_done = True
            # held >= hold steps: classified with the width hold steps after the close (or after the last close
            # step when the segment lasted exactly `hold` steps)
            if close or self.prev_close:
                width = width_after if close else self.prev_width
                if width is None:
                    self.unclassified += 1
                else:
                    empty = bool(width <= self.empty_w)
                    counted = (empty and not self.seg_in_window and self.seg_start >= self.rearm_step
                               and not was_window)
                    self.empties.append({"close": self.seg_start, "classified_at": k, "width": float(width),
                                         "empty": empty, "counted": counted})
                    if counted:
                        self.empty_count += 1
                        if self.empty_count >= self.n_empty:
                            reasons.append("empty_close")
        if not close:
            if self.seg_start is not None and not self.seg_done:
                self.seg_done = True      # released before `hold` steps: not a hold
        self.prev_close = close
        self.prev_width = None if width_after is None else float(width_after)

        # (a) identical near-zero executed actions (greedy steps after the last window only)
        if was_window:
            self.run_action, self.run_len = None, 0
        else:
            if self.run_action is not None and np.array_equal(a, self.run_action):
                self.run_len += 1
            else:
                self.run_action, self.run_len = a, 1
            if self.run_len >= self.W and float(np.abs(self.run_action[:6]).max()) < self.eps:
                reasons.append("fixed_point")

        if was_window and self.window_left == 0:
            self.windows[-1]["end"] = k
            self.rearm_step = k + 1
            self.run_action, self.run_len = None, 0
        if was_window or not reasons:
            return None                # no trigger is evaluated inside a window
        event = {"step": k, "reasons": reasons, "run_len": self.run_len, "empty_count": self.empty_count,
                 "window": len(self.windows) < self.max_windows}
        self.fires.append(event)
        # re-arm both counters at every fire (also a suppressed one, so later fires are distinct events)
        self.empty_count = 0
        self.run_action, self.run_len = None, 0
        if event["window"]:
            self.window_left = self.M
            self.windows.append({"start": k + 1, "trigger": list(reasons), "end": None})
        return event

    def summary(self) -> dict:
        return {"windows": len(self.windows), "window_starts": [w["start"] for w in self.windows],
                "window_list": [dict(w) for w in self.windows], "fires": [dict(f) for f in self.fires],
                "suppressed_fires": sum(1 for f in self.fires if not f["window"]),
                "first_fire_step": self.fires[0]["step"] if self.fires else None,
                "escape_decisions": self.window_steps, "empty_closes_classified": len(self.empties),
                "empty_closes_empty": sum(1 for e in self.empties if e["empty"]),
                "empty_closes_counted": sum(1 for e in self.empties if e["counted"]),
                "unclassified_holds": self.unclassified, "width_missing_steps": self.width_missing,
                "steps": self.k + 1}
