"""gpp rules: public trackers, the frozen online triggers and the scripted primitives (pure Python + numpy, unit-tested).

Review runs/scalepp-20261102/reviews/proposals-r2-review.json (r2b H1/H2, H1-RC1..5, H2-RC1..4, D1-D4, N1-N4); proposal
runs/scalepp-20261102/ideation/idea-r2b.json; offline replays idea-r2b-evidence/scripts/{h1,h2}_trigger_final.py and
common.py (EMPTY 4 mm, close_runs, segments). Every constant is frozen in the cards (D4: no sweep before the Campaign).

Conventions (as the offline replays): step k = index of the k-th EXECUTED action (policy or scripted); A[k] = executed
action with the gripper discretized like vla-eval LIBEROBenchmark.step (-1 if a[6] < 0 else +1; close iff +1);
S[k] = public post-step proprio states = [eef xyz, axis-angle(3), finger qL, qR]; w[k] = qL - qR; z[k] = S[k][2].
A close run is a maximal run [c, j) of executed close commands.

H2 deeper re-close (H2-RC1, online form, authoritative): pre-carry = no earlier completed close run is a lifted carry
  (run >= 10 steps, held = w[min(c+10, j-1)] > 4 mm, lift = max z[c:j] - z[c] > 3 cm). In a pre-carry run starting at c,
  fire after the first action k (in the run) at which the run has >= 3 close commands (k >= c+2) AND some w[c+q] <= 4 mm
  with q <= 6 and c+q <= k was observed. So k = max(c+2, c+lat) (offline: c+lat with the whole-run length >= 3).
  At most 2 fires; a 2nd needs the EE to have risen >= 2 cm (max z after the 1st primitive ended minus z at the 1st
  fire). Primitive: 3 steps [0,0,0,0,0,0,-1] (open, hold), then [0,0,-0.5,0,0,0,-1] until z <= z0 - 3 cm or 8 steps,
  contact stop when a step's realized dz > -1 mm, tested from move step 2 on (v1.1: OSC ramp-up from rest; move step 1
  can end only at the distance) (z0 = z after the open steps). Sham: ASCEND [0,0,+0.5,...,-1] until +3 cm or 8 steps,
  mirrored stop (realized dz < +1 mm, also from move step 2). Then hand back. Every move step's dz is logged (dz_mm).
H1 clamp-lift (H1-RC1, online at decision j): the policy's PROPOSED action at j is an open that would end a close run
  [c, j) with j-c >= 10, w[j-1] > 1 cm, |w[j-1] - w[j-4]| < 0.5 mm, max z[c:j] - z[c] < 5 mm, z[c] - min z[c:j] < 2 cm,
  sum |A[c:j, 3:6]| < 1.0, not carry_in_progress(c) (no lift > 3 cm since the start of a chain of close runs separated
  by <= 5 open steps). At most 3 fires, one per close run. Primitive (H1-RC2): instead of the open, [0,0,+0.5,0,0,0,+1]
  for up to 12 steps, stop at z >= z_fire + 6 cm; after the 4th step ABORT if z - z_fire < 1 cm or w <= 4 mm: execute
  the vetoed policy action (the open) and hand back. Sham (H1-RC5): [0,0,0,0,0,0,+1] for 12 steps, then hand back.
H3 freeze nudge (module only, off in gpp-core): once, at step >= 25 when the last 20 executed actions are identical with
  max|a[0:6]| < 0.01 and the EE is < 2 cm from its initial position: 8 steps [0,0,-0.5,0,0,0,-1], contact-stopped as
  H2's descent. Sham: upward, mirrored stop.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

EMPTY = 0.004


def executed_action(action: Any) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size != 7:
        raise ValueError(f"gpp expects 7-D LIBERO actions, got {a.size}")
    out = a.copy()
    out[6] = np.float32(-1.0) if float(a[6]) < 0 else np.float32(1.0)
    return out


def proprio(states: Any) -> Optional[tuple]:
    """(z, w, xyz) from a public `states` vector, or None if absent/short/non-finite."""
    if states is None:
        return None
    try:
        s = np.asarray(states, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if s.size < 8 or not np.all(np.isfinite(s[:8])):
        return None
    return float(s[2]), float(s[6] - s[7]), s[:3].copy()


class Tracker:
    """Executed-action and proprio history plus close-run bookkeeping (offline common.close_runs / segments)."""

    def __init__(self, xyz0: Optional[np.ndarray] = None) -> None:
        self.A: list = []          # executed (discretized) actions
        self.z: list = []          # post-step z (None when proprio missing)
        self.w: list = []          # post-step width
        self.xyz: list = []
        self.xyz0 = None if xyz0 is None else np.asarray(xyz0, dtype=np.float64)
        self.runs: list = []       # completed close runs (c, j)
        self.run_start: Optional[int] = None
        self.carried = False       # a completed run was a lifted carry (held, lift > 3 cm)

    @property
    def k(self) -> int:
        return len(self.A) - 1

    def update(self, action: Any, prop: Optional[tuple]) -> None:
        a = executed_action(action)
        close = bool(a[6] > 0)
        k = len(self.A)
        self.A.append(a)
        self.z.append(None if prop is None else prop[0])
        self.w.append(None if prop is None else prop[1])
        self.xyz.append(None if prop is None else prop[2])
        if close and self.run_start is None:
            self.run_start = k
        elif not close and self.run_start is not None:
            self._finish(self.run_start, k)
            self.run_start = None

    def _finish(self, c: int, j: int) -> None:
        self.runs.append((c, j))
        if j - c >= 10:
            kk = min(c + 10, j - 1)
            zs = self.z[c:j]
            if self.w[kk] is not None and None not in zs:
                if self.w[kk] > EMPTY and max(zs) - zs[0] > 0.03:
                    self.carried = True

    def run_len(self) -> int:
        return 0 if self.run_start is None else len(self.A) - self.run_start


class H2Trigger:
    def __init__(self, min_close: int = 3, within: int = 6, max_fires: int = 2, refire_rise: float = 0.02) -> None:
        self.min_close, self.within, self.max_fires, self.refire_rise = min_close, within, max_fires, refire_rise
        self.fires = 0
        self.fired_runs: set = set()
        self.first_fire_z: Optional[float] = None
        self.max_z_since_first_end: Optional[float] = None

    def note_primitive_end(self, z: Optional[float]) -> None:
        self.max_z_since_first_end = z

    def observe_z(self, z: Optional[float]) -> None:
        if self.max_z_since_first_end is not None and z is not None:
            self.max_z_since_first_end = max(self.max_z_since_first_end, z)

    def check(self, tr: Tracker) -> Optional[dict]:
        """After action k: fire? Uses only the past (online H2-RC1)."""
        c = tr.run_start
        if c is None or tr.carried or self.fires >= self.max_fires or c in self.fired_runs:
            return None
        k = tr.k
        if k - c + 1 < self.min_close:
            return None
        lat = next((q for q in range(0, self.within + 1) if c + q <= k and tr.w[c + q] is not None
                    and tr.w[c + q] <= EMPTY), None)
        if lat is None:
            return None
        if self.fires >= 1:
            if (self.first_fire_z is None or self.max_z_since_first_end is None
                    or self.max_z_since_first_end - self.first_fire_z < self.refire_rise):
                return None
        return {"close": c, "lat": lat, "detect": k, "z": tr.z[k], "w": tr.w[c + lat]}

    def fired(self, ev: dict) -> None:
        self.fires += 1
        self.fired_runs.add(ev["close"])
        if self.fires == 1:
            self.first_fire_z = ev["z"]


class H1Trigger:
    def __init__(self, min_run: int = 10, clamp_w: float = 0.01, settle: float = 0.0005, max_rise: float = 0.005,
                 max_descent: float = 0.02, max_rot: float = 1.0, chain_gap: int = 5, chain_lift: float = 0.03,
                 max_fires: int = 3) -> None:
        self.min_run, self.clamp_w, self.settle = min_run, clamp_w, settle
        self.max_rise, self.max_descent, self.max_rot = max_rise, max_descent, max_rot
        self.chain_gap, self.chain_lift, self.max_fires = chain_gap, chain_lift, max_fires
        self.fires = 0
        self.fired_runs: set = set()

    def carry_in_progress(self, tr: Tracker, c: int) -> bool:
        prev = [r for r in tr.runs if r[1] <= c]
        if not prev or c - prev[-1][1] > self.chain_gap:
            return False
        k = len(prev) - 1
        while k > 0 and prev[k][0] - prev[k - 1][1] <= self.chain_gap:
            k -= 1
        zs = tr.z[prev[k][0]:c + 1]
        if None in zs:
            return False
        return bool(max(zs) - zs[0] > self.chain_lift)

    def check(self, tr: Tracker, proposed: Any) -> Optional[dict]:
        """At decision j = len(A): the policy proposes `proposed`; fire if it is an open ending a qualifying run."""
        if self.fires >= self.max_fires or tr.run_start is None:
            return None
        if bool(executed_action(proposed)[6] > 0):
            return None                      # not an open
        c, j = tr.run_start, len(tr.A)
        if c in self.fired_runs or j - c < self.min_run:
            return None
        w, zs = tr.w, tr.z[c:j]
        if None in zs or w[j - 1] is None or w[j - 4] is None:
            return None
        rot = float(np.abs(np.asarray(tr.A[c:j])[:, 3:6]).sum())
        if not (w[j - 1] > self.clamp_w and abs(w[j - 1] - w[j - 4]) < self.settle
                and max(zs) - zs[0] < self.max_rise and zs[0] - min(zs) < self.max_descent and rot < self.max_rot):
            return None
        if self.carry_in_progress(tr, c):
            return None
        return {"close": c, "veto": j, "w": w[j - 1], "z": tr.z[j - 1], "rot": rot}

    def fired(self, ev: dict) -> None:
        self.fires += 1
        self.fired_runs.add(ev["close"])


class H3Trigger:
    def __init__(self, min_step: int = 25, identical: int = 20, eps: float = 0.01, max_move: float = 0.02) -> None:
        self.min_step, self.identical, self.eps, self.max_move = min_step, identical, eps, max_move
        self.fires = 0

    def check(self, tr: Tracker) -> Optional[dict]:
        k = tr.k
        if self.fires >= 1 or k < self.min_step or len(tr.A) < self.identical or tr.xyz0 is None or tr.xyz[k] is None:
            return None
        last = tr.A[-self.identical:]
        if any(not np.array_equal(a, last[0]) for a in last[1:]) or float(np.abs(last[0][:6]).max()) >= self.eps:
            return None
        if float(np.linalg.norm(tr.xyz[k] - tr.xyz0)) >= self.max_move:
            return None
        return {"detect": k, "z": tr.z[k]}

    def fired(self, ev: dict) -> None:
        self.fires += 1


# ---- scripted primitives: action() -> next action (or None when done); after(prop) after each executed step -------
class Primitive:
    module = ""
    variant = ""

    def __init__(self) -> None:
        self.steps = 0
        self.outcome: Optional[str] = None
        self.info: dict = {}

    def action(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def after(self, prop: Optional[tuple]) -> None:
        raise NotImplementedError


def _act(dz: float, grip: float) -> np.ndarray:
    return np.array([0.0, 0.0, dz, 0.0, 0.0, 0.0, grip], dtype=np.float32)


class Reclose(Primitive):
    """H2 (and H3 without the open steps): open/hold steps, then a vertical move with a stop rule."""
    module = "h2"

    def __init__(self, direction: int, open_steps: int = 3, speed: float = 0.5, distance: Optional[float] = 0.03,
                 max_move_steps: int = 8, stop_dz: float = 0.001, module: str = "h2",
                 z_start: Optional[float] = None, stop_from_move: int = 2) -> None:
        super().__init__()
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 (descend) or +1 (ascend sham)")
        self.direction, self.open_steps, self.speed = direction, int(open_steps), float(speed)
        self.distance = None if distance is None else float(distance)     # None: step cap only (H3)
        self.max_move, self.stop_dz = int(max_move_steps), float(stop_dz)
        # v1.1 (critic gpp-src-review G1): OSC delta control ramps up from rest, so the per-step stop rule is tested
        # only from move step `stop_from_move` (2); move step 1 can end only at the distance.
        self.stop_from_move = int(stop_from_move)
        if self.stop_from_move < 1:
            raise ValueError("stop_from_move must be >= 1")
        self.dz_mm: list = []
        self.module = module
        self.variant = "candidate" if direction < 0 else "sham_ascend"
        self.opened = 0
        self.moved = 0
        # with no open steps (H3) the move starts from the current post-step z
        self.z0: Optional[float] = z_start if self.open_steps == 0 else None
        self.z_prev: Optional[float] = self.z0

    def action(self) -> Optional[np.ndarray]:
        if self.outcome is not None:
            return None
        if self.opened < self.open_steps:
            return _act(0.0, -1.0)
        return _act(self.direction * self.speed, -1.0)

    def after(self, prop: Optional[tuple]) -> None:
        self.steps += 1
        z = None if prop is None else prop[0]
        if self.opened < self.open_steps:
            self.opened += 1
            if self.opened == self.open_steps:
                self.z0 = self.z_prev = z
            return
        self.moved += 1
        if z is None or self.z0 is None or self.z_prev is None:
            self.outcome = "no_proprio"
            return
        dz = z - self.z_prev
        self.z_prev = z
        self.dz_mm.append(round(dz * 1000.0, 3))
        self.info = {"z0": self.z0, "z": z, "travel": z - self.z0, "moved": self.moved, "dz_mm": list(self.dz_mm)}
        far = self.distance is not None and self.direction * (z - self.z0) >= self.distance
        testable = self.moved >= self.stop_from_move
        if far:
            self.outcome = "distance"
        elif testable and self.direction < 0 and dz > -self.stop_dz:
            self.outcome = "contact_stop"
        elif testable and self.direction > 0 and dz < self.stop_dz:
            self.outcome = "stop_mirrored"
        if self.outcome is None and self.moved >= self.max_move:
            self.outcome = "max_steps"


class ClampLift(Primitive):
    module = "h1"

    def __init__(self, vetoed: Any, z_fire: float, lift: bool = True, speed: float = 0.5, max_steps: int = 12,
                 height: float = 0.06, check_at: int = 4, min_rise: float = 0.01, empty_w: float = EMPTY) -> None:
        super().__init__()
        self.vetoed = np.asarray(vetoed, dtype=np.float32).reshape(-1).copy()
        self.z_fire, self.lift, self.speed = float(z_fire), bool(lift), float(speed)
        self.max_steps, self.height, self.check_at = int(max_steps), float(height), int(check_at)
        self.min_rise, self.empty_w = float(min_rise), float(empty_w)
        self.variant = "candidate" if lift else "sham_veto_only"
        self.abort_pending = False
        self.info = {"z_fire": self.z_fire}

    def action(self) -> Optional[np.ndarray]:
        if self.outcome is not None:
            return None
        if self.abort_pending:
            return self.vetoed.copy()
        return _act(self.speed if self.lift else 0.0, 1.0)

    def after(self, prop: Optional[tuple]) -> None:
        self.steps += 1
        if self.abort_pending:
            self.outcome = "abort"
            return
        if prop is None:
            self.outcome = "no_proprio"
            return
        z, w, _ = prop
        rise = z - self.z_fire
        self.info.update({"rise": rise, "w": w, "lift_steps": self.steps})
        if not self.lift:
            if self.steps >= self.max_steps:
                self.outcome = "veto_done"
            return
        if rise >= self.height:
            self.outcome = "lift_done"
        elif self.steps == self.check_at and (rise < self.min_rise or w <= self.empty_w):
            self.abort_pending = True
            self.info["abort_reason"] = "blocked" if rise < self.min_rise else "custody_lost"
        elif self.steps >= self.max_steps:
            self.outcome = "lift_cap"
