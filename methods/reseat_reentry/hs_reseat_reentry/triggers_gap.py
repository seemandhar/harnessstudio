"""Post-release gap trigger (Run X r2, Scientist C's H1; review proposal_r2_review.json merged_spec.trigger).

Public proprio only. Rows S[k] = post-step proprio [eef xyz, axis-angle(3), finger qL, qR]; width = qL - qR.

Carry (C's definition, ported from C's carry.py as code): a carry starts when 0.010 < width < 0.070 m and the eef z
rises > 0.03 m above z at the start of that width run; it ends at the first row with width >= 0.070 ("release") or
<= 0.004 ("lost").

FIRE (C's gap.py, made causal): after a carry has ended (end_kinds; C's measured script counts both a release and a
loss as an end - see report), no new carry for W = 140 steps, steps_left >= 150. C's offline script decides "no new
carry" with the START row of a later carry, which is only known once that carry's lift is detected (non-causal). The
online rule therefore DEFERS while a width run is in progress at the window end: if it becomes a carry there is no
fire (C's decision), if it ends without a lift the fire happens at that decision (C fires at en + W). Same decisions as
C's script except timing on deferred cases; on the seed-9 SCALE-CRN replay it reproduces 7/15 failures + 1/35
successes with 6/8 identical fire steps (t7e4 215->223, t8e2 353->360). The SAME function serves the controller online
and the offline replay (`replay`), so online first-fire == offline predicted fire on the same trajectory.
<= 2 fires per episode; each completed carry fires at most once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

DEFAULTS = {
    "W": 140,
    "min_steps_left": 150,
    "max_fires": 2,
    "carry_width_lo": 0.010,
    "carry_width_hi": 0.070,
    "carry_lift": 0.03,
    "lost_width": 0.004,
    "end_kinds": ["release", "lost"],     # C's measured script (any end); the review text says "release"
    "horizon": 520,
}


def carries(S: np.ndarray, p: Mapping[str, Any] = DEFAULTS) -> list:
    """[(start_row, end_row|None, end_kind|None, detected_row)] over rows S (C's carry.py, as code)."""
    w = S[:, 6] - S[:, 7]
    z = S[:, 2]
    out, inc, z0, st = [], False, None, None
    lo, hi, lift, lost = p["carry_width_lo"], p["carry_width_hi"], p["carry_lift"], p["lost_width"]
    for k in range(len(S)):
        if not inc:
            if lo < w[k] < hi:
                if z0 is None:
                    z0, st = z[k], k
                if z[k] - z0 > lift:
                    inc = True
                    out.append([st, None, None, k])
            else:
                z0 = None
        elif w[k] >= hi or w[k] <= lost:
            out[-1][1] = k
            out[-1][2] = "release" if w[k] >= hi else "lost"
            inc, z0 = False, None
    return out


@dataclass
class Fire:
    step: int
    carry_end: int
    end_kind: str
    steps_left: int
    deferred_by: int = 0
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"kind": "post_release_gap", "step": int(self.step), "carry_end": int(self.carry_end),
                "end_kind": self.end_kind, "steps_left": int(self.steps_left), "deferred_by": int(self.deferred_by),
                **self.detail}


class GapMonitor:
    def __init__(self, params: Optional[Mapping[str, Any]] = None) -> None:
        p = dict(DEFAULTS)
        p.update(dict(params or {}))
        unknown = set(p) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown trigger params {sorted(unknown)}")
        p["end_kinds"] = list(p["end_kinds"])
        if not set(p["end_kinds"]) <= {"release", "lost"} or not p["end_kinds"]:
            raise ValueError("end_kinds must be a non-empty subset of ['release', 'lost']")
        self.p = p
        self.fired_ends: set = set()
        self.fires: list = []

    def check(self, S: Sequence, steps_left: int) -> Optional[Fire]:
        """Evaluate at decision d = len(S) (rows 0..d-1 known). Returns a Fire or None."""
        if len(S) == 0:
            return None
        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 2 or S.shape[1] < 8:
            return None
        p, d = self.p, len(S)
        if len(self.fires) >= p["max_fires"]:
            return None
        c = carries(S, p)
        done = [x for x in c if x[1] is not None and x[2] in p["end_kinds"]]
        if not done or c[-1][1] is None or c[-1] is not done[-1]:
            return None            # no ended carry, a carry is in progress, or the newest carry ended differently
        st, en, kind, _ = done[-1]
        if en in self.fired_ends or d < en + p["W"]:
            return None
        if steps_left < p["min_steps_left"]:
            self.fired_ends.add(en)          # too late for this carry: it can never fire
            return None
        w = S[d - 1, 6] - S[d - 1, 7]
        if p["carry_width_lo"] < w < p["carry_width_hi"]:
            return None            # a width run is in progress: wait until it lifts (carry) or ends
        self.fired_ends.add(en)
        fire = Fire(d, en, kind, steps_left, d - (en + p["W"]))
        self.fires.append(fire)
        return fire


def replay(S: Sequence, horizon: int = 520, params: Optional[Mapping[str, Any]] = None, n_steps: Optional[int] = None) -> list:
    """Offline replay over a recorded episode's post-step rows (decisions 0..len-1)."""
    S = np.asarray(S, dtype=np.float64)
    mon = GapMonitor(params)
    n = len(S) if n_steps is None else n_steps
    for d in range(n):
        mon.check(S[:d], horizon - d)
    return [f.to_json() for f in mon.fires]


def replay_jsonl(path: str, horizon: int = 520, params: Optional[Mapping[str, Any]] = None) -> list:
    import json
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return replay(np.array([r["state"] for r in rows], dtype=np.float64), horizon, params)
