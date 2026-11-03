"""Public-proprio grasp-failure triggers (H2 empty close, H1 settled unlifted clamp, optional drop).

Ported as code from the SCALE++ development replays (runs/scalepp-20261102/ideation/idea-r2b-evidence/scripts:
common.py sha256 02b1e827..., h2_trigger_final.py 96ccc790..., h1_trigger_final.py 1a0a05a5...). Inputs are
public only: executed action rows A[k] (7-D, A[k, 6] > 0 = close command) and post-step proprio rows
S[k] = [eef xyz, axis-angle(3), finger qL, qR]; width = qL - qR.

ONE function (`TriggerMonitor.check`) is used online by the controller and offline by `replay`, so the online
first-fire step equals the offline-predicted step by construction (review condition A2). At decision d the
controller has A[0:d], S[0:d] (S[d-1] = the current observation's proprio) and the policy's proposed action for d.
The offline replay of a vla-eval episode jsonl calls the same function with A[:d], S[:d] and proposal A[d].

Differences from the replay scripts (all forced by causality, none tuned):
- H2: the script fires at step c+lat once the whole close run is known to last >= 3 steps. Online the run's
  length is only known as it happens, so the fire decision is the first decision at which every condition is
  observable: max(c + lat + 1, c + 3).
- H1: the script fires at j, the step whose action is the policy's first OPEN after the close run. Online that
  open is the *proposed* action of decision j; the controller intercepts it (the open is not executed when the
  response moves the arm).
- A re-arm window (`rearm_steps`) after each hand-back and a per-episode escalation cap (review spec).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

DEFAULTS = {
    "h2_empty_close": True,
    "h1_settled_clamp": True,
    "drop": False,                 # optional; admitted only by the A2 replay rule (0 added success exposures)
    "h2_after_carry": False,       # optional post-carry extension (same A2 rule); spec default: before any carry
    "empty_width": 0.004,          # m; EMPTY in the replay scripts
    "h2_latency": 6,               # width <= empty within 6 steps of the close command
    "h2_min_run": 3,
    "h1_min_run": 10,
    "h1_min_width": 0.010,
    "h1_settle": 0.0005,           # |w[j-1] - w[j-4]|
    "h1_rise_max": 0.005,
    "h1_descent_max": 0.02,
    "h1_rot_max": 1.0,
    "carry_lift": 0.03,
    "blip_gap": 5,
    "segment_min_run": 10,
    "drop_high_width": 0.010,
    "drop_min_lift": 0.03,
    "rearm_steps": 20,
    "max_escalations": 3,
    "min_steps_left": 170,         # fix2: = min_steps_left_place (a fire that cannot afford a place is not escalated)
}


@dataclass
class Fire:
    kind: str                    # "H2_empty_close" | "H1_settled_clamp" | "drop"
    step: int                    # decision index at which the trigger fires
    close: int                   # first step of the close run
    detail: dict = field(default_factory=dict)
    eligible: bool = True
    steps_left: int = 0

    def to_json(self) -> dict:
        return {"kind": self.kind, "step": int(self.step), "close": int(self.close), "eligible": bool(self.eligible),
                "steps_left": int(self.steps_left), **{k: _plain(v) for k, v in self.detail.items()}}


def _plain(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def width(S: np.ndarray) -> np.ndarray:
    return S[:, 6] - S[:, 7]


def close_runs(A: np.ndarray) -> list:
    """Maximal runs [i, j) of close commands; an ongoing run at the end has j = len(A)."""
    if len(A) == 0:
        return []
    g = A[:, 6] > 0
    n, out, i = len(g), [], 0
    while i < n:
        if g[i]:
            j = i
            while j < n and g[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def segments(A: np.ndarray, S: np.ndarray, min_run: int = 10, empty: float = 0.004) -> list:
    """Close runs >= min_run steps, classified held (width > empty 10 steps after the close) or empty (common.py)."""
    w, P, out = width(S), S[:, :3], []
    for i, j in close_runs(A):
        if j - i < min_run:
            continue
        k = min(i + 10, j - 1)
        held = bool(w[k] > empty)
        lift = float(P[i:j, 2].max() - P[i, 2])
        out.append({"close": i, "open": j if j < len(A) else None, "dur": j - i, "held": held, "lift": lift})
    return out


class TriggerMonitor:
    """Stateful trigger evaluation over a growing public trajectory (online) or a recorded one (offline)."""

    def __init__(self, params: Optional[Mapping[str, Any]] = None) -> None:
        p = dict(DEFAULTS)
        p.update(dict(params or {}))
        unknown = set(p) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown trigger params {sorted(unknown)}")
        self.p = p
        self.fired_runs: set = set()          # (kind, close) already fired (eligible or not)
        self.escalations = 0                  # eligible fires (each one becomes an escalation)
        self.rearm_from = 0                   # close runs must start at or after this step
        self.fires: list = []

    # -- helpers --------------------------------------------------------------------------------------------
    def _carry_in_progress(self, runs: list, c: int, P: np.ndarray) -> bool:
        """h1_trigger_final.carry_in_progress: a lift > 3 cm since the start of a blip-chained close chain."""
        prev = [r for r in runs if r[1] <= c]
        if not prev or c - prev[-1][1] > self.p["blip_gap"]:
            return False
        k = len(prev) - 1
        while k > 0 and prev[k][0] - prev[k - 1][1] <= self.p["blip_gap"]:
            k -= 1
        start = prev[k][0]
        return bool(P[start:c + 1, 2].max() - P[start, 2] > self.p["carry_lift"])

    def lifted_carry_now(self, A: np.ndarray, S: np.ndarray) -> bool:
        """The latest close run is ongoing, holds something and has lifted > carry_lift."""
        if len(A) == 0 or not A[-1, 6] > 0:
            return False
        c, j = close_runs(A)[-1]
        w, P = width(S), S[:, :3]
        return bool(w[j - 1] > self.p["empty_width"] and P[c:j, 2].max() - P[c, 2] > self.p["carry_lift"])

    # -- the trigger rules ------------------------------------------------------------------------------------
    def _h2(self, A, S, d) -> Optional[Fire]:
        p, w, P = self.p, width(S), S[:, :3]
        runs = close_runs(A)
        first_carry = None
        if not p["h2_after_carry"]:
            first_carry = next((s for s in segments(A, S, p["segment_min_run"], p["empty_width"])
                                if s["held"] and s["lift"] > p["carry_lift"]), None)
        for c, j in runs:
            if first_carry is not None and c >= first_carry["close"]:
                break
            if c < self.rearm_from or ("H2_empty_close", c) in self.fired_runs:
                continue
            if j - c < p["h2_min_run"]:
                continue   # too short so far (an ongoing run is re-checked at the next decision)
            lat = next((q for q in range(0, p["h2_latency"] + 1) if c + q < j and w[c + q] <= p["empty_width"]), None)
            if lat is None:
                continue
            return Fire("H2_empty_close", d, c, {"latency": lat, "width": round(float(w[c + lat]), 5),
                                                 "z_close": round(float(P[c, 2]), 4)})
        return None

    def _h1(self, A, S, d, proposal) -> Optional[Fire]:
        p = self.p
        if proposal is None or d == 0 or not A[d - 1, 6] > 0 or float(proposal[6]) > 0:
            return None   # fires only when the policy proposes the first OPEN after a close run
        runs = close_runs(A)
        c, j = runs[-1]
        assert j == d
        if c < self.rearm_from or ("H1_settled_clamp", c) in self.fired_runs or j - c < p["h1_min_run"]:
            return None
        w, P = width(S), S[:, :3]
        rot = float(np.abs(A[c:j, 3:6]).sum())
        ok = (w[j - 1] > p["h1_min_width"] and abs(w[j - 1] - w[j - 4]) < p["h1_settle"]
              and not self._carry_in_progress(runs, c, P)
              and P[c:j, 2].max() - P[c, 2] < p["h1_rise_max"]
              and P[c, 2] - P[c:j, 2].min() < p["h1_descent_max"] and rot < p["h1_rot_max"])
        if not ok:
            return None
        return Fire("H1_settled_clamp", d, c, {"width": round(float(w[j - 1]), 5), "z_close": round(float(P[c, 2]), 4),
                                               "run": j - c, "rot_sum": round(rot, 4)})

    def _drop(self, A, S, d) -> Optional[Fire]:
        p = self.p
        if d == 0 or not A[d - 1, 6] > 0:
            return None
        c, j = close_runs(A)[-1]
        if c < self.rearm_from or ("drop", c) in self.fired_runs:
            return None
        w, P = width(S), S[:, :3]
        if w[d - 1] > p["empty_width"]:
            return None
        high = np.nonzero(w[c:d] > p["drop_high_width"])[0]
        if not len(high):
            return None
        m = c + int(high[0])
        if P[m:d, 2].max() - P[c, 2] < p["drop_min_lift"]:
            return None
        return Fire("drop", d, c, {"width": round(float(w[d - 1]), 5), "held_from": m})

    # -- public ----------------------------------------------------------------------------------------------
    def check(self, A: Sequence, S: Sequence, proposal: Optional[Sequence], steps_left: int) -> Optional[Fire]:
        """Evaluate at decision d = len(A) (= len(S)). Returns a Fire (eligible or not) or None.
        An ineligible fire is recorded and consumes its close run but no escalation."""
        A = np.asarray(A, dtype=np.float64).reshape(-1, 7) if len(A) else np.zeros((0, 7))
        S = np.asarray(S, dtype=np.float64).reshape(len(A), -1) if len(A) else np.zeros((0, 8))
        d = len(A)
        if d == 0 or S.shape[1] < 8:
            return None
        if self.escalations >= self.p["max_escalations"]:
            return None
        prop = None if proposal is None else np.asarray(proposal, dtype=np.float64).reshape(-1)
        fire = None
        if self.p["drop"]:
            fire = self._drop(A, S, d)
        if fire is None and not self.lifted_carry_now(A, S):   # no new escalation while carrying (except drop)
            if self.p["h1_settled_clamp"]:
                fire = self._h1(A, S, d, prop)
            if fire is None and self.p["h2_empty_close"]:
                fire = self._h2(A, S, d)
        if fire is None:
            return None
        fire.steps_left = int(steps_left)
        fire.eligible = steps_left >= self.p["min_steps_left"]
        self.fired_runs.add((fire.kind, fire.close))
        if fire.eligible:
            self.escalations += 1
        self.fires.append(fire)
        return fire

    def handback(self, step: int) -> None:
        """Called when control returns to the policy (with or without motion): re-arm after rearm_steps."""
        self.rearm_from = max(self.rearm_from, int(step) + int(self.p["rearm_steps"]))


def replay(A: Sequence, S: Sequence, horizon: int = 520, params: Optional[Mapping[str, Any]] = None) -> list:
    """Offline replay of a recorded episode under the respondoff semantics (every escalation hands back at its
    fire decision without motion). Returns the fires as JSON dicts (eligible and ineligible)."""
    A = np.asarray(A, dtype=np.float64).reshape(-1, 7)
    S = np.asarray(S, dtype=np.float64)
    mon = TriggerMonitor(params)
    for d in range(len(A)):
        fire = mon.check(A[:d], S[:d], A[d], horizon - d)
        if fire is not None and fire.eligible:
            mon.handback(d)
    return [f.to_json() for f in mon.fires]


def replay_jsonl(path: str, horizon: int = 520, params: Optional[Mapping[str, Any]] = None) -> list:
    """Replay a vla-eval episode jsonl (rows with 'action' and post-step 'state')."""
    import json
    rows = [json.loads(line) for line in open(path) if line.strip()]
    A = np.array([r["action"] for r in rows], dtype=np.float64)
    S = np.array([r["state"] for r in rows], dtype=np.float64)
    return replay(A, S, horizon, params)
