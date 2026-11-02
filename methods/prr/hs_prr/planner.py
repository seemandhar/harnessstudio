"""PRR trigger, checkpoint and guard state machine (pure; shared by the controller and the C3 replay-table script).

Frozen semantics (review r6-recovery-review.json C2-C4; replay scripts-r6-recovery/frozen_rule_replay.py):

Bookkeeping. Steps are executed-row indices s = 0, 1, ...; every executed row (native or harness) is fed with
``note_row(s, pre_state, row)`` where pre_state is the public state observed when row s was issued (= the recorded
post-step state S[s-1]). Native decisions happen every ``decision_steps`` (5) steps; ``decide(d, state)`` is called
with the decision-time public state (= S[d-1]). Gripper signs use the benchmark discretization (< 0 open, else closed).
``rel`` = first executed step with an open command after the first executed close command.

Route A (confirmed runaway, first-onset-only). cond(d) = tilt(state_d) > 55 deg OR (rel known AND d > rel AND
state_d[3] > 3.40). While route A is armed and no onset is pending, the first decision d > 0 with d >= search_start
and cond(d) is the onset U. At decision U + 15: cond true -> FIRE (f = U + 15); false -> route A disarmed. A fire that
abstains also disarms route A. Disarmed route A re-arms only at a restoration hand-back h, with search_start = h + 30.
Checkpoint A: the most recent recorded decision k with f - 65 <= k < f, k > 0, k > h (strictly after the last hand-back h,
whose decision-time state was produced by harness settle rows),
tilt(state_k) <= 30, state_k[3] <= 3.2705 and gsign(row k-1) == gsign(row f-1).

Route B (stall, first-stall-only per arming). At decision f: the window decisions f-60, f-55, ..., f (13 decision-time
samples) all recorded, f - 60 >= 250 and f - 60 > h (every sample produced by native rows); FIRE iff the EE position range ||max - min|| < 0.02 m. An
abstaining fire disarms route B until the next hand-back. Checkpoint B: the most recent recorded decision k with
f - 100 <= k < f, k > 0, k > h and ||p_k - p_f|| >= 0.03 m.

Route A is evaluated before route B at the same decision; route B is evaluated only if route A produced no valid plan.

Guards (both routes, abstain on failure, checked in this order after a checkpoint exists):
  C2 custody window: no step s in [cp, f) with gsign(row s) != gsign(row s-1) and ||p_s - p_f|| > 0.02 m, where p_s is
     the pre-row EE position of row s and p_f the decision-time position at f.
  C4 horizon: f + L + 10 + 5 <= 460 (L = f - cp; 10 = 2 correction decisions; 5 settle steps).
Latch = gsign(row f-1) (the fire-time executed gripper command). At most 2 restorations per episode.
Reversal rows = rows cp..f-1 in reverse order, dims 0-5 negated, gripper = latch (rows are never harness rows,
because cp > h).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .geometry import gsign, tilt_deg


@dataclass(frozen=True)
class Params:
    tilt_fire_deg: float = 55.0
    rx_fire: float = 3.40
    confirm_steps: int = 15
    checkpoint_tilt_max_deg: float = 30.0
    checkpoint_rx_max: float = 3.2705
    checkpoint_age_max_steps: int = 65
    stall_window_steps: int = 60
    stall_range_m: float = 0.02
    stall_earliest_step: int = 250
    stall_checkpoint_min_dist_m: float = 0.03
    stall_checkpoint_age_max: int = 100
    custody_tol_m: float = 0.02
    horizon_guard_last_step: int = 460
    correction_max_decisions: int = 2
    decision_steps: int = 5
    settle_steps: int = 5
    rearm_gap_steps: int = 30
    max_restorations: int = 2

    @classmethod
    def from_mapping(cls, m) -> "Params":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(m).items() if k in names})


@dataclass
class Plan:
    route: str
    fire: int
    checkpoint: Optional[int]
    L: Optional[int]
    latch: Optional[float]
    abstain: Optional[str]
    checks: dict = field(default_factory=dict)
    checkpoint_state: Optional[list] = None
    fire_state: Optional[list] = None
    rows: Optional[np.ndarray] = None       # reversal rows [L, 7] (PRR), None when abstaining

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("rows", None)
        return d


class Planner:
    def __init__(self, params: Params) -> None:
        self.p = params
        self.rows: dict = {}          # step -> executed row (float64 copy)
        self.gs: dict = {}            # step -> gripper sign
        self.pre_pos: dict = {}       # step -> pre-row EE position
        self.dec: dict = {}           # decision step -> decision-time state (native decisions only)
        self.first_close: Optional[int] = None
        self.rel: Optional[int] = None
        self.a_armed = True
        self.a_search = 0
        self.U: Optional[int] = None
        self.b_armed = True
        self.h: Optional[int] = None  # last hand-back step
        self.n_rest = 0
        self.events: list = []        # (name, fields) drained by the caller

    # ---- bookkeeping ----------------------------------------------------------------------------------------------
    def note_row(self, s: int, pre_state, row) -> None:
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        g = gsign(row[6])
        self.rows[s] = row.copy()
        self.gs[s] = g
        self.pre_pos[s] = None if pre_state is None else np.asarray(pre_state, dtype=np.float64)[:3].copy()
        if g > 0 and self.first_close is None:
            self.first_close = s
        if self.rel is None and self.first_close is not None and s > self.first_close and g < 0:
            self.rel = s

    def drain(self) -> list:
        ev, self.events = self.events, []
        return ev

    def _cond_a(self, d: int, st: np.ndarray):
        tilt = tilt_deg(st[3:6])
        rx = bool(self.rel is not None and d > self.rel and st[3] > self.p.rx_fire)
        return (tilt > self.p.tilt_fire_deg) or rx, tilt, rx

    # ---- decision -------------------------------------------------------------------------------------------------
    def decide(self, d: int, state) -> Optional[Plan]:
        """Feed one native decision; returns a Plan when a route fires (valid or abstaining), else None."""
        st = np.asarray(state, dtype=np.float64).reshape(-1)
        self.dec[d] = st.copy()
        p = self.p
        if d <= 0 or self.n_rest >= p.max_restorations:
            return None
        fired = None
        if self.a_armed:
            c, tilt, rx = self._cond_a(d, st)
            if self.U is not None:
                if d >= self.U + p.confirm_steps:
                    U, self.U = self.U, None
                    if d == U + p.confirm_steps and c:
                        fired = self._plan("A", d, st, {"onset": U, "tilt_at_fire": round(tilt, 4), "rx_route": rx})
                    else:
                        self.a_armed = False
                        self.events.append(("prr_cancel", {"route": "A", "onset": U, "step": d, "tilt": round(tilt, 4),
                                                           "rx": float(st[3])}))
            elif d >= self.a_search and c:
                self.U = d
                self.events.append(("prr_onset", {"route": "A", "step": d, "tilt": round(tilt, 4), "rx": float(st[3]),
                                                  "rx_route": rx}))
        if fired is not None:
            self.events.append(("prr_fire", fired.summary()))
            if fired.abstain is None:
                return fired
            self.a_armed = False
            out = fired
        else:
            out = None
        if self.b_armed:
            rng = self._stall_range(d)
            if rng is not None and rng < p.stall_range_m:
                planB = self._plan("B", d, st, {"stall_range_m": float(rng)})
                self.events.append(("prr_fire", planB.summary()))
                if planB.abstain is not None:
                    self.b_armed = False
                return planB
        return out

    def _stall_range(self, d: int) -> Optional[float]:
        p = self.p
        start = d - p.stall_window_steps
        if start < p.stall_earliest_step or (self.h is not None and start <= self.h):
            return None
        ks = range(start, d + 1, p.decision_steps)
        if any(k not in self.dec for k in ks):
            return None
        P = np.array([self.dec[k][:3] for k in ks])
        return float(np.linalg.norm(P.max(0) - P.min(0)))

    def _plan(self, route: str, f: int, st: np.ndarray, checks: dict) -> Plan:
        p = self.p
        latch = self.gs.get(f - 1)
        age = p.checkpoint_age_max_steps if route == "A" else p.stall_checkpoint_age_max
        cands = sorted((k for k in self.dec if f - age <= k < f and k > 0 and (self.h is None or k > self.h)),
                       reverse=True)
        cp = None
        for k in cands:
            sk = self.dec[k]
            if route == "A":
                ok = (tilt_deg(sk[3:6]) <= p.checkpoint_tilt_max_deg and sk[3] <= p.checkpoint_rx_max
                      and self.gs.get(k - 1) == latch)
            else:
                ok = float(np.linalg.norm(sk[:3] - st[:3])) >= p.stall_checkpoint_min_dist_m
            if ok:
                cp = k
                break
        checks = dict(checks)
        if cp is None:
            return Plan(route, f, None, None, latch, "no_checkpoint", checks, None, st.tolist())
        L = f - cp
        bad = []
        for s in range(cp, f):
            prev = self.gs.get(s - 1)
            if prev is not None and self.gs[s] != prev:
                dist = float(np.linalg.norm(self.pre_pos[s] - st[:3]))
                if dist > p.custody_tol_m:
                    bad.append([s, round(dist, 5)])
        checks["custody_bad_toggles"] = bad
        checks["gripper_changes_in_window"] = int(sum(1 for s in range(cp, f) if self.gs.get(s - 1) is not None
                                                      and self.gs[s] != self.gs[s - 1]))
        horizon_end = f + L + p.correction_max_decisions * p.decision_steps + p.settle_steps
        checks["horizon_end"] = horizon_end
        abstain = None
        if bad:
            abstain = "custody_window"
        elif horizon_end > p.horizon_guard_last_step:
            abstain = "horizon"
        rows = None
        if abstain is None:
            rows = np.array([np.concatenate([-self.rows[s][:6], [latch]]) for s in range(f - 1, cp - 1, -1)])
        return Plan(route, f, cp, L, latch, abstain, checks, self.dec[cp].tolist(), st.tolist(), rows)

    def handback(self, h: int) -> None:
        self.n_rest += 1
        self.h = h
        self.a_armed, self.a_search, self.U = True, h + self.p.rearm_gap_steps, None
        self.b_armed = True
        self.events.append(("prr_rearm", {"step": h, "restorations": self.n_rest, "a_search_from": self.a_search}))
