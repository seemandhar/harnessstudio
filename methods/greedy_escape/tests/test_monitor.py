"""greedy-escape StallMonitor unit tests on synthetic action/width streams (CPU, no framework stack)."""
from __future__ import annotations

import numpy as np
import pytest

from hs_greedy_escape.monitor import StallMonitor, executed_action

OPEN, CLOSE = -1.0, 1.0
WIDE, NARROW = 0.06, 0.002     # finger widths (m): holding an object vs closed on nothing


def act(x=0.0, g=OPEN, y=0.0):
    return np.array([x, y, 0.0, 0.0, 0.0, 0.0, g], dtype=np.float32)


def moving(k, g=OPEN):
    return act(0.1 + 0.001 * k, g)          # never identical, never near zero


def feed(mon, actions, widths=None):
    fires = []
    for i, a in enumerate(actions):
        w = WIDE if widths is None else widths[i]
        ev = mon.step(a, w)
        if ev is not None:
            fires.append(ev)
    return fires


# ---- executed action ---------------------------------------------------------------------------------------
def test_executed_action_discretizes_gripper_like_libero():
    assert executed_action(act(g=-0.3))[6] == -1.0
    assert executed_action(act(g=0.0))[6] == 1.0       # LIBEROBenchmark.step: -1 only if a[6] < 0
    assert executed_action(act(g=-0.0))[6] == 1.0
    assert executed_action(act(g=0.7))[6] == 1.0
    with pytest.raises(ValueError):
        executed_action(np.zeros(6))


# ---- (a) fixed point -----------------------------------------------------------------------------------------
def test_fixed_point_fires_after_the_15th_identical_near_zero_action():
    mon = StallMonitor()
    stream = [moving(k) for k in range(5)] + [act(0.005)] * 15
    fires = feed(mon, stream)
    assert len(fires) == 1 and fires[0]["step"] == 19 and fires[0]["reasons"] == ["fixed_point"]
    assert mon.in_window and mon.windows[0]["start"] == 20


def test_fixed_point_needs_15_and_strict_eps():
    assert feed(StallMonitor(), [moving(0)] + [act(0.005)] * 14) == []
    # max|a| < 0.01 is strict, compared on the float32 executed value (as the offline rule on the recorded rows):
    # float32(0.01) = 0.0099999998 < 0.01 fires, float32(0.0101) does not
    assert feed(StallMonitor(), [act(0.0101)] * 30) == []
    assert len(feed(StallMonitor(), [act(0.01)] * 15)) == 1
    assert len(feed(StallMonitor(), [act(0.0099)] * 15)) == 1
    assert feed(StallMonitor(), [act(0.0, y=0.02)] * 30) == []      # any arm dim >= eps blocks it
    # identical arm but alternating gripper: not identical executed actions
    assert feed(StallMonitor(), [act(0.0, g=OPEN if k % 2 else CLOSE) for k in range(40)]) == []


def test_fixed_point_identity_uses_discretized_gripper():
    stream = [act(0.0, g=0.2 + 0.01 * k) for k in range(15)]       # raw gripper differs, executed +1 each time
    assert len(feed(StallMonitor(), stream)) == 1


# ---- (b) empty closes ----------------------------------------------------------------------------------------
def segment(start, dur, width_at_check, gap=5, prefix=None):
    """Actions/widths for: `gap` moving open steps, then a close held `dur` steps, then open."""
    acts, widths = [], []
    for k in range(gap):
        acts.append(moving(start + k)); widths.append(WIDE)
    for k in range(dur):
        acts.append(moving(start + gap + k, CLOSE)); widths.append(WIDE)
    return acts, widths


def empty_stream(durs, empty_flags, gap=5):
    acts, widths = [], []
    closes = []
    for dur, empty in zip(durs, empty_flags):
        a, w = segment(len(acts), dur, None, gap)
        i = len(acts) + gap
        closes.append(i)
        acts += a; widths += w
        # width checked after action i+10 (dur > 10) or after i+9 (dur == 10)
        chk = i + 10 if dur > 10 else i + 9
        if dur >= 10:
            widths[chk] = NARROW if empty else WIDE
    acts += [moving(len(acts) + k) for k in range(gap + 1)]
    widths += [WIDE] * (gap + 1)
    return acts, widths, closes


def test_third_empty_close_fires_at_close_plus_10():
    acts, widths, closes = empty_stream([12, 15, 11], [True, True, True])
    mon = StallMonitor()
    fires = feed(mon, acts, widths)
    assert [f["step"] for f in fires] == [closes[2] + 10] and fires[0]["reasons"] == ["empty_close"]
    assert [e["close"] for e in mon.empties if e["counted"]] == closes


def test_exactly_10_step_hold_uses_width_after_the_last_close_step():
    acts, widths, closes = empty_stream([10, 10, 10], [True, True, True])
    for c in closes:                       # the step after the hold (open) shows a wide width: must be ignored
        widths[c + 10] = WIDE
    fires = feed(StallMonitor(), acts, widths)
    assert [f["step"] for f in fires] == [closes[2] + 10]


def test_short_holds_and_held_objects_do_not_count():
    acts, widths, closes = empty_stream([9, 12, 12, 30, 12], [True, False, True, True, True])
    mon = StallMonitor()
    fires = feed(mon, acts, widths)
    # counted empties: closes[2], closes[3], closes[4] -> fire at closes[4] + 10
    assert [f["step"] for f in fires] == [closes[4] + 10]
    assert sum(e["empty"] for e in mon.empties) == 3 and len(mon.empties) == 4


def test_width_boundary_is_inclusive():
    acts, widths, closes = empty_stream([12, 12, 12], [True, True, True])
    for c in closes:
        widths[c + 10] = 0.004
    assert len(feed(StallMonitor(), acts, widths)) == 1


def test_missing_width_never_classifies():
    acts, _, _ = empty_stream([12, 12, 12, 12], [True] * 4)
    mon = StallMonitor()
    assert feed(mon, acts, [None] * len(acts)) == []
    assert mon.unclassified == 4
    assert mon.summary()["width_missing_steps"] == len(acts)


# ---- windows, re-arming and caps -----------------------------------------------------------------------------
def test_window_lasts_40_decisions_and_suspends_triggers():
    mon = StallMonitor()
    fires = feed(mon, [act(0.0)] * 15)
    assert fires and mon.in_window
    # 40 window decisions of the same stalled action: no new fire inside the window
    for j in range(40):
        assert mon.in_window
        assert mon.step(act(0.0), WIDE) is None
    assert not mon.in_window and mon.windows[0] == {"start": 15, "trigger": ["fixed_point"], "end": 54}
    assert mon.window_steps == 40
    # re-arm: needs 15 NEW identical greedy actions after the window
    fires = feed(mon, [act(0.0)] * 14)
    assert fires == []
    ev = mon.step(act(0.0), WIDE)
    assert ev is not None and ev["step"] == 69 and mon.windows[1]["start"] == 70


def test_empty_closes_in_window_or_straddling_it_do_not_count():
    mon = StallMonitor()
    # two counted empty closes, then a fixed-point fire opens a window
    acts, widths, closes = empty_stream([12, 12], [True, True])
    assert feed(mon, acts, widths) == []
    assert mon.empty_count == 2
    k0 = mon.k + 1
    # a close that starts BEFORE the window (k0) and is classified after the window starts
    pre = [act(0.0, g=CLOSE)] * 15          # 15 identical near-zero closed actions -> fixed-point fire at k0+14
    w = [WIDE] * 15
    w[10] = NARROW                          # empty at k0+10 (before the fire): counted (3rd) -> fires there
    fires = feed(mon, pre, w)
    assert fires[0]["step"] == k0 + 10 and fires[0]["reasons"] == ["empty_close"]
    assert mon.empty_count == 0 and mon.in_window
    # inside the window: empty closes are classified but not counted
    win_acts, win_w, _ = empty_stream([12], [True], gap=3)
    feed(mon, win_acts, win_w)
    assert all(not e["counted"] for e in mon.empties if e["classified_at"] > k0 + 10)
    while mon.in_window:
        mon.step(moving(mon.k), WIDE)
    end = mon.k
    assert mon.windows[0]["end"] == end and mon.rearm_step == end + 1
    # after the window three fresh empty closes are needed
    acts, widths, closes = empty_stream([12, 12, 12], [True] * 3)
    fires = feed(mon, acts, widths)
    assert [f["step"] for f in fires] == [end + 1 + closes[2] + 10]


def test_segment_started_in_window_and_classified_after_is_not_counted():
    mon = StallMonitor(window_decisions=5)
    feed(mon, [act(0.0)] * 15)               # window decisions 15..19
    acts = [act(0.0, g=CLOSE)] * 3          # close starts at 15 inside the window
    feed(mon, acts)
    feed(mon, [moving(k, CLOSE) for k in range(12)], [NARROW] * 12)   # classified at 25, after the window
    e = [x for x in mon.empties if x["close"] == 15]
    assert e and e[0]["empty"] and not e[0]["counted"]


def test_at_most_4_windows_then_fires_are_suppressed():
    mon = StallMonitor()
    starts = []
    for _ in range(6):
        for _ in range(15):
            ev = mon.step(act(0.0), WIDE)
        assert ev is not None
        starts.append(ev["window"])
        while mon.in_window:
            mon.step(act(0.0), WIDE)
    assert starts == [True, True, True, True, False, False]
    s = mon.summary()
    assert s["windows"] == 4 and s["suppressed_fires"] == 2 and s["escape_decisions"] == 160
    assert s["window_starts"] == [15, 70, 125, 180]


# ---- F10 (source review): finger width through the libero contract's public observation --------------------------
def test_finger_width_on_libero_contract_obs():
    from harnessstudio.contracts import load_contract
    from hs_greedy_escape.controller import GreedyEscapeController
    contract = load_contract("libero")
    raw = {"images": {"agentview": np.zeros((4, 4, 3), np.uint8)}, "task_description": "x",
           "states": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0215, -0.0205], dtype=np.float64),
           "controller_states": np.zeros(8, np.float32)}
    obs = contract.filter_public(raw)
    ctrl = GreedyEscapeController()
    assert ctrl.finger_width(obs) == pytest.approx(0.042)
    assert ctrl.finger_width(contract.filter_public({k: v for k, v in raw.items() if k != "states"})) is None
    assert ctrl.finger_width({"states": [0.0] * 7}) is None
