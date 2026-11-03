"""gpp rules on synthetic streams: online triggers (H2-RC1, H1-RC1, H3) and primitives (H2-RC2, H1-RC2/RC5)."""
from __future__ import annotations

import numpy as np
import pytest

from hs_gpp.rules import ClampLift, H1Trigger, H2Trigger, H3Trigger, Reclose, Tracker, executed_action, proprio

OPEN, CLOSE = -1.0, 1.0


def act(g, x=0.0, z=0.0, r=0.0):
    return np.array([x, 0.0, z, r, 0.0, 0.0, g], dtype=np.float32)


def st(z, w, x=0.0):
    return np.array([x, 0.0, z, 0.0, 0.0, 0.0, w / 2, -w / 2])


def feed(tr, rows):
    for a, s in rows:
        tr.update(a, proprio(s))


def test_proprio_and_discretization():
    assert proprio(None) is None and proprio([0.0] * 7) is None and proprio([np.nan] * 8) is None
    z, w, xyz = proprio(st(1.0, 0.05))
    assert z == 1.0 and w == pytest.approx(0.05)
    assert executed_action(act(0.0))[6] == 1.0 and executed_action(act(-0.2))[6] == -1.0


# ---- H2 ---------------------------------------------------------------------------------------------------------------
def h2():
    return H2Trigger(min_close=3, within=6, max_fires=2, refire_rise=0.02)


def test_h2_fires_at_max_c_plus_2_and_lat():
    for widths, expect in (([0.002, 0.002, 0.002], 2), ([0.05, 0.03, 0.01, 0.01, 0.003], 4), ([0.003], None)):
        tr, t = Tracker(), h2()
        feed(tr, [(act(OPEN), st(1.0, 0.08))] * 3)
        got = None
        for q, w in enumerate(widths):
            feed(tr, [(act(CLOSE), st(1.0, w))])
            ev = t.check(tr)
            if ev is not None and got is None:
                got = ev["detect"] - 3
        assert got == expect, (widths, got)


def test_h2_needs_empty_within_6_and_pre_carry():
    tr, t = Tracker(), h2()
    feed(tr, [(act(CLOSE), st(1.0, 0.05))] * 7 + [(act(CLOSE), st(1.0, 0.002))])   # empty only at q = 7
    assert t.check(tr) is None
    tr, t = Tracker(), h2()
    # a completed lifted carry (held at c+10, lift > 3 cm) disables H2 for later runs
    carry = [(act(CLOSE), st(1.0 + 0.004 * k, 0.05)) for k in range(15)] + [(act(OPEN), st(1.06, 0.08))]
    feed(tr, carry)
    assert tr.carried
    feed(tr, [(act(CLOSE), st(1.0, 0.002))] * 3)
    assert t.check(tr) is None


def test_h2_second_fire_needs_2cm_rise_and_max_two():
    tr, t = Tracker(), h2()
    feed(tr, [(act(CLOSE), st(1.0, 0.002))] * 3)
    ev = t.check(tr)
    t.fired(ev)
    t.note_primitive_end(0.97)
    feed(tr, [(act(OPEN), st(0.97, 0.08))])
    t.observe_z(0.97)
    feed(tr, [(act(CLOSE), st(0.97, 0.002))] * 3)
    assert t.check(tr) is None                       # no rise
    feed(tr, [(act(OPEN), st(1.021, 0.08))])
    t.observe_z(1.021)
    feed(tr, [(act(CLOSE), st(1.021, 0.002))] * 3)
    ev2 = t.check(tr)
    assert ev2 is not None
    t.fired(ev2)
    feed(tr, [(act(OPEN), st(1.05, 0.08))] + [(act(CLOSE), st(1.05, 0.002))] * 3)
    t.observe_z(1.05)
    assert t.check(tr) is None                       # max 2


# ---- H1 ---------------------------------------------------------------------------------------------------------------
def h1():
    return H1Trigger()


def clamp(n=12, w=0.05, z=1.0, rot=0.0, drift=0.0):
    return [(act(CLOSE, r=rot), st(z + drift * k, w)) for k in range(n)]


def test_h1_fires_on_open_ending_settled_clamp():
    tr, t = Tracker(), h1()
    feed(tr, [(act(OPEN), st(1.0, 0.08))] * 8 + clamp())
    ev = t.check(tr, act(OPEN))
    assert ev is not None and ev["veto"] == 20 and ev["close"] == 8
    assert t.check(tr, act(CLOSE)) is None           # only an open can be vetoed


@pytest.mark.parametrize("rows", [clamp(n=9), clamp(w=0.009), clamp(drift=0.001), clamp(drift=-0.003),
                                  clamp(rot=0.1)])
def test_h1_exclusions(rows):
    tr, t = Tracker(), h1()
    feed(tr, [(act(OPEN), st(1.0, 0.08))] * 8 + rows)
    assert t.check(tr, act(OPEN)) is None


def test_h1_unsettled_and_blip_chain_carry_excluded():
    tr, t = Tracker(), h1()
    rows = clamp(n=12)
    rows[-1] = (act(CLOSE), st(1.0, 0.051))          # |w[j-1] - w[j-4]| = 1 mm
    feed(tr, [(act(OPEN), st(1.0, 0.08))] * 8 + rows)
    assert t.check(tr, act(OPEN)) is None
    tr, t = Tracker(), h1()
    lifted = [(act(CLOSE), st(1.0 + 0.005 * k, 0.05)) for k in range(10)]      # a previous run lifted 4.5 cm
    feed(tr, [(act(OPEN), st(1.0, 0.08))] * 3 + lifted + [(act(OPEN), st(1.045, 0.05))] * 2 + clamp(z=1.045))
    assert t.check(tr, act(OPEN)) is None            # blip-split carry in progress


# ---- H3 ---------------------------------------------------------------------------------------------------------------
def test_h3_needs_25_steps_20_identical_and_no_motion():
    tr, t = Tracker(np.array([0.0, 0.0, 1.0])), H3Trigger()
    for k in range(26):
        feed(tr, [(act(OPEN, x=0.002), st(1.0, 0.08))])
        ev = t.check(tr)
        if ev:
            break
    assert ev is not None and ev["detect"] == 25
    tr, t = Tracker(np.array([0.0, 0.0, 1.0])), H3Trigger()
    feed(tr, [(act(OPEN, x=0.002), st(1.0, 0.08, x=0.03))] * 30)
    assert t.check(tr) is None                        # moved >= 2 cm


# ---- primitives -------------------------------------------------------------------------------------------------------
def run_prim(p, zs, ws=None):
    out = []
    for i, z in enumerate(zs):
        a = p.action()
        if a is None:
            break
        out.append(a)
        p.after((z, 0.08 if ws is None else ws[i], np.array([0, 0, z])))
    return out


def test_h2_descent_distance_contact_and_ascend_sham():
    p = Reclose(-1)
    acts = run_prim(p, [1.0] * 3 + [1.0 - 0.0041 * k for k in range(1, 12)])
    assert [a[6] for a in acts] == [-1.0] * 11 and [a[2] for a in acts[:3]] == [0.0] * 3
    assert all(a[2] == pytest.approx(-0.5) for a in acts[3:]) and p.outcome == "distance" and p.steps == 11
    p = Reclose(-1)
    run_prim(p, [1.0] * 3 + [0.996, 0.992, 0.9915])  # dz -0.5 mm on the 3rd move: contact
    assert p.outcome == "contact_stop" and p.steps == 6
    p = Reclose(-1)
    run_prim(p, [1.0] * 3 + [1.0 - 0.002 * k for k in range(1, 12)])
    assert p.outcome == "max_steps" and p.moved == 8
    p = Reclose(1)
    acts = run_prim(p, [1.0] * 3 + [1.0 + 0.0041 * k for k in range(1, 12)])
    assert p.variant == "sham_ascend" and acts[3][2] == pytest.approx(0.5) and p.outcome == "distance"


def test_h1_lift_done_cap_abort_and_sham():
    vetoed = act(OPEN, x=0.1)
    p = ClampLift(vetoed, 1.0)
    acts = run_prim(p, [1.0 + 0.0041 * k for k in range(1, 20)], [0.05] * 19)
    assert p.outcome == "lift_cap" and p.steps == 12 and all(a[6] == 1.0 and a[2] == pytest.approx(0.5) for a in acts)
    p = ClampLift(vetoed, 1.0)
    run_prim(p, [1.0 + 0.007 * k for k in range(1, 20)], [0.05] * 19)
    assert p.outcome == "lift_done" and p.steps == 9               # 6.3 cm at step 9
    p = ClampLift(vetoed, 1.0)
    acts = run_prim(p, [1.0 + 0.002 * k for k in range(1, 20)], [0.05] * 19)
    assert p.outcome == "abort" and p.info["abort_reason"] == "blocked" and np.array_equal(acts[-1], vetoed)
    assert p.steps == 5
    p = ClampLift(vetoed, 1.0)
    acts = run_prim(p, [1.0 + 0.0041 * k for k in range(1, 20)], [0.05, 0.03, 0.01, 0.003] + [0.0] * 15)
    assert p.outcome == "abort" and p.info["abort_reason"] == "custody_lost"
    p = ClampLift(vetoed, 1.0, lift=False)
    acts = run_prim(p, [1.0] * 20, [0.05] * 20)
    assert p.outcome == "veto_done" and p.steps == 12 and all(a[2] == 0.0 and a[6] == 1.0 for a in acts)
