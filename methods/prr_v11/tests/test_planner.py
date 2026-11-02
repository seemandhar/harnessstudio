"""Pure planner/controller-unit tests (C2, C3, C4, re-arm, cap, rx route, ortho coin, cut restoration)."""
from __future__ import annotations

import hashlib

import anyio
import numpy as np
import pytest

from hs_prr_v11.controller import PRR, ortho_coin, state_digest
from hs_prr_v11.geometry import exp, gsign, log, rot_error, rot_z, tilt_deg
from hs_prr_v11.planner import Params, Planner

P = Params()
DOWN = log(exp([np.pi, 0, 0]))


def st(pos=(0, 0, 1), tilt=0.0, rx=None, width=0.08):
    """State with the tool tilted by `tilt` deg about world x from gripper-down (or an explicit rx with zero tilt)."""
    aa = log(exp([np.radians(tilt), 0, 0]) @ exp([np.pi, 0, 0]))
    if rx is not None:
        aa = np.array([rx, 0.0, 0.0])
    return np.concatenate([np.asarray(pos, float), aa, [width / 2, -width / 2]])


def feed(pl, T, state_fn, row_fn, until_plan=True):
    plans = []
    for s in range(T):
        if s % 5 == 0 and s > 0:
            plan = pl.decide(s, state_fn(s))
            if plan is not None:
                plans.append(plan)
                if until_plan and plan.abstain is None:
                    return plans
        pl.note_row(s, state_fn(s), row_fn(s))
    return plans


def test_geometry():
    assert abs(tilt_deg(DOWN)) < 1e-6
    assert abs(tilt_deg(st(tilt=40)[3:6]) - 40) < 1e-6
    assert gsign(0.0) == 1.0 and gsign(-1e-9) == -1.0
    assert np.allclose(rot_z(np.array([[1.0, 2.0, 3.0]]), +1), [[-2.0, 1.0, 3.0]])
    assert np.allclose(rot_z(np.array([[1.0, 2.0, 3.0]]), -1), [[2.0, -1.0, 3.0]])
    e = rot_error(st(tilt=0)[3:6], st(tilt=20)[3:6])
    assert abs(np.degrees(np.linalg.norm(e)) - 20) < 1e-6 and e[0] < 0     # world-frame x rotation back to 0
    # angle > pi representation (quat_no_antipodal): 3.4 rad about x == -(2 pi - 3.4) rad about x
    assert abs(tilt_deg([3.4, 0, 0]) - tilt_deg([-(2 * np.pi - 3.4), 0, 0])) < 1e-9


def ramp(t0, rate, s):     # tilt (deg) after rows t0..s-1 at `rate` deg/row
    return max(0.0, (s - t0) * rate)


def test_route_a_confirm_fire_checkpoint_latch_rows():
    pl = Planner(P)
    rows = {}
    def row(s):
        r = np.array([0.01 * s, 0.002, 0.0, 0.1 if 100 <= s < 130 else 0.0, 0, 0, 1.0 if s >= 40 else -1.0])
        rows[s] = r
        return r
    plans = feed(pl, 400, lambda s: st((0.001 * s, 0, 1), tilt=min(ramp(100, 2.3, s), 69)), row)
    ev = [n for n, _ in pl.drain()]
    plan = plans[-1]
    # tilt > 55 first at decision 125 (25 rows * 2.3 = 57.5); confirm at 140; checkpoint = latest tilt <= 30 -> 110
    assert ev[:2] == ["prr_onset", "prr_fire"] and plan.route == "A" and plan.fire == 140 and plan.checkpoint == 110
    assert plan.L == 30 and plan.latch == 1.0 and plan.abstain is None
    exp_rows = np.array([np.concatenate([-rows[s][:6], [1.0]]) for s in range(139, 109, -1)])
    assert np.array_equal(plan.rows, exp_rows)


def test_confirmation_cancel_disarms_until_handback():
    pl = Planner(P)
    tilt = lambda s: 60.0 if 120 <= s < 130 else (60.0 if 200 <= s else 0.0)
    plans = feed(pl, 300, lambda s: st(tilt=tilt(s)), lambda s: np.zeros(7))
    names = [n for n, _ in pl.drain()]
    assert names == ["prr_onset", "prr_cancel"] and plans == []          # no second onset at 200+ (first-onset-only)
    pl.handback(300)
    assert pl.a_search == 330
    assert pl.decide(325, st(tilt=60)) is None and pl.U is None          # before h + 30: no onset
    pl.decide(330, st(tilt=60))
    assert pl.U == 330


def test_age_cap_abstain():
    pl = Planner(P)
    plans = feed(pl, 300, lambda s: st(tilt=ramp(100, 0.4125, s)), lambda s: np.zeros(7), until_plan=False)
    assert plans and plans[0].abstain == "no_checkpoint" and plans[0].route == "A"
    assert not pl.a_armed


def test_checkpoint_requires_same_gripper_sign():
    pl = Planner(P)
    plans = feed(pl, 200, lambda s: st(tilt=min(ramp(100, 2.3, s), 69)),
                 lambda s: np.array([0, 0, 0, 0, 0, 0, 1.0 if s >= 112 else -1.0]), until_plan=False)
    # rows before decision 110 are open, fire-time row is closed -> the first same-sign checkpoint is 115 (tilt 34.5 > 30)
    assert plans[0].abstain == "no_checkpoint"


def test_custody_window_abstain_and_near_toggle_allowed():
    def run(toggle_moves):
        pl = Planner(P)
        def pos(s):
            if toggle_moves:
                return (0.005 * min(s, 140), 0, 1)
            return (0.005 * min(s, 112), 0, 1)
        g = lambda s: -1.0 if 112 <= s < 117 else 1.0
        return feed(pl, 300, lambda s: st(pos(s), tilt=min(ramp(100, 2.3, s), 69)),
                    lambda s: np.array([0, 0, 0, 0, 0, 0, g(s)]), until_plan=False)[0]
    far = run(True)
    assert far.abstain == "custody_window" and [s for s, _ in far.checks["custody_bad_toggles"]] == [112, 117]
    near = run(False)            # the arm does not move after step 112: toggles within 0.02 m of the fire position
    assert near.abstain is None and near.checks["gripper_changes_in_window"] == 2
    assert np.all(near.rows[:, 6] == 1.0)       # latch, the open toggles are not replayed


def test_horizon_guard_boundary():
    for t0, end_expected in ((375, 460), (380, 465)):      # fire = t0 + 40, L = 30: end = t0 + 85
        pl = Planner(P)
        plans = feed(pl, 520, lambda s: st(tilt=min(ramp(t0, 2.3, s), 69)), lambda s: np.zeros(7), until_plan=False)
        p = [x for x in plans if x.route == "A"][0]
        end = p.fire + p.L + 15
        assert end == end_expected and p.checks["horizon_end"] == end
        assert (p.abstain is None) == (end <= 460), (t0, end, p.abstain)
    pl = Planner(P)
    p = [x for x in feed(pl, 520, lambda s: st(tilt=min(ramp(400, 2.3, s), 69)), lambda s: np.zeros(7), until_plan=False)
         if x.route == "A"][0]
    assert p.abstain == "horizon" and p.checks["horizon_end"] > 460


def test_rx_route_only_after_release():
    pl = Planner(P)
    rx = lambda s: 3.5 if s >= 150 else 3.1
    g = lambda s: 1.0 if 50 <= s < 100 else -1.0      # close at 50, release at 100
    feed(pl, 160, lambda s: st(rx=rx(s)), lambda s: np.array([0, 0, 0, 0, 0, 0, g(s)]), until_plan=False)
    names = [(n, f.get("rx_route")) for n, f in pl.drain()]
    assert names == [("prr_onset", True)] and pl.rel == 100
    pl2 = Planner(P)                                   # never released: rx alone does not trigger
    feed(pl2, 160, lambda s: st(rx=rx(s)), lambda s: np.array([0, 0, 0, 0, 0, 0, 1.0]), until_plan=False)
    assert pl2.drain() == []


def test_route_b_stall_window_after_handback_and_cap():
    pos = lambda s: (0.0027 * min(max(s - 200, 0), 100), 0, 1)
    pl = Planner(P)
    plans = feed(pl, 520, lambda s: st(pos(s)), lambda s: np.zeros(7))
    p = plans[-1]
    assert p.route == "B" and p.fire == 355 and p.checkpoint == 285 and p.L == 70 and p.abstain is None
    pl.handback(430)
    # windows must start at or after the hand-back
    assert pl._stall_range(485) is None
    for d in range(430, 495, 5):
        assert pl.decide(d, st((0.1, 0, 1))) is None          # window start must be strictly after h = 430
    pb = pl.decide(495, st((0.1, 0, 1)))
    assert pb.route == "B" and pb.abstain == "no_checkpoint"
    pl.handback(500)
    assert pl.n_rest == 2 and pl.decide(505, st(tilt=80)) is None and pl.drain()[-1][0] == "prr_rearm"


def test_ortho_coin_public_and_deterministic():
    s0 = st((0.1, 0.2, 1.0))
    a = ortho_coin("put the pot on the stove", s0, 140)
    assert a == ortho_coin("put the pot on the stove", s0.copy(), 140)
    text = f"prr-ortho|put the pot on the stove|{state_digest(s0)}|140"
    assert a[1] == text and a[2] == hashlib.sha256(text.encode()).hexdigest()
    assert a[0] == (1 if int(a[2], 16) % 2 == 0 else -1)
    signs = {ortho_coin("x", s0, f)[0] for f in range(5, 400, 5)}
    assert signs == {1, -1}


def test_card_scalars_and_q99_pin():
    from harnessstudio.core import registry
    for name, variant in (("prr-v11", "prr"), ("prr-v11-hold", "hold"), ("prr-v11-ortho", "ortho")):
        cfg = registry.load(name)
        params = dict(cfg.controller.params)
        assert params["variant"] == variant and params["correction_max_decisions"] == 10
        assert (params["correction_kp_pos"], params["correction_kd_pos"], params["correction_kp_rot"],
                params["correction_kd_rot"], params["correction_pos_scale"], params["correction_rot_scale"]) == (1.0, 2.0, 1.0, 4.0, 0.05, 0.5)
        assert "rot_gain_rad_per_unit" not in params and "pos_gain_m_per_unit" not in params
        assert cfg.null == "shadow" and cfg.budgets.policy_calls == 110 and cfg.budgets.llm_calls == 0
        c = PRR(**params)
        assert c.params == Params.from_mapping(params)
    with pytest.raises(RuntimeError):
        PRR(checkpoint_rx_max=3.0)


class _Ctx:
    """Minimal context whose episode ends after `limit` executed rows (C4 cut path)."""
    def __init__(self, limit):
        self.limit, self.steps, self.events, self.instruction = limit, 0, [], "x"
        self.ended = False

    def observe(self):
        return {"states": st((0.0, 0.0, 1.0))}

    async def execute(self, a):
        if self.steps >= self.limit:
            self.ended = True
            return 0
        self.steps += 1
        return 1

    def record(self, name, **f):
        self.events.append(name)


def test_restoration_stops_when_execute_returns_zero():
    c = PRR()
    pl = Planner(c.params)
    plan = feed(pl, 200, lambda s: st(tilt=min(ramp(100, 2.3, s), 69)), lambda s: np.array([0.1, 0, 0, 0, 0, 0, 1.0]))[-1]
    ctx = _Ctx(limit=7)
    c.s0 = st()
    anyio.run(c._restore, ctx, plan)
    assert ctx.steps == 7 and "prr_restore_cut" in ctx.events and "prr_handback" not in ctx.events


def test_checkpoint_strictly_after_handback():
    pl = Planner(P)
    for s in range(0, 200):
        pl.note_row(s, st(), np.array([0, 0, 0, 0, 0, 0, -1.0]))
    pl.handback(200)
    tilt = lambda d: 20.0 if d == 200 else (40.0 if d < 230 else 60.0)
    plan = None
    for s in range(200, 260):
        if s % 5 == 0:
            plan = pl.decide(s, st((0.001 * s, 0, 1), tilt=tilt(s))) or plan
        pl.note_row(s, st((0.001 * s, 0, 1), tilt=tilt(s)), np.array([0, 0, 0, 0, 0, 0, -1.0]))
    # onset at 230 (= h + 30), fire at 245; the only tilt <= 30 decision is the hand-back decision 200 -> abstain
    assert plan is not None and plan.fire == 245 and plan.route == "A" and plan.abstain == "no_checkpoint"


def card_params():
    from harnessstudio.core import registry
    return dict(registry.load("prr-v11").controller.params)


def test_non_correction_params_identical_to_v100():
    """Every scalar outside the correction phase equals hs-prr 1.0.0's card (frozen package, read-only)."""
    import yaml
    from pathlib import Path
    v100 = yaml.safe_load(Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio/methods/prr/hs_prr/paper.yaml").read_text())["params"]
    v11 = card_params()
    corr_old = {"correction_max_decisions", "pos_gain_m_per_unit", "rot_gain_rad_per_unit"}
    corr_new = {"correction_max_decisions", "correction_kp_pos", "correction_kd_pos", "correction_kp_rot", "correction_kd_rot",
                "correction_pos_scale", "correction_rot_scale"}
    assert {k: v for k, v in v100.items() if k not in corr_old} == {k: v for k, v in v11.items() if k not in corr_new}


def test_horizon_guard_uses_new_budget():
    p = Params.from_mapping(card_params())
    assert p.correction_max_decisions == 10
    for t0, end_expected in ((335, 460), (340, 465)):      # fire = t0 + 40, L = 30: end = fire + L + 50 + 5
        pl = Planner(p)
        plans = feed(pl, 520, lambda s: st((0.001 * s, 0, 1), tilt=min(ramp(t0, 2.3, s), 69)), lambda s: np.zeros(7),
                     until_plan=False)
        q = [x for x in plans if x.route == "A"][0]
        assert q.checks["horizon_end"] == end_expected and (q.abstain is None) == (end_expected <= 460)


@pytest.mark.parametrize("dist", [0.04, 0.10, 0.20, 0.35])
@pytest.mark.parametrize("deg", [10.0, 25.0])
def test_correction_closes_scripted_error_on_osc_stub(dist, deg):
    """W1 on the kinematic stub: the 1.1.0 correction (+ settle) closes 4-35 cm / 10-25 deg to <= 2 cm / 10 deg within
    10 decisions, from 6 fixed random directions and initial velocities; 1.0.0's law is shown failing the same cases."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from osc_stub import OSCStub
    rng = np.random.default_rng(int(dist * 1000 + deg))
    c = PRR(**{k: v for k, v in card_params().items()})
    R_cp = exp([np.pi, 0, 0])
    cp_state = np.concatenate([[0.0, 0.0, 1.0], log(R_cp), [0.04, -0.04]])
    worst = (0.0, 0.0)
    for _ in range(6):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        plant = OSCStub(cp_state[:3] + dist * d, exp(np.radians(deg) * ax) @ R_cp)
        plant.v = 0.2 * rng.normal(size=3) * 0.01 / 0.05       # residual reversal motion, up to ~1 cm/step
        prev = plant.state()
        steps = 0
        for j in range(10):
            stt = plant.state()
            ep, eg = pose_err(cp_state, stt)
            if ep < 0.01 and eg < 5.0:
                break
            for _ in range(5):
                stt = plant.state()
                row = c._correction_row(cp_state, stt, prev, -1.0)
                assert np.max(np.abs(row[:6])) <= 1.0 + 1e-12
                prev = stt
                plant.step(row); steps += 1
        for _ in range(5):
            plant.step(np.zeros(7))
        ep, eg = pose_err(cp_state, plant.state())
        worst = (max(worst[0], ep), max(worst[1], eg))
        assert steps <= 50
    assert worst[0] <= 0.02 and worst[1] <= 10.0, worst


def pose_err(cp_state, st):
    from hs_prr_v11.controller import pose_error
    return pose_error(cp_state, st)



def test_correction_law_specification():
    """1.1.0 law: u = (Kp e - Kd v)/scale, direction-preserving saturation, goal scale 0.05 m / 0.5 rad; hold zero rows;
    ortho rotation-only."""
    params = card_params()
    c = PRR(**params)
    cp = np.concatenate([[0.0, 0.0, 1.0], log(exp([np.pi, 0, 0])), [0.04, -0.04]])
    R = exp(np.radians(50) * np.array([0.0, 0.6, 0.8])) @ exp([np.pi, 0, 0])
    st_ = np.concatenate([[0.12, -0.03, 1.02], log(R), [0.04, -0.04]])
    row = c._correction_row(cp, st_, st_, 1.0)                     # prev == current: v = w = 0
    e = cp[:3] - st_[:3]
    assert np.isclose(np.max(np.abs(row[:3])), 1.0) and np.allclose(row[:3] / np.linalg.norm(row[:3]), e / np.linalg.norm(e))
    er = rot_error(cp[3:6], st_[3:6])
    assert np.isclose(np.max(np.abs(row[3:6])), 1.0) and np.allclose(row[3:6] / np.linalg.norm(row[3:6]), er / np.linalg.norm(er))
    assert row[6] == 1.0
    small = np.concatenate([cp[:3] + [0.01, 0.0, -0.005], cp[3:]])  # inside the unsaturated region: exact goal law
    r2 = c._correction_row(cp, small, small, -1.0)
    assert np.allclose(r2[:3], np.array([-0.01, 0.0, 0.005]) / 0.05) and np.allclose(r2[3:6], 0.0, atol=1e-9)
    moving = np.concatenate([small[:3] + [0.0, 0.0, 0.0], cp[3:]]); prev = moving.copy(); prev[:3] -= [0.002, 0.0, 0.0]
    r3 = c._correction_row(cp, moving, prev, -1.0)                  # damping: v = +2 mm/step in x -> -2*0.002/0.05
    assert np.allclose(r3[:3], (np.array([-0.01, 0.0, 0.005]) - 2.0 * np.array([0.002, 0.0, 0.0])) / 0.05)
    w_prev = np.concatenate([cp[:3], log(exp([0.0, 0.0, -0.01]) @ exp(cp[3:6])), cp[6:]])   # rotated +0.01 rad about z since prev
    r4 = c._correction_row(cp, cp, w_prev, -1.0)
    assert np.allclose(r4[3:6], (np.zeros(3) - 4.0 * np.array([0.0, 0.0, 0.01])) / 0.5, atol=1e-9)
    hold = PRR(**dict(params, variant="hold"))._correction_row(cp, st_, st_, 1.0)
    assert np.all(hold[:6] == 0) and hold[6] == 1.0
    ortho = PRR(**dict(params, variant="ortho"))._correction_row(cp, st_, st_, 1.0)
    assert np.all(ortho[:3] == 0) and np.allclose(ortho[3:6], row[3:6])
