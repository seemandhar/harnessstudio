"""Unit tests for hs_tilt_guard (CPU). Covers: component on/off, fixed order (C8/T10), clip ordering (C7/T5),
k singularity (C6/T6), analytic sign (T3), consecutive-substitution cap (general C4), bit identity when inactive
(T7), yaw control (T9), public coins (C3/C9), detector semantics (C2), equality with the frozen hs_band_guard."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from hs_tilt_guard import ENTRIES  # noqa: E402
from hs_tilt_guard import action_op as A  # noqa: E402
from hs_tilt_guard import geometry as geo  # noqa: E402

NS = A.load_norm_stats("assets/pi0_fast_libero_norm_stats.json", "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e")
AMEAN = np.asarray(NS["actions"]["mean"])


class StubGuard:
    """Same signature as hs_band_guard BandGuardActionOp; adds +0.25 on rx of every row and records calls."""
    calls: list = []

    def __init__(self, **kw):
        self._diag = {}
        StubGuard.calls = []

    def transform(self, chunk, obs, memory):
        StubGuard.calls.append(np.array(chunk, copy=True))
        out = np.array(chunk, copy=True)
        out[:, 3] += 0.25
        self._diag = {"active": True}
        return out

    def diagnostics(self):
        return dict(self._diag)


def make(**kw):
    base = dict(norm_stats_sha256="4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e")
    base.update(kw)
    return A.TiltGuardActionOp(**base)


def aa_for_tilt(tilt_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    """Axis-angle of R = Rz(az) Ry(tilt) Rx(pi): approach axis tilted by tilt_deg from straight down."""
    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rz(np.radians(azimuth_deg)) @ Ry(np.radians(tilt_deg)) @ Rx(np.pi)
    return geo.log_so3(R)


def state(tilt_deg=3.0, az=0.0, width=0.03, pos=(0.0, 0.0, 1.0)):
    aa = aa_for_tilt(tilt_deg, az)
    if aa[0] < 0:   # robosuite / quat_no_antipodal convention near the start pose: rx ~ +pi (same rotation)
        aa = aa * (1 - 2 * np.pi / np.linalg.norm(aa))
    return np.array([*pos, *aa, width / 2, -width / 2], dtype=np.float64)


def chunk(rng, grip=1.0, rows=10):
    c = rng.uniform(-0.5, 0.5, size=(rows, 7))
    c[:, 6] = grip
    return c


def zero_chunk(s, rows=10):
    z = np.tile(AMEAN.copy(), (rows, 1))
    z[:, :6] += s[:6]
    return z


def drive_to_boundary(op, rng, s_closed, s_open=None):
    """4 close decisions holding (width > 0.004), then an open decision = boundary at decision 4."""
    for _ in range(4):
        op.transform(chunk(rng, grip=1.0), {"states": s_closed}, {})
    return op.transform(chunk(rng, grip=-1.0), {"states": s_open if s_open is not None else s_closed}, {})


# ---------------------------------------------------------------------------------------------------------------- T2
def test_rotmat_antipodal_invariance_and_tilt_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        th = rng.uniform(np.pi + 1e-3, 2 * np.pi - 1e-3)
        aa = ax * th
        assert np.allclose(geo.rotmat(aa), geo.rotmat(aa * (1 - 2 * np.pi / th)), atol=1e-9)
    assert abs(np.degrees(geo.tilt_rad(aa_for_tilt(3.0, 40))) - 3.0) < 1e-6


# ---------------------------------------------------------------------------------------------------------------- T3
@pytest.mark.parametrize("tilt", [60.0, 90.0, 135.0])
@pytest.mark.parametrize("az", [0.0, 70.0, 200.0, 315.0])
def test_analytic_sign(tilt, az):
    R = geo.rotmat(aa_for_tilt(tilt, az))
    k, src = geo.envelope_axis(R, None, 1e-3)
    assert src == "cross"
    t0 = geo.tilt_of_matrix(R)
    push = geo.push_magnitude(t0, np.radians(55), 0.5, 0.2)
    assert push > 0
    # first-order: rotation about +k decreases tilt
    for g in (0.136, 1.0):
        Rm = geo.rotmat(g * 0.5 * 0.3 * k) @ R
        assert geo.tilt_of_matrix(Rm) < t0
    minus = (-0.6 * k)[None, :]
    plus = (0.6 * k)[None, :]
    new_m, rem_m, _ = geo.bte_rotation_rows(minus, k, push)
    new_p, rem_p, _ = geo.bte_rotation_rows(plus, k, push)
    assert rem_m[0] < 0 and rem_p[0] == 0.0
    assert np.allclose(new_m[0], np.clip(push * k, -1, 1), atol=1e-12)        # -k part removed, push added
    assert np.allclose(new_p[0], np.clip(0.6 * k + push * k, -1, 1), atol=1e-12)  # +k kept
    rng = np.random.default_rng(int(tilt + az))
    for _ in range(50):
        row = rng.uniform(-1, 1, size=(1, 3))
        new, _, _ = geo.bte_rotation_rows(row, k, push)
        w = 0.5 * new[0]
        assert -(w @ k) <= 1e-12                                   # first-order dtilt <= 0
        for g in (0.136, 1.0):
            Rn = geo.rotmat(g * w) @ R
            # composed (finite) tilt below start whenever the command has a non-trivial +k part
            if (w @ k) > 0.02:
                assert geo.tilt_of_matrix(Rn) < t0 + 1e-9 or g == 1.0


# ---------------------------------------------------------------------------------------------------------------- T5
def test_clip_ordering_random_rows():
    rng = np.random.default_rng(1)
    for _ in range(20000):
        k = rng.normal(size=3)
        k[2] = 0.0
        k /= np.linalg.norm(k)
        rows = rng.uniform(-3, 3, size=(5, 3))
        push = rng.uniform(0.0, 0.2)
        new, _, _ = geo.bte_rotation_rows(rows, k, push)
        final = np.clip(new, -1, 1)                               # robosuite's own per-component clip
        assert np.all(final @ k >= -1e-9)
        assert np.all(np.abs(new) <= 1 + 1e-12)


def test_bte_rows_equal_literal_c7_formula():
    """Output equals the literal C7 order (clip -> remove negative k-part -> push -> re-clip) wherever that keeps
    k >= 0; the uniform-rescale fallback fires only where the literal re-clip would make k negative."""
    rng = np.random.default_rng(21)
    n_fb = 0
    for _ in range(20000):
        k = rng.normal(size=3)
        k[2] = 0.0
        k /= np.linalg.norm(k)
        row = rng.uniform(-3, 3, size=3)
        push = rng.uniform(0.0, 0.2)
        c = np.clip(row, -1, 1)
        kc = c @ k
        if kc < 0:
            c = c - kc * k
        lit = np.clip(c + push * k, -1, 1)
        new, removed, resc = geo.bte_rotation_rows(row[None, :], k, push)
        if lit @ k >= 0:
            assert not resc[0] and np.array_equal(new[0], lit)
        else:
            n_fb += 1
            assert resc[0] and new[0] @ k > 0
        assert removed[0] == min(np.clip(row, -1, 1) @ k, 0.0)
    assert n_fb < 0.06 * 20000   # measured 851/20000 on uniform [-3, 3] rows; real-data rate is reported by T4


def test_clip_ordering_needs_preclip():
    """Projecting the unclipped vector and clipping afterwards can reinstate a negative k part (why C7 exists)."""
    k = np.array([0.6, 0.8, 0.0])
    row = np.array([[3.0, -2.9, 0.0]])
    naive = row[0] - min(row[0] @ k, 0) * k
    assert np.clip(naive, -1, 1) @ k < 0
    new, _, _ = geo.bte_rotation_rows(row, k, 0.0)
    assert new[0] @ k >= -1e-12


# ---------------------------------------------------------------------------------------------------------------- T6
@pytest.mark.parametrize("tilt", [179.95, 180.0, 0.05])
def test_singularity_fallback(tilt):
    R = geo.rotmat(aa_for_tilt(tilt, 30.0))
    k1, src1 = geo.envelope_axis(R, None, 1e-3)
    k2, src2 = geo.envelope_axis(R, None, 1e-3)
    assert np.all(np.isfinite(k1)) and np.array_equal(k1, k2) and src1 == src2
    assert src1 in ("x_axis", "unit_x") and abs(np.linalg.norm(k1) - 1) < 1e-12 and abs(k1[2]) < 1e-12
    prev = np.array([0.0, 1.0, 0.0])
    k3, src3 = geo.envelope_axis(R, prev, 1e-3)
    assert src3 == "previous" and np.array_equal(k3, prev)


def test_singularity_in_op_no_nan():
    rng = np.random.default_rng(2)
    for tilt in (179.95, 180.0):
        op = make(envelope="bte")
        drive_to_boundary(op, rng, state(5.0))
        s = state(tilt, 30.0)
        outs = [op.transform(chunk(rng, grip=-1.0), {"states": s}, {}) for _ in range(3)]
        for o in outs:
            assert np.all(np.isfinite(o))
        assert op.stats["b_active"] == 3


# ---------------------------------------------------------------------------------------------------------------- T7
def test_bitwise_when_inactive_random():
    rng = np.random.default_rng(3)
    op = make(envelope="bte")
    for i in range(10000):
        c = chunk(rng, grip=float(rng.choice([-1.0, 1.0])))
        s = state(float(rng.uniform(0, 175)), float(rng.uniform(0, 360)), width=float(rng.uniform(0, 0.002)))
        # width <= 0.004 => close decisions never count, so no boundary is ever found
        assert op.transform(c, {"states": s}, {}) is c
    assert op.detector.boundary is None


def test_t7_inactive_conditions_and_active_columns():
    rng = np.random.default_rng(4)
    op = make(envelope="bte")
    s_hold = state(5.0)
    # before any boundary, high tilt: bitwise
    c = chunk(rng)
    assert op.transform(c, {"states": state(120.0)}, {}) is c
    op = make(envelope="bte")
    out = drive_to_boundary(op, rng, s_hold)                       # boundary decision with tilt 5 <= theta
    assert op.detector.boundary == 4
    c = chunk(rng, grip=-1.0)
    assert op.transform(c, {"states": state(50.0)}, {}) is c       # tilt <= theta: bitwise
    c = chunk(rng, grip=-1.0)
    assert op.transform(c, {"states": None}, {}) is c              # missing state
    c = chunk(rng, grip=-1.0)
    assert op.transform(c, {"states": state(120.0)[:7]}, {}) is c  # 7-d state
    s = state(120.0)
    z = zero_chunk(s)
    assert op.transform(z, {"states": s}, {}) is z                 # native zero chunk (Z off): bitwise
    c = chunk(rng, grip=-1.0)
    out = op.transform(c, {"states": state(120.0, 45.0)}, {})      # active
    assert out is not c
    assert np.array_equal(out[:, :3], c[:, :3]) and np.array_equal(out[:, 6], c[:, 6])
    assert np.array_equal(out[5:], c[5:])                          # rows 5-9 untouched (only rows[:5] execute)
    assert not np.array_equal(out[:5, 3:6], c[:5, 3:6])
    # outside the window (30 decisions after the boundary at decision 4): bitwise again
    while op.t < 4 + 30:
        op.transform(chunk(rng, grip=-1.0), {"states": state(120.0)}, {})
    c = chunk(rng, grip=-1.0)
    assert op.transform(c, {"states": state(120.0)}, {}) is c


def test_window_includes_boundary_decision_and_30_decisions():
    rng = np.random.default_rng(5)
    op = make(envelope="bte")
    for _ in range(4):
        op.transform(chunk(rng, 1.0), {"states": state(5.0)}, {})
    op.transform(chunk(rng, -1.0), {"states": state(100.0)}, {})   # boundary decision 4, tilt > theta: active
    assert op.stats["b_active"] == 1
    for _ in range(40):
        op.transform(chunk(rng, -1.0), {"states": state(100.0)}, {})
    assert op.stats["b_active"] == 30


# ---------------------------------------------------------------------------------------------------------------- C2
def test_detector_semantics():
    d = A.BoundaryDetector(4, 0.004)
    seq = [(1, 0.03, False)] * 3 + [(1, 0.001, False)] + [(-1, 0.03, True)] + [(1, 0.03, False)] + [(-1, 0.03, False)]
    hits = [d.update(i, g, w, z) for i, (g, w, z) in enumerate(seq)]
    # empty close (width 0.001) and zero chunk neither increment nor reset: run reaches 4 at decision 5, boundary 6
    assert hits == [False] * 6 + [True] and d.boundary == 6
    d2 = A.BoundaryDetector(4, 0.004)
    for i, (g, w) in enumerate([(1, .03)] * 3 + [(-1, .03)] + [(1, .03)] * 3 + [(-1, .03)]):
        d2.update(i, g, w, False)
    assert d2.boundary is None and d2.run == 0                      # open after 3 resets
    d3 = A.BoundaryDetector(4, 0.004)
    for i, (g, w) in enumerate([(1, .03)] * 4 + [(-1, .03)] + [(1, .03)] * 4 + [(-1, .03)]):
        d3.update(i, g, w, False)
    assert d3.boundary == 4                                         # first boundary only


# ---------------------------------------------------------------------------------------------------------------- T9
def test_yaw_control():
    rng = np.random.default_rng(6)
    for _ in range(200):
        tilt = float(rng.uniform(56, 175))
        R = geo.rotmat(aa_for_tilt(tilt, float(rng.uniform(0, 360))))
        k, _ = geo.envelope_axis(R, None, 1e-3)
        push = geo.push_magnitude(geo.tilt_of_matrix(R), np.radians(55), 0.5, 0.2)
        rows = rng.uniform(-1.5, 1.5, size=(5, 3))
        new, _, _ = geo.bte_rotation_rows(rows, k, push)
        norms = np.linalg.norm(new - np.clip(rows, -1, 1), axis=1)
        for sign in (1, -1):
            yaw = geo.yaw_rotation_rows(rows, norms, sign)
            delta = yaw - np.clip(rows, -1, 1)
            assert np.allclose(delta[:, :2], 0.0, atol=0) and np.allclose(np.abs(delta[:, 2]), norms, atol=1e-12)
            assert np.allclose(delta @ k, 0.0, atol=1e-12)          # tilt-neutral to first order
            assert abs(np.linalg.norm(delta, axis=1) - norms).max() < 1e-9
        # rotation about world z leaves tilt unchanged exactly
        assert abs(geo.tilt_of_matrix(geo.rotmat(np.array([0, 0, 0.3])) @ R) - geo.tilt_of_matrix(R)) < 1e-9
    coins = [A.yaw_coin(20261101, d) for d in range(64)]
    assert coins == [A.yaw_coin(20261101, d) for d in range(64)] and set(coins) == {1, -1}


def test_yaw_op_matches_bte_norm_same_state():
    rng = np.random.default_rng(7)
    ops = {m: make(envelope=m) for m in ("bte", "yawctrl")}
    rows = [chunk(rng, 1.0) for _ in range(4)] + [chunk(rng, -1.0)] + [chunk(rng, -1.0) for _ in range(6)]
    states = [state(5.0)] * 5 + [state(float(rng.uniform(60, 170)), float(rng.uniform(0, 360))) for _ in range(6)]
    for c, s in zip(rows, states):
        outs = {m: op.transform(c, {"states": s}, {}) for m, op in ops.items()}
        if ops["bte"].stats["b_active"] and ops["bte"].diagnostics().get("envelope"):
            db = np.linalg.norm(outs["bte"][:5, 3:6] - np.clip(c[:5, 3:6], -1, 1), axis=1)
            dy = np.linalg.norm(outs["yawctrl"][:5, 3:6] - np.clip(c[:5, 3:6], -1, 1), axis=1)
            assert np.allclose(db, dy, atol=1e-9)
            assert np.array_equal(outs["yawctrl"][:, :3], c[:, :3]) and np.array_equal(outs["yawctrl"][:, 6], c[:, 6])
    assert ops["bte"].stats["b_active"] == ops["yawctrl"].stats["b_active"] == 6


# --------------------------------------------------------------------------------------------- Z cap / T10 order
def test_zero_substitution_cap_and_tail():
    rng = np.random.default_rng(8)
    op = make(zero="substitute")
    s = state(5.0)
    c0 = chunk(rng, grip=1.0)
    out0 = op.transform(c0, {"states": s}, {})
    assert out0 is c0
    outs = []
    for i in range(4):
        z = zero_chunk(s)
        outs.append((z, op.transform(z, {"states": s}, {})))
    (z1, o1), (z2, o2), (z3, o3), (z4, o4) = outs
    assert np.array_equal(o1[:5, :6], c0[5:10, :6]) and np.all(o1[:, 6] == 1.0)  # tail + latched close gripper
    assert np.all(o1[5:, :6] == 0.0)
    assert np.all(o2[:, :6] == 0.0) and np.all(o2[:, 6] == 1.0)                   # run 2: hold
    assert o3 is z3 and o4 is z4                                                   # run >= 3: native passthrough
    assert op.stats["z_tail"] == 1 and op.stats["z_hold"] == 1 and op.stats["z_passthrough"] == 2
    c5 = chunk(rng, grip=1.0)
    assert op.transform(c5, {"states": s}, {}) is c5 and op.zero_run == 0
    z6 = zero_chunk(s)
    o6 = op.transform(z6, {"states": s}, {})
    assert np.array_equal(o6[:5, :6], c5[5:10, :6])                                # cap resets after a valid chunk


def test_zero_first_decision_hold_with_native_gripper():
    s = state(5.0)
    op = make(zero="substitute")
    z = zero_chunk(s)
    o = op.transform(z, {"states": s}, {})
    assert np.all(o[:, :6] == 0.0) and np.all(o[:, 6] == z[0, 6])
    assert op.stats["z_hold"] == 1


def test_zero_short_previous_chunk_gives_hold():
    rng = np.random.default_rng(9)
    s = state(5.0)
    op = make(zero="substitute")
    op.transform(chunk(rng, 1.0, rows=5), {"states": s}, {})
    o = op.transform(zero_chunk(s, rows=5), {"states": s}, {})
    assert np.all(o[:, :6] == 0.0) and op.stats["z_hold"] == 1


def test_composite_order_tail_is_post_guard_and_guard_skipped_on_substitution():
    rng = np.random.default_rng(10)
    op = make(guard=True, envelope="bte", zero="substitute", guard_factory="test_tilt_guard:StubGuard")
    s_hold = state(5.0)
    for _ in range(4):
        op.transform(chunk(rng, 1.0), {"states": s_hold}, {})
    c_b = chunk(rng, -1.0)
    s_hi = state(110.0, 20.0)
    out_b = op.transform(c_b, {"states": s_hi}, {})                 # boundary decision: G then B
    guarded = c_b.copy()
    guarded[:, 3] += 0.25
    assert len(StubGuard.calls) == 5 and np.array_equal(StubGuard.calls[-1], c_b)
    assert np.array_equal(out_b[5:], guarded[5:])                   # rows 5-9 = post-G
    assert np.array_equal(op.prev_chunk, guarded)                   # stored tail source = post-G, pre-B
    z = zero_chunk(s_hi)
    out_z = op.transform(z, {"states": s_hi}, {})                   # Z substitutes, G skipped, B applied last
    assert len(StubGuard.calls) == 5                                # G not called on the substituted chunk
    d = op.diagnostics()
    assert d["z_mode"] == "tail" and d["envelope"]["active"]
    expect = np.zeros_like(z)
    expect[:5] = guarded[5:10]
    expect[:, 6] = out_b[4, 6]                                      # latched = last executed gripper (-1)
    R = geo.rotmat(s_hi[3:6])
    k, _ = geo.envelope_axis(R, op.prev_k, 1e-3)
    push = geo.push_magnitude(geo.tilt_of_matrix(R), np.radians(55), 0.5, 0.2)
    new, _, _ = geo.bte_rotation_rows(expect[:5, 3:6], np.asarray(d["envelope"]["k"]), push)
    expect[:5, 3:6] = new
    assert np.allclose(out_z, expect, atol=1e-12)
    assert not op.detector.boundary != 4                            # zero chunk is never a release


def test_memory_resets_with_fresh_instance():
    rng = np.random.default_rng(11)
    op = make(guard=True, envelope="bte", zero="substitute", guard_factory="test_tilt_guard:StubGuard")
    drive_to_boundary(op, rng, state(5.0))
    fresh = make(guard=True, envelope="bte", zero="substitute", guard_factory="test_tilt_guard:StubGuard")
    assert fresh.t == 0 and fresh.detector.boundary is None and fresh.prev_chunk is None and fresh.last_grip is None
    assert op.detector.boundary == 4


def test_zero_signctrl_public_coins_and_magnitude():
    rng = np.random.default_rng(12)
    s = state(5.0)
    runs = []
    for _ in range(2):
        op = make(zero="signctrl")
        c0 = chunk(np.random.default_rng(99), grip=1.0)
        op.transform(c0, {"states": s}, {})
        z = zero_chunk(s)
        o = op.transform(z, {"states": s}, {})
        runs.append((o, op.diagnostics()["z_coins"]))
        sub = np.zeros_like(z)
        sub[:5] = c0[5:10]
        sub[:, 6] = 1.0
        r = np.asarray(op.diagnostics()["z_coins"], float)
        exp = z + r * (sub - z)
        exp[:, 6] = np.clip(exp[:, 6], -1, 1)
        assert np.allclose(o, exp, atol=1e-12)
        assert np.allclose(np.abs(o[:, :6] - z[:, :6]), np.abs(sub[:, :6] - z[:, :6]), atol=1e-12)
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]   # deterministic, public inputs
    seed = __import__("hashlib").sha256(s.tobytes()).digest()
    assert runs[0][1] == A.zero_coins(seed, 1, 7)
    del rng


# ------------------------------------------------------------------------------------------ hs_band_guard (frozen G)
def test_guard_params_equal_frozen_band_guard_card():
    import hs_band_guard
    from harnessstudio.core.card import load_card
    bg = load_card(os.path.join(os.path.dirname(hs_band_guard.__file__), "paper.yaml"))["params"]
    for v in ENTRIES:
        card = load_card(os.path.join(os.path.dirname(A.__file__), f"paper_{v}.yaml"))
        assert card["params"]["guard_params"] == bg
        assert card["params"]["guard_factory"] == "hs_band_guard.action_op:BandGuardActionOp"


def test_guard_only_composite_equals_band_guard():
    from hs_band_guard.action_op import BandGuardActionOp, is_openpi_zero_chunk as bg_zero
    import hs_band_guard
    from harnessstudio.core.card import load_card
    params = load_card(os.path.join(os.path.dirname(hs_band_guard.__file__), "paper.yaml"))["params"]
    rng = np.random.default_rng(13)
    bg = BandGuardActionOp(**params)
    op = make(guard=True, guard_params=params)
    n_active = 0
    for i in range(3000):
        s = state(3.0)
        s[3] = float(rng.uniform(1.0, 4.5))
        c = chunk(rng, grip=float(rng.choice([-1.0, 1.0])))
        c[:, 3] = s[3] + rng.uniform(-3.5, -1.2, size=10) * (1 if rng.random() < 0.5 else 0) + (
            rng.uniform(-3.4, -3.3) if rng.random() < 0.3 else 0)
        if rng.random() < 0.05:
            c = zero_chunk(s)
        a, b = bg.transform(c, {"states": s}, {}), op.transform(c, {"states": s}, {})
        assert np.array_equal(np.asarray(a), np.asarray(b))
        assert (a is c) == (b is c)
        n_active += a is not c
        assert bg_zero(c, s, AMEAN, 1e-4) == A.is_openpi_zero_chunk(c, s, AMEAN, 1e-4)
    assert n_active > 50


# ---------------------------------------------------------------------------------------------- entry points/cards
def test_entry_points_build_with_expected_switches():
    from harnessstudio.core import registry
    expect = {"gb": (True, "bte", "off"), "gyaw": (True, "yawctrl", "off"), "gbz": (True, "bte", "substitute"),
              "gz": (True, "off", "substitute"), "gbzsign": (True, "bte", "signctrl"),
              "gzsign": (True, "off", "signctrl"), "z": (False, "off", "substitute"), "zsign": (False, "off", "signctrl")}
    fps = set()
    for v, (g, e, z) in expect.items():
        cfg = registry.load(f"tilt-guard-{v}")
        op = cfg.slots["action_op"].build()
        assert (op.use_guard, op.envelope, op.zero) == (g, e, z)
        assert cfg.budgets.llm_calls == 0 and cfg.budgets.policy_calls is None and cfg.null == "derived"
        assert cfg.card["development_only"] == (e == "yawctrl" or z == "signctrl")
        p = cfg.card["params"]
        assert (p["theta_deg"], p["window_steps"], p["hold_decisions"], p["kappa_per_rad"], p["omega_max"],
                p["zero_max_consecutive"], p["width_min"]) == (55.0, 150, 4, 0.5, 0.2, 2, 0.004)
        fps.add(cfg.fingerprint())
    assert len(fps) == len(expect)
