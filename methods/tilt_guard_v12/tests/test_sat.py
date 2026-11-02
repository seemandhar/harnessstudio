"""tilt-guard v1.1 saturation-arming tests (r2 review coder_tests S3 random part, S4, S5, S6, S8, S9, S10).
S1, S2, S3 (replay part) and S7 run on recorded traces in scripts/replay_sat.py."""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from harnessstudio.core import registry  # noqa: E402
from hs_tilt_guard_v12 import action_op as A  # noqa: E402
from hs_tilt_guard_v12 import geometry as geo  # noqa: E402
from test_tilt_guard_v10_suite import AMEAN, NS, make, state, zero_chunk  # noqa: E402

Q01 = np.asarray(NS["actions"]["q01"])
Q99 = np.asarray(NS["actions"]["q99"])


class StubUnpin:
    """Guard with BandGuardActionOp's signature that rewrites rx to s_rx - 2.4 (mid-band, unpinned) on every row."""

    def __init__(self, **kw):
        self._diag = {}

    def transform(self, chunk, obs, memory):
        out = np.array(chunk, copy=True)
        out[:, 3] = float(obs["states"][3]) - 2.4
        self._diag = {"active": True}
        return out

    def diagnostics(self):
        return dict(self._diag)


def sat_rows(s, mag=0.8, rows=10, grip=-1.0, pin_row=None, pin_dim=4, pin_val=4.4):
    """Rotation rows along the tilt-increasing axis d x u (c = mag), optional pin in one row/dim."""
    R = geo.rotmat(s[3:6])
    v = np.cross(geo.DOWN, R[:, 2])
    v /= np.linalg.norm(v)
    c = np.zeros((rows, 7))
    c[:, 3:6] = mag * v
    c[:, 6] = grip
    if pin_row is not None:
        c[pin_row, pin_dim] = pin_val
    return c


def sat_op(mode="on", **kw):
    return make(envelope="bte", saturation_arming=mode, **kw)


# ------------------------------------------------------------------------------------------------------------ S4
def test_s4_synthetic_arming_units():
    s = state(60.0, 90.0)
    # pinned (ry high in row 0, clip leaves c = 0.8), no boundary, tilt 60 -> armed and envelope active
    op = sat_op()
    c = sat_rows(s, pin_row=0)
    out = op.transform(c, {"states": s}, {})
    d = op.diagnostics()
    assert d["sat_armed_now"] and d["sat_first"] == 0 and d["sat_c"] > 0.5 and [0, 4, "high"] in d["sat_pinned"]
    assert d["envelope"]["active"] and d["envelope"]["route"] == "saturation" and out is not c
    assert op.detector.boundary is None
    # same, unpinned -> not armed, bitwise
    op = sat_op()
    c = sat_rows(s)
    assert op.transform(c, {"states": s}, {}) is c
    assert not op.diagnostics()["sat_armed_now"] and op.diagnostics()["sat_pinned"] == []
    # c == c_th exactly -> not armed (strict)
    op = sat_op()
    c = sat_rows(s, pin_row=0)
    _, info = op._sat_arm(c, s)
    op.c_th = info["sat_c"]
    assert op.transform(c, {"states": s}, {}) is c and op.sat_first is None
    # tilt == arm_tilt exactly -> not armed (strict)
    s25 = state(25.0, 90.0)
    op = sat_op()
    op.arm_tilt = geo.tilt_rad(s25[3:6])
    c = sat_rows(s25, pin_row=0)
    op.transform(c, {"states": s25}, {})
    assert op.sat_first is None
    op2 = sat_op()
    op2.transform(c, {"states": s25}, {})
    assert op2.sat_first is None or op2.diagnostics()["sat_tilt_deg"] > 25.0
    # axis guard: tilt 5 and 175 deg -> |d x u| < 0.17 -> not armed
    for tilt in (5.0, 175.0):
        st = state(tilt, 90.0)
        op = sat_op()
        R = geo.rotmat(st[3:6])
        v = np.cross(geo.DOWN, R[:, 2])
        assert np.linalg.norm(v) < 0.17
        c = np.zeros((10, 7))
        c[:, 3:6] = 0.9 * v / np.linalg.norm(v)
        c[0, 4] = 9.0
        c[:, 6] = -1
        op.transform(c, {"states": st}, {})
        assert op.sat_first is None and op.diagnostics().get("sat_axis_guard")
    # native zero chunk -> never armed
    op = sat_op()
    z = zero_chunk(s)
    assert op.transform(z, {"states": s}, {}) is z and op.sat_first is None and not op.diagnostics()["sat_armed_now"]
    assert "sat_c" not in op.diagnostics() and "sat_pinned" not in op.diagnostics()   # arming not evaluated (spec b)
    # pinned only in rows >= 5 -> not armed
    op = sat_op()
    c = sat_rows(s, pin_row=6)
    assert op.transform(c, {"states": s}, {}) is c and op.sat_first is None


# ------------------------------------------------------------------------------------------------------------ S5
def test_s5_arming_reads_native_chunk():
    s = state(60.0, 90.0)
    c = sat_rows(s, mag=1.8)                      # rx = -1.8 -> m_rx = -1.8 - 1.58 <= q01 + 0.02 (pinned low)
    assert np.any(c[:5, 3] - s[3] <= Q01[3] + 0.02)
    op = make(envelope="bte", saturation_arming="on")   # v1.2 has no G: the native chunk is what B sees
    op.transform(c, {"states": s}, {})
    assert op.sat_first == 0
    # a Z-substituted decision cannot arm (native zero chunk)
    op = make(envelope="bte", zero="substitute", saturation_arming="on")
    op.transform(sat_rows(s), {"states": s}, {})  # unpinned valid chunk
    out = op.transform(zero_chunk(s), {"states": s}, {})
    d = op.diagnostics()
    assert d["z_mode"] == "tail" and not d["sat_armed_now"] and op.sat_first is None
    assert out is not None


# ------------------------------------------------------------------------------------------------------------ S6
def test_s6_window_first_event_union_and_reset():
    s = state(60.0, 90.0)
    op = sat_op()
    for _ in range(45):
        op.transform(sat_rows(s, pin_row=0), {"states": s}, {})
    assert op.sat_first == 0 and op.stats["b_active"] == 30 and op.stats["n_sat_only_active"] == 30   # no re-arm
    # tilt <= theta inside the window -> inactive
    op = sat_op()
    op.transform(sat_rows(s, pin_row=0), {"states": s}, {})
    s50 = state(50.0, 90.0)
    c = sat_rows(s50)
    assert op.transform(c, {"states": s50}, {}) is c and op.diagnostics()["sat_window_open"]
    # OR with the boundary window: arm at 0, window closes at 30, boundary at 36 reopens B
    op = sat_op()
    seq = [(sat_rows(s, pin_row=0), s)] * 32
    s_hold = state(60.0, 90.0)
    s_hold[6], s_hold[7] = 0.015, -0.015
    for _ in range(4):
        seq.append((sat_rows(s_hold, grip=1.0), s_hold))
    seq.append((sat_rows(s, grip=-1.0), s))      # release -> boundary at decision 36
    seq += [(sat_rows(s), s)] * 3
    routes = []
    for c, st in seq:
        op.transform(c, {"states": st}, {})
        routes.append(op.diagnostics().get("envelope", {}).get("route"))
    assert op.detector.boundary == 36
    assert routes[:30] == ["saturation"] * 30 and routes[30:36] == [None] * 6 and routes[36:] == ["boundary"] * 4
    assert op.stats["b_active_by_route"] == {"boundary": 4, "saturation": 30, "both": 0}
    # fresh op = per-episode reset
    assert sat_op().sat_first is None


def test_s6_route_both():
    s = state(60.0, 90.0)
    s_hold = s.copy()
    op = sat_op()
    for _ in range(4):
        op.transform(sat_rows(s_hold, grip=1.0), {"states": s_hold}, {})   # unpinned close holds; tilt 60
    op.transform(sat_rows(s, grip=-1.0, pin_row=0), {"states": s}, {})     # boundary AND arming at decision 4
    assert op.detector.boundary == 4 and op.sat_first == 4
    assert op.diagnostics()["envelope"]["route"] == "both"


# ------------------------------------------------------------------------------------------------------------ S8
def test_s8_yaw_sat_control():
    s = state(60.0, 90.0)
    s_hold = s.copy()
    seq = [(sat_rows(s, pin_row=0, mag=0.6), s)] * 3 + [(sat_rows(s_hold, grip=1.0, mag=0.6), s_hold)] * 4 \
        + [(sat_rows(s, grip=-1.0, mag=0.6), s)] * 3
    on, yaw = sat_op("on"), sat_op("yawctrl")
    n_sat = n_bnd = 0
    for t, (c, st) in enumerate(seq):
        a = on.transform(c, {"states": st}, {})
        b = yaw.transform(c, {"states": st}, {})
        route = on.diagnostics().get("envelope", {}).get("route")
        assert route == yaw.diagnostics().get("envelope", {}).get("route")
        if route == "saturation":
            n_sat += 1
            base = np.clip(c[:5, 3:6], -1, 1)
            da, db = a[:5, 3:6] - base, b[:5, 3:6] - base
            assert np.allclose(db[:, :2], 0.0, atol=0)
            assert np.allclose(np.linalg.norm(db, axis=1), np.linalg.norm(da, axis=1), atol=1e-9)
            assert np.all(np.sign(db[:, 2])[np.abs(db[:, 2]) > 0] == A.yaw_coin(20261101, t))
            assert np.array_equal(b[:, :3], c[:, :3]) and np.array_equal(b[:, 6], c[:, 6]) and np.array_equal(b[5:], c[5:])
        elif route in ("boundary", "both"):
            n_bnd += 1
            assert np.array_equal(a, b)
        else:
            assert np.array_equal(a, b)
    assert n_sat >= 3 and n_bnd >= 1


# ------------------------------------------------------------------------------------------------------------ S9
def test_s9_derived_null_native_with_shadow_diagnostics():
    from harnessstudio.components.decision_loop import DecisionLoop
    from harnessstudio.core.arms import Arm
    from test_seam import FakeCtx
    s = state(60.0, 90.0)
    chunks = [sat_rows(s, pin_row=0)] * 6
    states = [s] * 6
    for entry in ("tilt-guard-v12-bsatz", "tilt-guard-v12-bsat-yaw", "tilt-guard-v12-bz"):
        out = {}
        for arm in (Arm.NULL, Arm.CANDIDATE):
            loop = DecisionLoop(registry.load(entry))
            ctx = FakeCtx(arm, [np.array(c) for c in chunks], states, horizon=30)

            async def go():
                while not ctx.ended:
                    await loop.decide(ctx)
            asyncio.run(go())
            diags = [f["diagnostics"]["action_op"] for e, f in ctx.logs if e == "decision"]
            assert all("sat_only_would_act" in d and "sat_armed_now" in d for d in diags)
            assert diags[0]["sat_first"] == 0
            out[arm] = (np.array(ctx.emitted), loop.action_op)
        native = np.concatenate([c[:5] for c in chunks])
        assert np.array_equal(out[Arm.NULL][0], native)                    # null executes native bitwise
        op_c = out[Arm.CANDIDATE][1]
        if op_c.sat_arming != "off":
            assert not np.array_equal(out[Arm.CANDIDATE][0], native)
        else:
            assert np.array_equal(out[Arm.CANDIDATE][0], native)          # flag off: no boundary -> native
            assert op_c.stats["n_sat_only_would_act"] == 6                 # shadow diagnostic still logged


# ----------------------------------------------------------------------------------------------------------- S10
def test_s10_pinned_edges_from_pinned_asset():
    op = sat_op()
    assert np.array_equal(op.aq01[3:6], np.asarray(NS["actions"]["q01"])[3:6])
    assert np.array_equal(op.aq99[3:6], np.asarray(NS["actions"]["q99"])[3:6])
    with pytest.raises(RuntimeError):
        make(envelope="bte", saturation_arming="on", norm_stats_sha256="0" * 64)
    for v in ("bsatz", "bsat", "bsat-yaw"):
        p = registry.load(f"tilt-guard-v12-{v}").card["params"]
        assert p["pin_margin"] == 0.02
        assert p["norm_stats_sha256"] == "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"
    with pytest.raises(ValueError):
        make(envelope="off", saturation_arming="on")
    with pytest.raises(ValueError):
        make(envelope="yawctrl", saturation_arming="on")


# ------------------------------------------------------------------------------------------------- S3 (random part)
def _random_episode(rng, n):
    seq = []
    for j in range(n):
        tilt = float(rng.choice([3.0, 20.0, 40.0, 60.0, 90.0, 140.0, 178.0])) + float(rng.uniform(-2, 2))
        st = state(tilt, float(rng.uniform(0, 360)), width=float(rng.choice([0.0005, 0.03])))
        r = rng.random()
        rows = int(rng.choice([10, 10, 10, 5, 3]))
        if r < 0.1:
            c = zero_chunk(st, rows=rows)
        else:
            c = rng.uniform(-1.6, 1.6, size=(rows, 7))
            c[:, 6] = float(rng.choice([-1.0, 1.0, 1.0]))
            if rng.random() < 0.3:          # saturate rx (G) / pin rotation dims
                c[:, 3] = st[3] + float(rng.uniform(-3.4, -3.25))
            if rng.random() < 0.2:
                c[int(rng.integers(0, rows)), int(rng.integers(3, 6))] = float(rng.uniform(-6, 6))
        if rng.random() < 0.03:
            c = c.copy()
            c[0, 0] = np.nan
        q = rng.random()
        if q < 0.04:
            obs_state = None
        elif q < 0.08:
            obs_state = st[:7]
        elif q < 0.1:
            obs_state = st.copy()
            obs_state[4] = np.nan
        else:
            obs_state = st
        seq.append((c, obs_state))
    return seq

