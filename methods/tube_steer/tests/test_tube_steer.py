"""tube-steer unit tests (CPU): memory pinning (C1), abstention/lookup (C2), gate convention (C3), restoring sign,
steering law and passthrough bit-identity, zero-chunk rows, hysteresis/hold limit/re-arm window, ortho control (C6),
diagnostics (C4) and the DecisionLoop seam (derived null native)."""
from __future__ import annotations

import asyncio
import glob
import json
import os

import numpy as np
import pytest

from harnessstudio.core import registry
from hs_tube_steer import action_op as A

RUN = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtcfpp-20261101"
INSTR = "synthetic instruction"


def log_so3(M):
    c = float(np.clip((np.trace(M) - 1) / 2, -1, 1))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        w, v = np.linalg.eigh((M + M.T) / 2)
        return v[:, int(np.argmax(w))] * th
    return th / (2 * np.sin(th)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def state(tilt_deg, az_deg=0.0):
    R = Rz(np.radians(az_deg)) @ Ry(np.radians(tilt_deg)) @ Rx(np.pi)
    return np.array([0.0, 0.0, 1.0, *log_so3(R), 0.02, -0.02])


def op_with_memory(mode="steer", tube_tilt=3.0, n=60):
    op = registry.load("tube-steer-ortho" if mode == "ortho" else "tube-steer").slots["action_op"].build()
    zs = np.array([A.zaxis(state(tube_tilt)[3:6]) for _ in range(n)])
    op.memory = {INSTR: [zs]}
    return op


def chunk(rng, rows=10):
    c = rng.uniform(-0.6, 0.6, size=(rows, 7))
    c[:, 6] = 1.0
    return c


def feed(op, tilts, rng, az=0.0, instr=INSTR, chunks=None):
    outs = []
    for i, tl in enumerate(tilts):
        c = chunks[i] if chunks is not None else chunk(rng)
        outs.append((c, op.transform(c, {"states": state(tl, az), "task_description": instr}, {}), op.diagnostics()))
    return outs


def test_memory_pinned_and_refuses_mismatch():
    m = registry.load("tube-steer").card["params"]
    assert m["memory_npz_sha256"] == "fcf8338306294cace8979ede718d2ede7d8b287198144fbf07e21604a5677f0a"
    assert m["memory_json_sha256"] == "00d2e87bc024ffed09fc04953e389f1bbded1f26c3ba3a4fe2be25d39fb2d180"
    A.TubeMemory._cache.clear()
    with pytest.raises(RuntimeError):
        A.TubeSteerActionOp(**{**m, "memory_npz_sha256": "0" * 64})
    op = A.TubeSteerActionOp(**m)
    assert sorted(len(v) for v in op.memory.values()) == sorted([19, 20, 8, 7, 17, 15, 15, 19, 7])


def test_lookup_benchmark_task_names_9_hits_t8_abstains():
    op = registry.load("tube-steer").slots["action_op"].build()
    names = set()
    for p in glob.glob(os.path.join(RUN, "eval/dev-baseline/shards/identity-t*-i30-31/baseline/traces/*.json")):
        names.add(json.load(open(p))["task"])          # written by the benchmark (vla-eval LIBERO task registry)
    instr = {n.rsplit("#", 1)[1]: n.split("/", 1)[1].rsplit("#", 1)[0] for n in names}
    assert len(instr) == 10
    hits = {tid for tid, i in instr.items() if i in op.memory}
    assert hits == {str(t) for t in range(10)} - {"8"}
    rng = np.random.default_rng(0)
    c = chunk(rng)
    for tilt in (3.0, 60.0, 60.0, 90.0):
        out = op.transform(c, {"states": state(tilt), "task_description": instr["8"]}, {})
        assert out is c and op.diagnostics()["tube_found"] is False


def test_gate_two_consecutive_and_arm_window():
    rng = np.random.default_rng(1)
    op = op_with_memory()
    res = feed(op, [3, 20, 3, 20, 20, 20], rng)                # single excursion does not arm; 2 consecutive do
    armed = [d["armed"] for _, _, d in res]
    assert armed == [False, False, False, False, True, True]
    assert all(o is c for c, o, d in res[:4])
    op = op_with_memory()
    res = feed(op, [3] * 39 + [20, 20, 20], rng)               # second > eps at q=40 > 39: never arms
    assert op.stats["first_arm"] is None and all(o is c for c, o, _ in res)
    op = op_with_memory()
    feed(op, [3] * 38 + [20, 20], rng)                         # q = 39 arms
    assert op.stats["first_arm"] == 39
    op = op_with_memory()
    feed(op, [3, 15.0, 15.0], rng)                             # D == eps (approximately) must not arm when <= 15
    assert op.stats["first_arm"] is None or op.diagnostics()["D"] > 15


def test_restoring_sign_and_steering_law():
    rng = np.random.default_rng(2)
    for az in (0.0, 70.0, 200.0):
        op = op_with_memory()
        res = feed(op, [40, 40], rng, az=az)
        c, out, d = res[1]
        assert d["steer"] and out is not c
        z, zref, u = np.array(d["z"]), np.array(d["z_ref"]), np.array(d["u"])
        assert np.allclose(u, np.cross(z, zref) / np.linalg.norm(np.cross(z, zref)))
        R = A.rotmat(state(40, az)[3:6])
        for g in (0.136, 1.0):                                  # one OSC-style world-frame step about u reduces D
            Rn = A.rotmat(g * 0.5 * 0.5 * u) @ R
            assert np.degrees(np.arccos(np.clip(Rn[:, 2] @ zref, -1, 1))) < d["D"]
        gain = float(np.clip(d["D"] / 15, 0.5, 1.0))
        exp = c[:5, 3:6] - (c[:5, 3:6] @ u)[:, None] * u[None, :] + gain * u[None, :]
        assert np.array_equal(out[:5, 3:6], np.clip(exp, -1, 1))
        assert np.array_equal(out[:, :3], c[:, :3]) and np.array_equal(out[:, 6], c[:, 6]) and np.array_equal(out[5:], c[5:])
        assert abs(d["executed_component"] - gain) < 1e-6 or np.any(np.abs(exp) > 1)


def test_gain_clip():
    op = op_with_memory()
    assert op.gain(3.0) == 0.5 and op.gain(15.0) == 1.0 and op.gain(30.0) == 1.0 and op.gain(11.25) == 0.75


def test_release_hysteresis_rearm_and_hold_limit():
    rng = np.random.default_rng(3)
    op = op_with_memory()
    res = feed(op, [20, 20, 12, 9, 7, 20, 20], rng)            # tube at 3 deg: D = tilt - 3 -> 17,17,9,6,4,17,17
    steer = [bool(d.get("steer")) for _, _, d in res]
    assert steer == [False, True, True, False, False, False, True]   # disarm at D 6 <= 7.5; re-arm needs 2 x > 15
    assert op.stats["arm_events"] == 2
    op = op_with_memory()
    res = feed(op, [3] * 30 + [20] * 40, rng)                  # armed at q=31; hold limit step 300 -> no steer after q=60
    steer = [bool(d.get("steer")) for _, _, d in res]
    assert steer[31] and steer[60] and not any(steer[61:])
    op = op_with_memory()
    res = feed(op, [3] * 30 + [20, 20, 5] + [20] * 10, rng)    # disarm at q=32; re-arm needs q <= 39: re-arms at 34
    assert op.stats["arm_events"] == 2
    op = op_with_memory()
    res = feed(op, [3] * 36 + [20, 20, 5] + [20] * 10, rng)    # disarm at q=38; next arm attempt q=40 > 39: no re-arm
    assert op.stats["arm_events"] == 1


def test_zero_chunk_rows_while_armed():
    rng = np.random.default_rng(4)
    op = op_with_memory()
    feed(op, [40], rng)
    s = state(40)
    z = np.tile(op.action_mean.copy(), (10, 1))
    z[:, :6] += s[:6]
    out = op.transform(z, {"states": s, "task_description": INSTR}, {})
    d = op.diagnostics()
    assert d["zero_chunk"] and d["steer"]
    u = np.array(d["u"])
    assert np.allclose(out[:5, 3:6], np.clip(d["c"] * u, -1, 1)[None, :])
    assert np.array_equal(out[:, :3], z[:, :3]) and np.array_equal(out[:, 6], z[:, 6]) and np.array_equal(out[5:], z[5:])


def test_ortho_same_arming_equal_norm_orthogonal_and_capped():
    rng = np.random.default_rng(5)
    tilts = [3, 30, 30] + [30] * 20
    chunks = [chunk(rng) for _ in tilts]
    T, O = op_with_memory("steer"), op_with_memory("ortho")
    rt = feed(T, tilts, rng, chunks=chunks)
    ro = feed(O, tilts, rng, chunks=chunks)
    assert T.stats["first_arm"] == O.stats["first_arm"] == 2
    n_o = 0
    for (c, ot, dt), (_, oo, do) in zip(rt, ro):
        if do.get("steer"):
            n_o += 1
            u, up = np.array(dt["u"]), np.array(do["axis"])
            assert abs(u @ up) < 1e-12 and abs(np.linalg.norm(up) - 1) < 1e-12
            assert do["ortho_sign"] == A.ortho_coin(do["q"]) and abs(do["c"] - dt["c"]) < 1e-12
            raw = c[:5, 3:6]
            exp = raw - (raw @ up)[:, None] * up[None, :] + do["ortho_sign"] * do["c"] * up[None, :]
            assert np.array_equal(oo[:5, 3:6], np.clip(exp, -1, 1))
            assert np.allclose(np.clip(oo[:5, 3:6], -1, 1) @ u, np.clip(exp, -1, 1) @ u)   # native component along u kept
        else:
            assert oo is c
    assert n_o == 6 and O.stats["arm_events"] == 1                     # cap 6, never re-arms
    O2 = op_with_memory("ortho")
    feed(O2, [3, 30, 30, 30, 5, 30, 30, 30], rng)                       # release ends the ortho run; no re-arm
    assert O2.stats["steered_decisions"] == 2 and O2.stats["arm_events"] == 1   # steer q2, q3; q4 D 2 <= 7.5 ends it


def test_passthrough_bitwise_random_and_invalid_inputs():
    rng = np.random.default_rng(6)
    op = op_with_memory()
    for _ in range(3000):
        c = chunk(rng)
        assert op.transform(c, {"states": state(float(rng.uniform(0, 12)), float(rng.uniform(0, 360))),
                                "task_description": INSTR}, {}) is c
    op = op_with_memory()
    feed(op, [40, 40], rng)
    c = chunk(rng)
    for obs in ({"task_description": INSTR}, {"states": state(40)[:5], "task_description": INSTR},
                {"states": [np.nan] * 8, "task_description": INSTR}, {"states": state(40)}):
        assert op.transform(c, obs, {}) is c
    bad = chunk(rng)[:, :6]
    assert op.transform(bad, {"states": state(40), "task_description": INSTR}, {}) is bad


def test_entries_cards():
    for e, mode in (("tube-steer", "steer"), ("tube-steer-ortho", "ortho")):
        cfg = registry.load(e)
        p = cfg.card["params"]
        assert p["mode"] == mode and cfg.null == "derived" and cfg.budgets.llm_calls == 0 and cfg.budgets.policy_calls is None
        assert (p["eps_deg"], p["consecutive"], p["arm_by_decision"], p["window_decisions"], p["release_deg"],
                p["hold_limit_step"], p["gain_floor"], p["gain_cap"], p["ortho_cap"]) == (15.0, 2, 39, 4, 7.5, 300, 0.5, 1.0, 6)
        assert cfg.fidelity.value == "tf_variant" and cfg.card["development_only"] == (mode == "ortho")


def test_seam_decision_loop_null_native_candidate_steers():
    from harnessstudio.components.decision_loop import DecisionLoop
    from harnessstudio.context.policy import PolicyProfile, Proposal
    from harnessstudio.core.arms import Arm
    from collections import deque

    rng = np.random.default_rng(7)
    instr = "turn on the stove and put the moka pot on it"
    tilts = [3, 50, 50, 50, 50, 50]
    chunks = [chunk(rng) for _ in tilts]

    class Ctx:
        def __init__(self, arm):
            self._arm = arm
            self._profile = PolicyProfile(capabilities=frozenset({"act", "chunk", "seeded_sampling"}), chunking="harness", execute_horizon=5)
            self._native_queue = deque()
            self.accounting = type("Acc", (), {"decisions": 0})()
            self.memory, self.instruction, self.emitted, self.logs = {}, instr, [], []
            q = deque(chunks)

            class P:
                async def sample(self, k, obs=None, instruction=None, sampler=None, readouts=()):
                    return [Proposal.make(q.popleft())]
            self.policy = P()

        @property
        def ended(self):
            return len(self.emitted) >= 5 * len(chunks)

        def observe(self):
            return {"states": state(tilts[min(len(self.emitted) // 5, len(tilts) - 1)]), "task_description": instr}

        async def _emit(self, a):
            self.emitted.append(np.array(a))

        def _log(self, e, **f):
            self.logs.append((e, f))
    out = {}
    for arm in (Arm.NULL, Arm.CANDIDATE):
        loop = DecisionLoop(registry.load("tube-steer"))
        ctx = Ctx(arm)

        async def go():
            while not ctx.ended:
                await loop.decide(ctx)
        asyncio.run(go())
        out[arm] = np.array(ctx.emitted)
        diags = [f["diagnostics"]["action_op"] for e, f in ctx.logs if e == "decision"]
        assert all(d["tube_found"] for d in diags)
    native = np.concatenate([c[:5] for c in chunks])
    assert np.array_equal(out[Arm.NULL], native)
    assert np.array_equal(out[Arm.CANDIDATE][:10], native[:10]) and not np.array_equal(out[Arm.CANDIDATE], native)


def test_window_is_four_decisions_and_short_memory_clamp():
    zs3 = A.zaxis(state(3.0)[3:6])
    zs40 = A.zaxis(state(40.0)[3:6])
    mem = np.array([zs3] * 15 + [zs40] + [zs3] * 10)
    z = A.zaxis(state(40.0)[3:6])
    D10, _ = A.tube_deviation(z, [mem], 10, 4)     # window [6, 14] excludes row 15
    D11, _ = A.tube_deviation(z, [mem], 11, 4)     # window [7, 15] includes row 15
    assert abs(D10 - 37.0) < 1e-6 and D11 < 1e-4          # arccos near 1: ~1e-6 deg float floor
    short = np.array([zs3] * 5 + [zs40])           # memory shorter than q - 4: its last row counts
    Ds, ref = A.tube_deviation(z, [short], 30, 4)
    assert Ds < 1e-4 and np.allclose(ref, zs40)
