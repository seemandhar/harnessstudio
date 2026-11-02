"""Synthetic precedence checks for the U5+Z glue (A6.Z.5 tier-1 cases). Each check takes a factory returning a FRESH
composite op (real or mutant) and raises AssertionError on a violation. The reference outputs come from standalone
frozen components (registry entries 'tube-steer', 'tilt-guard-v12-z', 'tilt-guard-v11-zsign') fed the same inputs.
Synthetic memory: test-only injection of a one-memory tube (tilt 3 deg) for a synthetic instruction into the U5
instance (and the standalone T instance); the frozen package memory is untouched."""
from __future__ import annotations

import numpy as np

from harnessstudio.core import registry
from hs_tube_steer import action_op as TS

INSTR = "synthetic u5z instruction"
T8 = "put both moka pots on the stove"          # real public instruction without success memory (T abstains)
LOW, HIGH = 3.0, 40.0                            # tilt (deg): D ~ 0 vs D ~ 37 > eps 15 (tube at 3 deg)
_MEAN = None


def log_so3(M):
    c = float(np.clip((np.trace(M) - 1) / 2, -1, 1))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        w, v = np.linalg.eigh((M + M.T) / 2)
        return v[:, int(np.argmax(w))] * th
    return th / (2 * np.sin(th)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def _R(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def state(tilt_deg, x=0.0):
    R = _R("y", np.radians(tilt_deg)) @ _R("x", np.pi)
    return np.array([x, 0.0, 1.0, *log_so3(R), 0.02, -0.02])


def mean():
    global _MEAN
    if _MEAN is None:
        _MEAN = registry.load("tilt-guard-v12-z").slots["action_op"].build().action_mean.copy()
    return _MEAN


def zero_chunk(s, rows=10):
    r = mean().copy()
    r[:6] += s[:6]
    return np.tile(r, (rows, 1)).astype(np.float32)


def normal_chunk(rng, rows=10):
    c = rng.uniform(-0.6, 0.6, size=(rows, 7)).astype(np.float32)
    c[:, 6] = 1.0
    return c


def tube():
    return [np.array([TS.zaxis(state(LOW)[3:6]) for _ in range(80)])]


def comp_with_tube(factory):
    op = factory()
    op.u5.memory = {INSTR: tube()}
    return op


def standalone(name):
    op = registry.load(name).slots["action_op"].build()
    if name == "tube-steer":
        op.memory = {INSTR: tube()}
    return op


def run(op, seq, instr=INSTR):
    """seq: list of (state, chunk). Returns list of (out array copy, diag)."""
    res = []
    for s, c in seq:
        out = op.transform(c, {"states": s, "task_description": instr}, {})
        res.append((np.array(out, copy=True), op.diagnostics()))
    return res


def eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


# ---------------------------------------------------------------------------------------------------------------------
def check_zero_at_arming(factory, zname="tilt-guard-v12-z"):
    """Zero chunk exactly at U5's arming decision (q6): U5's output executes, not Z's."""
    rng = np.random.default_rng(1)
    tilts = [LOW] * 5 + [HIGH, HIGH, HIGH]
    seq = []
    for q, tl in enumerate(tilts):
        s = state(tl, x=0.01 * q)
        seq.append((s, zero_chunk(s) if q == 6 else normal_chunk(rng)))
    comp = run(comp_with_tube(factory), seq)
    T = run(standalone("tube-steer"), seq)
    Z = run(standalone(zname), seq)
    assert T[6][1].get("steer") and T[6][1].get("zero_chunk"), "fixture: T must arm and steer a zero chunk at q6"
    assert Z[6][1].get("z_mode") in ("tail", "hold"), "fixture: Z would substitute at q6"
    for q in range(6):
        assert eq(comp[q][0], Z[q][0]), f"pre-arming q{q}: composite != Z"
    for q in range(6, 8):
        assert eq(comp[q][0], T[q][0]), f"q{q}: composite != U5"
    assert not eq(comp[6][0], Z[6][0]), "q6: composite executed Z's substitution at U5's arming decision"
    assert comp[6][1]["glue"]["executed"] == "u5" and comp[6][1]["glue"]["arm_decision"] == 6
    assert comp[6][1]["z"] is None and comp[6][1]["glue"]["native_zero"] is True


def check_zero_before_arming(factory, zname="tilt-guard-v12-z"):
    """Zero chunk one decision before U5 arms (q5; arms at q6): Z's substitution executes at q5, U5 from q6."""
    rng = np.random.default_rng(2)
    tilts = [LOW] * 5 + [HIGH, HIGH, HIGH]
    seq = []
    for q, tl in enumerate(tilts):
        s = state(tl, x=0.01 * q)
        seq.append((s, zero_chunk(s) if q == 5 else normal_chunk(rng)))
    comp = run(comp_with_tube(factory), seq)
    T = run(standalone("tube-steer"), seq)
    Z = run(standalone(zname), seq)
    assert Z[5][1].get("z_mode") == "tail", "fixture: Z tail at q5"
    for q in range(6):
        assert eq(comp[q][0], Z[q][0]), f"pre-arming q{q}: composite != Z"
    assert not eq(comp[5][0], seq[5][1]), "q5: native zero chunk executed instead of Z's substitution"
    assert comp[5][1]["glue"]["executed"] == "z" and comp[5][1]["glue"]["z_substituted"]
    for q in (6, 7):
        assert eq(comp[q][0], T[q][0]), f"q{q}: composite != U5"
        assert comp[q][1]["glue"]["executed"] == "u5"


def check_cap(factory, zname="tilt-guard-v12-z"):
    """Run of 3 zero chunks before arming: tail, hold, native passthrough (cap 2) = standalone Z bitwise."""
    rng = np.random.default_rng(3)
    seq = []
    for q in range(8):
        s = state(LOW, x=0.01 * q)
        seq.append((s, zero_chunk(s) if q in (3, 4, 5) else normal_chunk(rng)))
    comp = run(comp_with_tube(factory), seq)
    Z = run(standalone(zname), seq)
    for q in range(8):
        assert eq(comp[q][0], Z[q][0]), f"q{q}: composite != Z"
    assert eq(comp[5][0], seq[5][1]), "run 3 must pass the native zero chunk through"
    modes = [(comp[q][1]["z"] or {}).get("z_mode") for q in (3, 4, 5)]
    assert modes == ["tail", "hold", "passthrough"], modes
    assert comp[7][1]["glue"]["stats"]["z_executed_substitutions"] == 2


def check_abstain_t8(factory, zname="tilt-guard-v12-z", n=60):
    """U5 abstains (no tube for the instruction, real memory): composite == Z for the whole episode."""
    rng = np.random.default_rng(4)
    seq = []
    for q in range(n):
        s = state(HIGH if q % 3 else LOW, x=0.005 * q)
        seq.append((s, zero_chunk(s) if q in (2, 10, 11, 20, 21, 22, 40) else normal_chunk(rng)))
    comp = run(factory(), seq, instr=T8)
    Z = run(standalone(zname), seq, instr=T8)
    for q in range(n):
        assert eq(comp[q][0], Z[q][0]), f"q{q}: composite != Z on an abstaining U5"
        assert comp[q][1]["u5"].get("skip") == "abstain" and comp[q][1]["glue"]["executed"] == "z"
    assert comp[-1][1]["glue"]["stats"]["z_executed_substitutions"] == 6


def check_disarm(factory, zname="tilt-guard-v12-z"):
    """After U5 arms and later disarms, a zero chunk is U5's (native passthrough); Z never executes again."""
    rng = np.random.default_rng(5)
    tilts = [LOW] * 5 + [HIGH, HIGH] + [LOW] * 6
    seq = []
    for q, tl in enumerate(tilts):
        s = state(tl, x=0.01 * q)
        seq.append((s, zero_chunk(s) if q in (9, 11) else normal_chunk(rng)))
    comp = run(comp_with_tube(factory), seq)
    T = run(standalone("tube-steer"), seq)
    assert T[6][1].get("steer") and not T[8][1].get("armed"), "fixture: T arms at q6 and disarms by q8"
    for q in range(6, len(seq)):
        assert eq(comp[q][0], T[q][0]), f"q{q}: composite != U5 after arming"
    assert eq(comp[9][0], seq[9][1]), "q9: post-disarm zero chunk must be U5's native passthrough"
    for q in range(6, len(seq)):
        assert comp[q][1]["z"] is None and comp[q][1]["glue"]["executed"] == "u5", f"q{q}: Z called after arming"


def check_order_and_routing(factory):
    """U5 is called first, both components get the very same native chunk object, inputs are never mutated."""
    rng = np.random.default_rng(6)
    op = comp_with_tube(factory)
    calls = []
    u5t, zt = op.u5.transform, op.z.transform

    def spy(name, fn):
        def wrapped(chunk, obs, memory):
            calls.append((name, id(chunk), np.array(chunk, copy=True)))
            return fn(chunk, obs, memory)
        return wrapped
    op.u5.transform, op.z.transform = spy("u5", u5t), spy("z", zt)
    tilts = [LOW] * 4 + [HIGH, HIGH, HIGH]
    for q, tl in enumerate(tilts):
        s = state(tl)
        c = zero_chunk(s) if q in (2, 5) else normal_chunk(rng)
        before = np.array(c, copy=True)
        calls.clear()
        op.transform(c, {"states": s, "task_description": INSTR}, {})
        assert calls and calls[0][0] == "u5", f"q{q}: U5 must be called first"
        assert all(i == id(c) for _, i, _ in calls), f"q{q}: a component did not receive the native chunk object"
        assert all(eq(x, before) for _, _, x in calls) and eq(c, before), f"q{q}: native chunk mutated"
        names = [n for n, _, _ in calls]
        assert names == (["u5", "z"] if q < 5 else ["u5"]), f"q{q}: calls {names}"


def long_episode(seed=7, n=70):
    rng = np.random.default_rng(seed)
    tilts = [LOW] * 8 + [HIGH] * 2 + [LOW] * 5 + [HIGH] * 10 + [LOW] * (n - 25)
    seq = []
    for q in range(n):
        s = state(tilts[q], x=0.002 * q)
        seq.append((s, zero_chunk(s) if q in (3, 4, 9, 10, 30, 31, 32, 50) else normal_chunk(rng)))
    return seq


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k != "wall_s"}
    return d


def check_determinism(factory, runs=3):
    seq = long_episode()
    res = [run(comp_with_tube(factory), seq) for _ in range(runs)]
    for r in res[1:]:
        for q in range(len(seq)):
            assert eq(r[q][0], res[0][q][0]), f"run differs at q{q}"
            assert strip(r[q][1]) == strip(res[0][q][1]), f"diagnostics differ at q{q}"


ALL_CHECKS = (check_zero_at_arming, check_zero_before_arming, check_cap, check_abstain_t8, check_disarm,
              check_order_and_routing, check_determinism)
