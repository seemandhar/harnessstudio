"""Unit tests for band-guard (r1g-p1) and its sign control on synthetic chunks/states (critic C2, C3, C6, C10)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import hs_band_guard
from hs_band_guard.action_op import BandGuardActionOp, is_openpi_zero_chunk, public_coin, resolve

NS = json.loads(resolve("assets/pi0_fast_libero_norm_stats.json").read_text())
NS = NS.get("norm_stats", NS)
SQ01, SQ99 = NS["state"]["q01"][3], NS["state"]["q99"][3]
AQ01, AQ99 = NS["actions"]["q01"][3], NS["actions"]["q99"][3]
AMEAN = np.asarray(NS["actions"]["mean"])
SHA = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"


def op(correction="restore", **kw):
    params = dict(hs_band_guard.config().slots["action_op"].params)
    params.update(correction=correction, **kw)
    return BandGuardActionOp(**params)


def state(rx=3.0, rng=None):
    s = np.array([-0.1, 0.0, 1.0, rx, -0.05, -0.08, 0.03, -0.03])
    if rng is not None:
        s[:3] += rng.normal(scale=0.05, size=3)
    return s


def chunk_with_m(s, m_rx, rng, h=10, dtype=np.float32):
    """Absolute openpi chunk: dims 0-5 = m + s, gripper absolute; rx rows have m_rx (scalar or [h])."""
    c = rng.uniform(-0.3, 0.3, (h, 7))
    c[:, :6] += s[:6]
    c[:, 3] = np.asarray(m_rx, dtype=np.float64) + s[3]
    c[:, 6] = rng.choice([-1.0, 1.0])
    return c.astype(dtype)


# -- card / config ------------------------------------------------------------------------------------------
def test_cards_budgets_and_params_frozen():
    for cfg, corr in ((hs_band_guard.config(), "restore"), (hs_band_guard.config_signctrl(), "signctrl")):
        p = cfg.slots["action_op"].params
        assert cfg.fidelity.value == "faithful" and cfg.substitutions == ()
        assert cfg.null == "derived" and cfg.trainable_parameters == 0
        assert vars(cfg.budgets) == {"policy_calls": None, "llm_calls": 0, "tool_calls": 0, "max_tokens": 0}
        assert set(cfg.slots) == {"action_op"}
        assert p == {"dim": 3, "state_q01": SQ01, "state_q99": SQ99, "action_q01": AQ01, "action_q99": AQ99,
                     "margin": 0.02, "gate_rows": 5, "zero_chunk_tol": 1e-4,
                     "norm_stats": "assets/pi0_fast_libero_norm_stats.json", "norm_stats_sha256": SHA,
                     "state_key": "states", "correction": corr}
        assert cfg.card["disclosures"]
    assert hashlib.sha256(resolve("assets/pi0_fast_libero_norm_stats.json").read_bytes()).hexdigest() == SHA
    assert (SQ01, SQ99, AQ01, AQ99) == (1.5003631114959717, 3.2705323696136475, -3.306384325027466, -1.4915390014648438)


def test_edges_must_equal_pinned_norm_stats():
    with pytest.raises(ValueError):
        op(state_q99=3.3)
    with pytest.raises(ValueError):
        op(action_q01=-3.3)
    with pytest.raises(RuntimeError):
        op(norm_stats_sha256="0" * 64)
    with pytest.raises(ValueError):
        op(correction="other")


def test_constructor_is_side_effect_free_and_has_no_rng(monkeypatch):
    calls = []
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: calls.append(a))
    o = op()
    assert o.stats["decisions"] == 0 and o.diagnostics() == {}
    rng = np.random.Generator(np.random.PCG64(0))
    s = state(4.0)
    o.transform(chunk_with_m(s, -3.37, rng), {"states": s}, {})
    assert calls == []


# -- exact formula and gate ---------------------------------------------------------------------------------
def test_above_band_saturated_exact_formula_every_row():
    rng = np.random.Generator(np.random.PCG64(1))
    s = state(4.0)
    c = chunk_with_m(s, -3.37, rng, dtype=np.float64)
    out = op().transform(c, {"states": s}, {})
    assert out is not c
    expected = (c[:, 3] - 4.0) + SQ99                 # a' = m + clip(s)
    assert np.array_equal(out[:, 3], expected)
    assert np.allclose(out[:, 3], -3.37 + SQ99, atol=1e-12)
    assert abs(out[0, 3] - (-0.0995)) < 1e-3          # proposal unit example: s=4.0, m=-3.37 -> -0.0995
    others = [0, 1, 2, 4, 5, 6]
    assert np.array_equal(out[:, others], c[:, others])  # other dims bitwise untouched
    assert np.array_equal(c[:, 3], (np.full(10, -3.37) + 4.0))  # input not mutated


def test_below_band_saturated_lower_side():
    rng = np.random.Generator(np.random.PCG64(2))
    s = state(1.2)
    c = chunk_with_m(s, -1.48, rng, dtype=np.float64)   # m at/above action q99 - margin
    o = op()
    out = o.transform(c, {"states": s}, {})
    assert np.array_equal(out[:, 3], (c[:, 3] - 1.2) + SQ01)
    d = o.diagnostics()
    assert d["active"] and d["side"] == -1 and d["shift"] == pytest.approx(1.2 - SQ01)
    assert o.stats["active_below"] == 1


def test_float32_chunk_keeps_dtype():
    rng = np.random.Generator(np.random.PCG64(3))
    s = state(4.2)
    c = chunk_with_m(s, -3.40, rng)
    out = op().transform(c, {"states": s}, {})
    assert out.dtype == np.float32
    assert np.array_equal(out[:, 3], ((c[:, 3].astype(np.float64) - 4.2) + SQ99).astype(np.float32))


@pytest.mark.parametrize("rx,m,active", [
    (4.0, -3.20, False),            # above band but not saturated (m above floor + margin)
    (3.0, -3.37, False),            # saturated but in band
    (SQ99, -3.40, False),           # exactly at the band edge: not outside
    (1.6, -1.40, False),            # in band (lower side)
    (1.2, -2.00, False),            # below band, not saturated at the ceiling
    (SQ01, -1.40, False),
])
def test_gate_off_returns_input_object(rx, m, active):
    rng = np.random.Generator(np.random.PCG64(4))
    s = state(rx)
    c = chunk_with_m(s, m, rng)
    o = op()
    out = o.transform(c, {"states": s}, {})
    assert out is c and not o.diagnostics()["active"]


def test_saturation_margin_boundary():
    rng = np.random.Generator(np.random.PCG64(5))
    s = state(4.0)
    on = chunk_with_m(s, AQ01 + 0.02 - 1e-6, rng, dtype=np.float64)
    off = chunk_with_m(s, AQ01 + 0.02 + 1e-6, rng, dtype=np.float64)
    assert op().transform(on, {"states": s}, {}) is not on
    assert op().transform(off, {"states": s}, {}) is off
    s2 = state(1.0)
    on2 = chunk_with_m(s2, AQ99 - 0.02 + 1e-6, rng, dtype=np.float64)
    off2 = chunk_with_m(s2, AQ99 - 0.02 - 1e-6, rng, dtype=np.float64)
    assert op().transform(on2, {"states": s2}, {}) is not on2
    assert op().transform(off2, {"states": s2}, {}) is off2
    # one saturated executed row suffices (min/max over rows[:5])
    m = np.full(10, -3.0)
    m[2] = AQ01 + 0.019
    one = chunk_with_m(s, m, rng, dtype=np.float64)
    assert op().transform(one, {"states": s}, {}) is not one


def test_gate_uses_only_first_five_rows():
    rng = np.random.Generator(np.random.PCG64(6))
    s = state(4.0)
    m = np.full(10, -3.0)
    m[5:] = -3.40                      # saturated only in rows 5..9 (not executed natively)
    c = chunk_with_m(s, m, rng)
    assert op().transform(c, {"states": s}, {}) is c
    m2 = np.full(10, -3.0)
    m2[4] = -3.40                      # saturated in row 4 (executed)
    c2 = chunk_with_m(s, m2, rng, dtype=np.float64)
    out = op().transform(c2, {"states": s}, {})
    assert np.array_equal(out[:, 3], (c2[:, 3] - 4.0) + SQ99)   # applied to ALL rows incl. 5..9
    s3 = state(1.0)
    m3 = np.full(10, -2.0)
    m3[5:] = -1.40
    c3 = chunk_with_m(s3, m3, rng)
    assert op().transform(c3, {"states": s3}, {}) is c3


# -- no-op guards --------------------------------------------------------------------------------------------
def test_noop_without_states_short_states_bad_shape_nonfinite():
    rng = np.random.Generator(np.random.PCG64(7))
    s = state(4.0)
    c = chunk_with_m(s, -3.40, rng)
    for obs, reason in (({}, "no_state"), ({"state": s}, "no_state"), ({"states": s[:5]}, "state_short"),
                        (None, "no_state")):
        o = op()
        assert o.transform(c, obs, {}) is c and o.diagnostics()["skip"] == reason
    o = op()
    c6 = c[:, :6]
    assert o.transform(c6, {"states": s}, {}) is c6 and o.diagnostics()["skip"] == "chunk_shape"
    bad = c.copy()
    bad[2, 0] = np.nan
    o = op()
    assert o.transform(bad, {"states": s}, {}) is bad and o.diagnostics()["skip"] == "nonfinite"
    o = op()
    s_nan = s.copy()
    s_nan[1] = np.inf
    assert o.transform(c, {"states": s_nan}, {}) is c and o.diagnostics()["skip"] == "nonfinite"
    # a 7-d state (fake stack-like) above band still works on index 3 (>= 6 entries)
    o = op()
    assert o.transform(c, {"states": list(s[:7])}, {}) is not c


def test_zero_chunk_is_a_noop_even_out_of_band():
    for rx in (4.0, 1.0):
        s = state(rx)
        row = AMEAN.copy()
        row[:6] += s[:6]
        z = np.tile(row, (10, 1)).astype(np.float32)
        assert is_openpi_zero_chunk(z, s, AMEAN, 1e-4)
        o = op()
        assert o.transform(z, {"states": s}, {}) is z and o.diagnostics()["skip"] == "zero_chunk"
    # a stationary but non-zero chunk is not a zero chunk
    s = state(4.0)
    row = AMEAN.copy()
    row[:6] += s[:6]
    row[0] += 0.01
    assert not is_openpi_zero_chunk(np.tile(row, (10, 1)), s, AMEAN, 1e-4)
    # the actions mean rx lies strictly inside the saturation edges: a zero chunk could never pass the gate
    assert AQ01 + 0.02 < AMEAN[3] < AQ99 - 0.02


def test_property_inband_bitwise_identity():
    rng = np.random.Generator(np.random.PCG64(8))
    o = op()
    for _ in range(10000):
        rx = rng.uniform(SQ01, SQ99)
        s = state(rx, rng)
        c = chunk_with_m(s, rng.uniform(-4.0, -1.0, size=10), rng)
        assert o.transform(c, {"states": s}, {}) is c
    # out of band but never saturated on the matching side
    for _ in range(2000):
        if rng.random() < 0.5:
            s, m = state(rng.uniform(SQ99 + 1e-6, 6.0), rng), rng.uniform(AQ01 + 0.0201, -1.0, size=10)
        else:
            s, m = state(rng.uniform(0.5, SQ01 - 1e-6), rng), rng.uniform(-4.0, AQ99 - 0.0201, size=10)
        c = chunk_with_m(s, m, rng)
        assert o.transform(c, {"states": s}, {}) is c
    assert o.stats["active"] == 0


def test_deterministic_and_stateless_across_decisions():
    rng = np.random.Generator(np.random.PCG64(9))
    seq = []
    for _ in range(50):
        s = state(rng.uniform(1.0, 5.0), rng)
        seq.append((chunk_with_m(s, rng.uniform(-3.5, -1.3, size=10), rng), s))
    a, b = op(), op()
    outs_a = [np.array(a.transform(c, {"states": s}, {})) for c, s in seq]
    outs_b = [np.array(b.transform(c, {"states": s}, {})) for c, s in seq[::-1]][::-1]
    for x, y in zip(outs_a, outs_b):
        assert np.array_equal(x, y)   # restore depends only on (chunk, state): order-independent


# -- sign control (C3) ---------------------------------------------------------------------------------------
def test_signctrl_same_gate_same_magnitude_public_coin():
    rng = np.random.Generator(np.random.PCG64(10))
    o = op("signctrl")
    first = state(3.0)
    seed = hashlib.sha256(first.tobytes()).digest()
    assert o.transform(chunk_with_m(first, -2.0, rng), {"states": first}, {}) is not None   # decision 1 (inactive)
    coins = []
    for t in range(2, 200):
        s = state(rng.uniform(3.4, 5.0))
        c = chunk_with_m(s, -3.40, rng, dtype=np.float64)
        restore = op().transform(c, {"states": s}, {})
        out = o.transform(c, {"states": s}, {})
        d = o.diagnostics()
        assert d["active"] and d["coin"] == public_coin(seed, t)
        coins.append(d["coin"])
        if d["coin"] > 0:
            assert np.array_equal(out, restore)
        else:
            assert np.array_equal(out[:, 3], c[:, 3] + (s[3] - SQ99))
        assert np.allclose(np.abs(out[:, 3] - c[:, 3]), s[3] - SQ99, atol=1e-12)
        assert np.array_equal(out[:, [0, 1, 2, 4, 5, 6]], c[:, [0, 1, 2, 4, 5, 6]])
    assert 60 < sum(1 for x in coins if x > 0) < 140      # both signs occur
    assert o.stats["coin_plus"] + o.stats["coin_minus"] == o.stats["active"] == len(coins)


def test_signctrl_inactive_is_bitwise_and_coin_reproducible():
    rng = np.random.Generator(np.random.PCG64(11))
    s0 = state(3.0)
    seq = [(chunk_with_m(s0, -2.0, rng), s0)]
    for _ in range(30):
        s = state(rng.choice([3.0, 4.5]))
        seq.append((chunk_with_m(s, rng.choice([-2.0, -3.40]), rng), s))
    a, b = op("signctrl"), op("signctrl")
    for c, s in seq:
        x, y = a.transform(c, {"states": s}, {}), b.transform(c, {"states": s}, {})
        assert np.array_equal(x, y)
        if s[3] == 3.0:
            assert x is c


def test_public_coin_is_balanced_and_deterministic():
    seed = hashlib.sha256(b"x").digest()
    vals = [public_coin(seed, t) for t in range(4000)]
    assert vals == [public_coin(seed, t) for t in range(4000)]
    assert set(vals) == {-1, 1} and 1800 < sum(v > 0 for v in vals) < 2200
