"""Unit tests for the frequency-domain correction (Eq. 10-11) and the FAST coefficient coordinates."""
from __future__ import annotations

import numpy as np
import pytest

from hs_rtcf import spectral
from hs_rtcf.action_op import resolve

SCALE = spectral.load_scale(resolve("assets/pi0_fast_libero_norm_stats.json"),
                            "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e")
MOTION = [0, 1, 2, 3, 4, 5]


def chunk(rng, h=10, a=7):
    return rng.uniform(-1, 1, (h, a)).astype(np.float32)


def test_dct_matrix_is_orthonormal_dct2():
    for h in (1, 2, 5, 10, 16):
        m = spectral.dct_matrix(h)
        assert np.allclose(m @ m.T, np.eye(h), atol=1e-12)
        x = np.random.default_rng(h).normal(size=h)
        # explicit DCT-II (ortho) formula
        n = np.arange(h)
        ref = np.array([np.sqrt((1 if k == 0 else 2) / h) * np.sum(x * np.cos(np.pi * (2 * n + 1) * k / (2 * h)))
                        for k in range(h)])
        assert np.allclose(m @ x, ref, atol=1e-12)


def test_dct_matches_scipy_when_available():
    scipy_fft = pytest.importorskip("scipy.fft")
    x = np.random.default_rng(0).normal(size=(10, 7))
    assert np.allclose(spectral.dct_matrix(10) @ x, scipy_fft.dct(x, axis=0, norm="ortho"), atol=1e-12)


def test_scale_from_norm_stats():
    assert SCALE.shape == (7,)
    # q01/q99 of pi0_fast_libero actions dim 0: -0.7262..., 1.0789...
    assert abs(SCALE[0] - 2.0 / (1.0789 - (-0.7262))) < 1e-3


def test_per_dim_offsets_only_touch_dc():
    rng = np.random.default_rng(1)
    a = chunk(rng).astype(np.float64)
    off = rng.normal(size=7)
    c1, c2 = spectral.coefficients(a, SCALE), spectral.coefficients(a + off[None, :], SCALE)
    assert np.allclose(c1[1:], c2[1:], atol=1e-12)
    assert not np.allclose(c1[0], c2[0])


def test_token_integrality_roundtrip():
    rng = np.random.default_rng(2)
    q = rng.integers(-40, 40, (10, 7))
    a = (spectral.dct_matrix(10).T @ (q / 10.0)) / SCALE[None, :] + rng.normal(size=7)[None, :]
    c = spectral.coefficients(a, SCALE)
    assert np.array_equal(spectral.quantize(c)[1:], q[1:])
    assert np.allclose(spectral.dequantize(spectral.quantize(c))[1:], c[1:], atol=1e-9)


Q01, Q99 = spectral.load_norm_bounds(resolve("assets/pi0_fast_libero_norm_stats.json"))


def test_exact_zero_chunk_signature():
    """Critic B4 / audit C3: openpi's zero chunk = (q01+q99)/2 on every row, + state[0:6] on the motion dims."""
    state = np.array([0.05, -0.1, 1.0, 3.1, 0.2, -0.4, 0.03, -0.03])
    zero = np.tile(spectral.openpi_zero_chunk_row(state, Q01, Q99), (10, 1)).astype(np.float32)
    assert spectral.is_openpi_zero_chunk(zero, state, Q01, Q99)
    assert not spectral.is_openpi_zero_chunk(zero, None, Q01, Q99)          # no state -> never flagged
    assert not spectral.is_openpi_zero_chunk(zero, state + 0.01, Q01, Q99)  # other state
    moved = zero.copy(); moved[3, 0] += 1e-3
    assert not spectral.is_openpi_zero_chunk(moved, state, Q01, Q99)
    # a genuinely stationary chunk (all f > 0 coefficients zero) is NOT a decode failure
    still = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32), (10, 1))
    assert np.all(np.abs(spectral.coefficients(still, SCALE)[1:]) < 1e-9)
    assert not spectral.is_openpi_zero_chunk(still, state, Q01, Q99)


def test_clipped_fractions_reported():
    rng = np.random.default_rng(12)
    a = chunk(rng)
    _, info = spectral.rtcf_correct(a, rng.normal(0, 5, (10, 7)), SCALE, MOTION, 3, 0.1, 0.1)
    assert 0.0 <= info["clipped_frac"] <= 1.0 and info["active"] == 12 and "limit_clipped_frac" in info
    assert info["clipped_frac"] > 0.5


@pytest.mark.parametrize("mode", ["spectral", "time_domain"])
def test_lambda_zero_is_bitwise_noop(mode):
    rng = np.random.default_rng(4)
    a = chunk(rng)
    out, info = spectral.rtcf_correct(a, rng.normal(size=(10, 7)), SCALE, MOTION, 3, 1.0, 0.0, 1.0, mode)
    assert out.dtype == a.dtype and out.tobytes() == a.tobytes() and info["changed"] == 0


@pytest.mark.parametrize("mode", ["spectral", "time_domain"])
def test_gripper_untouched_bitwise(mode):
    rng = np.random.default_rng(5)
    for _ in range(50):
        a = chunk(rng)
        out, _ = spectral.rtcf_correct(a, rng.normal(0, 3, (10, 7)), SCALE, MOTION, 3, 1.0, 0.1, 1.0, mode)
        assert out[:, 6].tobytes() == a[:, 6].tobytes()


def test_memory_equal_to_native_changes_nothing():
    a = chunk(np.random.default_rng(6))
    out, info = spectral.rtcf_correct(a, spectral.coefficients(a, SCALE), SCALE, MOTION, 3, 1.0, 0.1)
    assert out.tobytes() == a.tobytes() and info["changed"] == 0


def test_mask_clip_and_scale_in_coefficient_space():
    rng = np.random.default_rng(7)
    for _ in range(200):
        a = (0.3 * chunk(rng)).astype(np.float32)        # well inside the limit: no magnitude clipping
        mem = rng.normal(0, 2, (10, 7))
        f_cut, dmax, lam = int(rng.integers(0, 11)), float(rng.uniform(0.05, 3)), float(rng.uniform(0, 0.2))
        out, info = spectral.rtcf_correct(a, mem, SCALE, MOTION, f_cut, dmax, lam, action_limit=None)
        c_pi = spectral.coefficients(a, SCALE)
        dc = spectral.coefficients(out.astype(np.float64), SCALE) - c_pi
        expect = np.zeros((10, 7))
        if f_cut > 1:
            expect[1:f_cut, :6] = lam * np.clip(mem - c_pi, -dmax, dmax)[1:f_cut, :6]
        assert np.allclose(dc, expect, atol=1e-5)        # float32 output rounding
        # L2 bound per motion channel in action units
        for d in range(6):
            bound = lam * dmax * np.sqrt(max(0, min(f_cut, 10) - 1)) / SCALE[d]
            assert np.linalg.norm(out[:, d].astype(np.float64) - a[:, d]) <= bound + 1e-5


def test_f_cut_at_most_one_changes_nothing():
    rng = np.random.default_rng(8)
    a = chunk(rng)
    for f_cut in (0, 1):
        out, info = spectral.rtcf_correct(a, rng.normal(size=(10, 7)), SCALE, MOTION, f_cut, 1.0, 0.1)
        assert out.tobytes() == a.tobytes()


def test_action_limit_only_on_changed_entries():
    rng = np.random.default_rng(9)
    a = chunk(rng)
    a[0, 0] = 1.7            # native out of range on a motion dim
    a[3, 6] = 2.0            # and on the gripper
    out, _ = spectral.rtcf_correct(a, np.zeros((10, 7)), SCALE, [1], 3, 1.0, 0.1, 1.0)   # only dim 1 corrected
    assert out[0, 0] == np.float32(1.7) and out[3, 6] == np.float32(2.0)
    assert np.all(np.abs(out[:, 1]) <= 1.0)


def test_time_domain_residual():
    rng = np.random.default_rng(10)
    a = (0.3 * chunk(rng)).astype(np.float32)
    mem = rng.normal(size=(10, 7))
    out, _ = spectral.rtcf_correct(a, mem, SCALE, MOTION, 3, 0.5, 0.1, None, "time_domain")
    x_mem = spectral.dct_matrix(10).T @ mem
    expect = a.astype(np.float64).copy()
    expect[:, :6] += (0.1 * np.clip(x_mem - a * SCALE, -0.5, 0.5))[:, :6] / SCALE[:6]
    assert np.allclose(out, expect, atol=1e-6)


def test_bad_inputs_raise():
    a = chunk(np.random.default_rng(11))
    with pytest.raises(ValueError):
        spectral.rtcf_correct(a, np.zeros((9, 7)), SCALE, MOTION, 3, 1.0, 0.1)
    with pytest.raises(ValueError):
        spectral.rtcf_correct(a, np.zeros((10, 7)), SCALE, [7], 3, 1.0, 0.1)
    with pytest.raises(ValueError):
        spectral.rtcf_correct(a, np.zeros((10, 7)), SCALE, MOTION, 3, 1.0, 0.1, mode="bogus")
