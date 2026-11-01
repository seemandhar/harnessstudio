"""Correct in Frequency (RTCF v2 §3.2, Eq. 10-11): bounded low-frequency residual in FAST coefficient space.

Coordinates. pi0-FAST (openpi pi0_fast_libero) tokenizes round(10 * DCT_ortho_time(N(A_model))), where A_model is
the action minus a per-dimension constant (state[:6] on dims 0-5, openpi DeltaActions) and N is the per-dimension
normalization the checkpoint was TRAINED with. For pi0_fast_libero (checkpoint 2025-05-28) that is z-score,
x -> (x - mean) / (std + 1e-6) (openpi transforms.Normalize._normalize, use_quantiles=False); openpi >= a3d8ea3
(2025-07-05) serves it with quantile norm x -> 2 (x - q01) / (q99 - q01 + 1e-6) - 1 unless the server is started
with use_quantile_norm=False (runs/rtcf-20261031/repair/diagnosis.md). Both maps are affine per dimension, the DCT
is linear and orthonormal, and every per-dimension constant (state offset, mean or q01) lands in the f = 0 (DC)
coefficient. So for f > 0

    C[f, d] = s_d * DCT(A)[f, d],   s_d = 1 / (std_d + 1e-6)          (mode "zscore", the default)
                                    s_d = 2 / (q99_d - q01_d + 1e-6)  (mode "quantile")

The mode must equal the policy server's normalization; `token_grid_error` detects a mismatch on live chunks
(f > 0 coefficients of a served chunk are integers / 10 only in the server's own mode).

computed from the executable action chunk A the server returns. DC is masked by Eq. 10, so the correction never
needs the state offset. `coefficients(A, s)` returns s * DCT(A) for every f (its DC row is in executable-action
coordinates, not the token's model coordinates; it is only used by the Time-Domain ablation).

Pure functions only (no state); the parity script imports them:
    dct_matrix(H) -> [H, H] orthonormal DCT-II
    coefficients(chunk [H, A], scale [A]) -> C [H, A]
    quantize(C, token_scale=10) -> int64 tokens-equivalent  /  dequantize(q, token_scale) -> C
    load_scale(norm_stats_json, sha256=None, mode="zscore") -> s [A]      (default mode changed to zscore in v0.4)
    load_norm_bounds(norm_stats_json) -> (q01 [A], q99 [A])
    norm_center(norm_stats_json, mode="zscore") -> the unnormalized value of a normalized 0 (mean, or (q01+q99+eps)/2)
    token_grid_error(chunk, scale, token_scale=10) -> max |10 s C - round(10 s C)| over f > 0
    is_openpi_zero_chunk(chunk, state, q01, q99, tol=1e-4) -> bool
        exact signature of openpi's zero chunk after a failed FAST decode (HF FAST `decode` and openpi
        `extract_actions` silently return zeros): every row identical, gripper == (q01+q99)/2, and motion dims
        == (q01+q99)/2 + state[0:6] (openpi AbsoluteActions). A genuinely stationary chunk does not match it.
    rtcf_correct(native_chunk, memory_coef, scale, motion_dims, f_cut, delta_max, lambda0,
                 action_limit=1.0, mode="spectral") -> (corrected [H, A] in native dtype, info dict)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

NORM_EPS = 1e-6   # openpi transforms.Normalize / Unnormalize quantile epsilon


@lru_cache(maxsize=16)
def _dct_matrix_cached(h: int) -> np.ndarray:
    n = np.arange(h, dtype=np.float64)
    k = n[:, None]
    m = np.cos(np.pi * (2.0 * n[None, :] + 1.0) * k / (2.0 * h))
    m[0, :] *= np.sqrt(1.0 / h)
    m[1:, :] *= np.sqrt(2.0 / h)
    m.setflags(write=False)
    return m


def dct_matrix(h: int) -> np.ndarray:
    """Orthonormal DCT-II matrix M (scipy.fft.dct(x, axis=0, norm='ortho') == M @ x; inverse is M.T)."""
    if h < 1:
        raise ValueError("chunk length must be positive")
    return _dct_matrix_cached(int(h))


NORM_MODES = ("zscore", "quantile")


def _action_stats(norm_stats_path: str | Path, expected_sha256: Optional[str] = None) -> dict:
    import hashlib
    raw = Path(norm_stats_path).read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"norm stats {norm_stats_path} do not match the pinned sha256")
    stats = json.loads(raw)
    return {k: np.asarray(v, dtype=np.float64) for k, v in stats.get("norm_stats", stats)["actions"].items()}


def load_scale(norm_stats_path: str | Path, expected_sha256: Optional[str] = None, mode: str = "zscore") -> np.ndarray:
    """s_d of the policy's action normalization: zscore 1 / (std + 1e-6); quantile 2 / (q99 - q01 + 1e-6)."""
    st = _action_stats(norm_stats_path, expected_sha256)
    if mode == "zscore":
        return 1.0 / (st["std"] + NORM_EPS)
    if mode == "quantile":
        return 2.0 / (st["q99"] - st["q01"] + NORM_EPS)
    raise ValueError(f"unknown normalization mode {mode!r}; expected one of {NORM_MODES}")


def norm_center(norm_stats_path: str | Path, mode: str = "zscore", expected_sha256: Optional[str] = None) -> np.ndarray:
    """openpi's Unnormalize(0): zscore -> mean; quantile -> (q01 + q99 + 1e-6) / 2."""
    st = _action_stats(norm_stats_path, expected_sha256)
    if mode == "zscore":
        return st["mean"].copy()
    if mode == "quantile":
        return (st["q01"] + st["q99"] + NORM_EPS) / 2.0
    raise ValueError(f"unknown normalization mode {mode!r}; expected one of {NORM_MODES}")


def token_grid_error(chunk: np.ndarray, scale: np.ndarray, token_scale: float = 10.0) -> float:
    """max |token_scale * C - round(token_scale * C)| over f > 0. ~1e-5 for a chunk the server decoded from FAST
    tokens in the same normalization mode; O(0.1-0.5) when the mode (scale) is wrong."""
    x = token_scale * coefficients(chunk, scale)[1:]
    return float(np.max(np.abs(x - np.round(x)))) if x.size else 0.0


def load_norm_bounds(norm_stats_path: str | Path, expected_sha256: Optional[str] = None) -> tuple:
    """(q01, q99) of the 'actions' entry of an openpi norm_stats.json."""
    import hashlib
    raw = Path(norm_stats_path).read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"norm stats {norm_stats_path} do not match the pinned sha256")
    stats = json.loads(raw)
    stats = stats.get("norm_stats", stats)["actions"]
    return np.asarray(stats["q01"], dtype=np.float64), np.asarray(stats["q99"], dtype=np.float64)


def openpi_zero_chunk_row(state: np.ndarray, q01: np.ndarray = None, q99: np.ndarray = None, delta_dims: int = 6,
                          center: Optional[np.ndarray] = None) -> np.ndarray:
    """The action row openpi returns for an all-zero normalized chunk: Unnormalize(0) (`center`: the mean under
    z-score, (q01+q99+1e-6)/2 under quantile; q01/q99 kept for backward compatibility = quantile), then
    AbsoluteActions adds state[0:delta_dims] (pi0_fast_libero: make_bool_mask(6, -1))."""
    if center is not None:
        row = np.asarray(center, np.float64).copy()
    else:
        row = (np.asarray(q01, np.float64) + np.asarray(q99, np.float64) + NORM_EPS) / 2.0
    st = np.asarray(state, dtype=np.float64).reshape(-1)
    row[:delta_dims] = row[:delta_dims] + st[:delta_dims]
    return row


def is_openpi_zero_chunk(chunk: np.ndarray, state: Optional[np.ndarray], q01: np.ndarray = None,
                         q99: np.ndarray = None, tol: float = 1e-4, delta_dims: int = 6,
                         center: Optional[np.ndarray] = None) -> bool:
    """Exact signature of a failed FAST decode (critic B4 / audit C3). Needs the observation state the policy saw;
    without a state (e.g. HarnessStudio's synthetic tier-1 benchmark) it returns False."""
    a = np.asarray(chunk, dtype=np.float64)
    width = len(center) if center is not None else len(q01)
    if state is None or a.ndim != 2 or a.shape[1] != width or np.asarray(state).size < delta_dims:
        return False
    if not np.all(np.abs(a - a[0:1]) <= tol):
        return False
    return bool(np.all(np.abs(a[0] - openpi_zero_chunk_row(state, q01, q99, delta_dims, center)) <= tol))


def coefficients(chunk: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalized temporal DCT coefficients C = s * DCT(chunk) (time axis 0), shape [H, A], float64."""
    a = np.asarray(chunk, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a chunk [H, A], got shape {a.shape}")
    s = np.asarray(scale, dtype=np.float64).reshape(-1)
    if s.shape[0] != a.shape[1]:
        raise ValueError(f"scale has {s.shape[0]} dims, chunk has {a.shape[1]}")
    return (dct_matrix(a.shape[0]) @ a) * s[None, :]


def quantize(coef: np.ndarray, token_scale: float = 10.0) -> np.ndarray:
    """FAST's integer coefficients round(token_scale * C) (np.around, as the FAST processor)."""
    return np.around(np.asarray(coef, dtype=np.float64) * token_scale).astype(np.int64)


def dequantize(q: np.ndarray, token_scale: float = 10.0) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / token_scale


def rtcf_correct(native_chunk: np.ndarray, memory_coef: np.ndarray, scale: np.ndarray,
                 motion_dims: Sequence[int], f_cut: int, delta_max: float, lambda0: float,
                 action_limit: Optional[float] = 1.0, mode: str = "spectral") -> tuple[np.ndarray, dict]:
    """Eq. 10-11 on one chunk.

    native_chunk: A^pi [H, A] (executable actions from the server).
    memory_coef:  C^mem [H, A] in normalized coefficient units (= FAST tokens / token_scale).
    mode "spectral":    dC = G * clip(C^mem - C^pi, -delta_max, delta_max), G = 1[0 < f < f_cut] 1[d in motion];
                        A_corr = A^pi + IDCT(lambda0 * dC) / s.
    mode "time_domain": the Time-Domain ablation (inferred): the same clipped, lambda0-scaled residual on the
                        normalized time-domain actions of the motion dims (all frequencies incl. DC). It needs
                        memory_coef in EXECUTABLE coordinates (s * DCT(A_mem), as the Preparer stores it). Raw FAST
                        tokens / 10 carry a model-coordinate DC row (state offset, q01 shift), which would be mixed
                        with the executable DC of A^pi. Not run in rtcf-20261031; disclosed in the card.
    action_limit: the 'final action magnitude limit' (inferred: LIBERO bounds [-1, 1]); applied only to entries
                  the residual changed, so an all-zero residual (or lambda0 = 0) returns the native chunk bitwise.
    Non-motion dims (the gripper) are returned bitwise unchanged.
    """
    a = np.asarray(native_chunk)
    h, dim = a.shape
    s = np.asarray(scale, dtype=np.float64).reshape(-1)
    mem = np.asarray(memory_coef, dtype=np.float64)
    if mem.shape != (h, dim):
        raise ValueError(f"memory coefficients {mem.shape} do not match the chunk {a.shape}")
    motion = sorted({int(d) for d in motion_dims})
    if any(d < 0 or d >= dim for d in motion):
        raise ValueError(f"motion dims {motion} outside 0..{dim - 1}")
    m = dct_matrix(h)
    a64 = a.astype(np.float64)
    if mode == "spectral":
        c_pi = (m @ a64) * s[None, :]
        gate = np.zeros((h, dim), dtype=bool)
        fmax = max(0, min(int(f_cut), h))
        if fmax > 1 and motion:
            gate[1:fmax, motion] = True
        resid = np.where(gate, np.clip(mem - c_pi, -float(delta_max), float(delta_max)), 0.0)
        delta = (m.T @ (float(lambda0) * resid)) / s[None, :]
        clipped = int(np.count_nonzero(gate & (np.abs(mem - c_pi) > float(delta_max))))
        active = int(np.count_nonzero(gate))
    elif mode == "time_domain":
        x_pi = a64 * s[None, :]
        x_mem = (m.T @ mem)
        gate = np.zeros((h, dim), dtype=bool)
        if motion:
            gate[:, motion] = True
        resid = np.where(gate, np.clip(x_mem - x_pi, -float(delta_max), float(delta_max)), 0.0)
        delta = (float(lambda0) * resid) / s[None, :]
        clipped = int(np.count_nonzero(gate & (np.abs(x_mem - x_pi) > float(delta_max))))
        active = int(np.count_nonzero(gate))
    else:
        raise ValueError(f"unknown correction mode {mode!r}")
    delta[:, [d for d in range(dim) if d not in motion]] = 0.0
    out = a64 + delta
    changed = delta != 0.0
    limited = 0
    if action_limit is not None:
        lim = float(action_limit)
        limited = int(np.count_nonzero(changed & (np.abs(out) > lim)))
        out = np.where(changed, np.clip(out, -lim, lim), out)
    result = np.where(changed, out, a64).astype(a.dtype, copy=False)
    if not np.any(changed):
        result = a.copy()
    n_changed = int(np.count_nonzero(changed))
    info = {"residual_l2": float(np.sqrt(np.sum((lambda0 * resid) ** 2))), "clipped": clipped, "active": active,
            "clipped_frac": (clipped / active) if active else 0.0,           # masked coefficients hitting delta_max
            "changed": n_changed, "limit_clipped": limited,
            "limit_clipped_frac": (limited / n_changed) if n_changed else 0.0,  # changed actions hitting the limit
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0}
    return result, info
