"""Saturation-gated band reconstruction of pi0_fast_libero's extra-delta rotation (rx) as a DecisionLoop action_op.

pi0_fast_libero (openpi 981483d, LeRobotLiberoDataConfig(extra_delta_transform=True)) outputs m = a - s on dims 0-5;
after Unnormalize, openpi's AbsoluteActions adds the decision-time state back: chunk[:, :6] = m + s. The model cannot
emit m_rx below its training floor (action q01_rx), so when the observed state s_rx is above the checkpoint's state
band the executed a_rx = m_rx + s_rx keeps rotating outward (positive feedback). This op re-expresses the policy's
saturated delta relative to the nearest in-band state:

    a'_rx = m_rx + clip(s_rx, state_q01_rx, state_q99_rx),   m_rx = a_rx - s_rx      (every row of the chunk)

and does so ONLY when (i) s_rx is outside [state_q01_rx, state_q99_rx] AND (ii) the policy's output over the rows the
native loop executes (rows[:gate_rows], gate_rows = profile execute_horizon = 5) is pinned at its training extreme on
that side: min m_rx <= action_q01_rx + margin (state above the band) or max m_rx >= action_q99_rx - margin (state
below). Otherwise the input chunk object is returned unchanged (bitwise native). Algebraically this is a gated
AbsoluteActions(clip(s)) on rx.

`correction="signctrl"` is the development-only norm-matched control: same gate in its own rollout, same magnitude
|s_rx - clip(s_rx)|, sign from a public-input coin (sha256 of the episode's first public state bytes and the
per-episode decision counter): coin +1 applies exactly the restoring correction above, coin -1 the opposite sign,
a'_rx = a_rx + (s_rx - clip(s_rx)).

Public inputs only: obs[state_key] (the decision-time observation the policy received; DecisionLoop passes
ctx.observe() taken at decision start, and the policy gets the raw observation since no observation_op is set) and the
native chunk. No RNG, no simulator state, no case id/seed. Constructors only read the pinned asset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

PKG = Path(__file__).resolve().parent
CORRECTIONS = ("restore", "signctrl")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PKG / p


def load_norm_stats(path: str, sha256: Optional[str]) -> dict:
    data = resolve(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if sha256 and digest != sha256:
        raise RuntimeError(f"norm stats {path}: sha256 {digest} != pinned {sha256}")
    ns = json.loads(data)
    return ns.get("norm_stats", ns)


def is_openpi_zero_chunk(chunk: np.ndarray, state: np.ndarray, action_mean: np.ndarray, tol: float) -> bool:
    """openpi/HF FAST decode failure served as Unnormalize(0) (= mean under z-score) + state on dims 0-5: every row
    identical, dims 0-5 == mean + state, other dims == mean (PITFALLS P-20261101-06)."""
    rows = np.asarray(chunk, dtype=np.float64)
    if not np.all(np.abs(rows - rows[0:1]) <= tol):
        return False
    ref = np.asarray(action_mean, dtype=np.float64).copy()
    ref[:6] += state[:6]
    return bool(np.all(np.abs(rows[0] - ref) <= tol))


def public_coin(seed_digest: bytes, counter: int) -> int:
    """Deterministic +/-1 from public inputs only (C3)."""
    h = hashlib.sha256(seed_digest + int(counter).to_bytes(8, "little", signed=False)).digest()
    return 1 if (h[0] & 1) == 0 else -1


class BandGuardActionOp:
    def __init__(self, dim: int = 3, state_q01: float = 1.5003631114959717, state_q99: float = 3.2705323696136475,
                 action_q01: float = -3.306384325027466, action_q99: float = -1.4915390014648438,
                 margin: float = 0.02, gate_rows: int = 5, zero_chunk_tol: float = 1e-4,
                 norm_stats: str = "assets/pi0_fast_libero_norm_stats.json", norm_stats_sha256: Optional[str] = None,
                 state_key: str = "states", correction: str = "restore") -> None:
        if correction not in CORRECTIONS:
            raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}")
        ns = load_norm_stats(norm_stats, norm_stats_sha256)
        self.dim = int(dim)
        # The band/saturation edges are COPIES of the checkpoint's own statistics (not tuned): refuse any mismatch.
        copied = {"state_q01": (state_q01, ns["state"]["q01"][self.dim]),
                  "state_q99": (state_q99, ns["state"]["q99"][self.dim]),
                  "action_q01": (action_q01, ns["actions"]["q01"][self.dim]),
                  "action_q99": (action_q99, ns["actions"]["q99"][self.dim])}
        for name, (given, pinned) in copied.items():
            if float(given) != float(pinned):
                raise ValueError(f"{name}={given!r} differs from the pinned norm_stats value {pinned!r}")
        self.sq01, self.sq99 = float(state_q01), float(state_q99)
        self.aq01, self.aq99 = float(action_q01), float(action_q99)
        self.margin = float(margin)
        self.gate_rows = int(gate_rows)
        if self.gate_rows < 1:
            raise ValueError("gate_rows must be >= 1")
        self.zero_chunk_tol = float(zero_chunk_tol)
        self.action_mean = np.asarray(ns["actions"]["mean"], dtype=np.float64)
        self.state_key = state_key
        self.correction = correction
        self._seed: Optional[bytes] = None   # sha256 of the episode's first public state (signctrl coin)
        self._diag: dict = {}
        self.stats = {"decisions": 0, "active": 0, "active_above": 0, "active_below": 0, "skip_no_state": 0,
                      "skip_state_short": 0, "skip_chunk_shape": 0, "skip_nonfinite": 0, "skip_zero_chunk": 0,
                      "out_of_band_unsaturated": 0}
        if correction == "signctrl":
            self.stats.update({"coin_plus": 0, "coin_minus": 0})

    def diagnostics(self) -> dict:
        return dict(self._diag)

    # -- pure decision rule (also used by the CPU replay) ----------------------------------------------
    def gate(self, rows_dim: np.ndarray, s: float) -> int:
        """+1 (above band, saturated at the floor), -1 (below band, saturated at the ceiling) or 0."""
        m = np.asarray(rows_dim, dtype=np.float64)[: self.gate_rows] - s
        if s > self.sq99 and float(m.min()) <= self.aq01 + self.margin:
            return 1
        if s < self.sq01 and float(m.max()) >= self.aq99 - self.margin:
            return -1
        return 0

    def _skip(self, reason: str, extra: Optional[dict] = None) -> None:
        self.stats["skip_" + reason] += 1
        self._diag = {"t": self.stats["decisions"], "active": False, "skip": reason, **(extra or {}),
                      "stats": dict(self.stats)}

    def transform(self, chunk, obs, memory):
        self.stats["decisions"] += 1
        t = self.stats["decisions"]
        state = obs.get(self.state_key) if isinstance(obs, Mapping) else None
        s_vec = None
        if state is not None:
            try:
                s_vec = np.asarray(state, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                s_vec = None
        if self.correction == "signctrl" and self._seed is None:
            # first decision of the episode: the coin's seed is the first public state the policy saw
            self._seed = hashlib.sha256(b"" if s_vec is None else s_vec.tobytes()).digest()
        if state is None:
            self._skip("no_state")
            return chunk
        if s_vec is None or s_vec.size < 6:
            self._skip("state_short")
            return chunk
        c = np.asarray(chunk)
        if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] != self.action_mean.size:
            self._skip("chunk_shape")
            return chunk
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(s_vec[:6]))):
            self._skip("nonfinite")
            return chunk
        if is_openpi_zero_chunk(c, s_vec, self.action_mean, self.zero_chunk_tol):
            self._skip("zero_chunk")
            return chunk
        s = float(s_vec[self.dim])
        rows = c[:, self.dim].astype(np.float64)
        m_gate = rows[: self.gate_rows] - s
        info = {"s": s, "m_gate_min": float(m_gate.min()), "m_gate_max": float(m_gate.max())}
        side = self.gate(rows, s)
        if side == 0:
            if s > self.sq99 or s < self.sq01:
                self.stats["out_of_band_unsaturated"] += 1
            self._diag = {"t": t, "active": False, "skip": None, **info, "stats": dict(self.stats)}
            return chunk
        clipped = min(max(s, self.sq01), self.sq99)
        restored = (rows - s) + clipped                  # a'_rx = m_rx + clip(s_rx), every row
        coin = None
        if self.correction == "restore":
            new = restored
        else:
            coin = public_coin(self._seed, t)
            self.stats["coin_plus" if coin > 0 else "coin_minus"] += 1
            new = restored if coin > 0 else rows + (s - clipped)   # same magnitude, opposite sign
        out = np.array(c, copy=True)
        out[:, self.dim] = new.astype(out.dtype)
        self.stats["active"] += 1
        self.stats["active_above" if side > 0 else "active_below"] += 1
        self._diag = {"t": t, "active": True, "side": side, **info, "clip": clipped, "shift": s - clipped,
                      "correction": self.correction, **({"coin": coin} if coin is not None else {}),
                      "stats": dict(self.stats)}
        return out
