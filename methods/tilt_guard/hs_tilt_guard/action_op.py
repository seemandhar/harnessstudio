"""tilt-guard: one composite DecisionLoop action_op with three switchable components for pi0_fast_libero on LIBERO.

  Z  decode-failure substitution (round-1 GENERAL P2, r1g-p2)            zero = "off" | "substitute" | "signctrl"
  G  saturation-gated rx band guard (round-1 GENERAL P1, r1g-p1)         guard = False | True  (hs_band_guard, read-only)
  B  boundary tilt envelope (round-1 DOMAIN P1, BTE)                     envelope = "off" | "bte" | "yawctrl"

Fixed order (domain review C8):
  (1) Z detects the openpi zero chunk on the NATIVE chunk. If it substitutes, the executed rows come from the previous
      decision's already-guarded (post-G, pre-B) chunk rows exec_rows..2*exec_rows-1 with the gripper column set to the
      last executed gripper command (latched); no re-guarding with the current state. Cap (general review C4): at most
      2 consecutive substituted decisions - run 1 = tail (hold if no valid tail: first decision, previous decision
      substituted, or previous chunk shorter than 2*exec_rows), run 2 = hold (zero delta on dims 0-5 + latched
      gripper), run >= 3 = the native zero chunk passes through unchanged.
  (2) G (hs_band_guard.action_op.BandGuardActionOp, built from its frozen card params) runs on every non-substituted
      chunk with the decision-time observation; it is itself a no-op on zero chunks.
  (3) B runs last, on whatever will execute (rows[:exec_rows]); never on an executed native zero chunk.
Translation (dims 0-2) and the gripper (dim 6) are never touched by G or B. Rows exec_rows.. are left as produced by
steps (1)-(2) (the candidate executes rows[:exec_rows] only; profile execute_horizon 5).

Boundary detector (ONE, frozen; domain review C2): per decision j, with g = sign(native chunk row 0 gripper) and
width = states[6] - states[7] (decision-time state):
  - a decision whose NATIVE chunk is the zero chunk neither increments nor resets the hold run (never a release);
  - g > 0 and width > width_min: run += 1;
  - g <= 0: if run >= hold_decisions -> boundary at j (FIRST boundary only; the detector then stops), else run = 0;
  - g > 0 and width <= width_min (empty close): neither increments nor resets.
This is the detector of ideation/scripts-domain/dev_activation.py and tilt_window.py (the source of the predeclared
activation table; 28 dev80 activated episodes), not the strictly-consecutive variant (25).
Window: decisions j with b <= j < b + window_steps / exec_rows (150 steps = 30 decisions, the boundary decision included).
B is active at decision j iff boundary found, j in the window, tilt(state) > theta, 8-d finite state, and the
executed chunk is not a native zero chunk. Then for each executed row (geometry.bte_rotation_rows): clip, remove the
negative k-part, add push = min(kappa (tilt - theta), omega_max) about +k, re-clip (C7).

Development-only controls (separate entry points, never Campaign arms):
  envelope = "yawctrl" (domain C9): same gate on its own rollout; per row, clip(a) + s * ||BTE(a) - clip(a)|| * e_z with
      one sign per decision s = coin(sha256("tilt-guard/yawctrl" | yaw_seed | decision index)) (fixed public seed).
  zero = "signctrl" (general P2 F3, C3): same Z gate and cap on its own rollout; executes z + r * (sub - z) with a
      per-dimension coin r in {+1,-1} from sha256(first public state bytes | decision index); gripper clipped [-1,1].

Public inputs only (obs[state_key], the native chunk, per-episode memory of this op); no RNG, no simulator state, no
case id, seed or arm. Constructor reads only the pinned norm-stats asset (and the guard's own pinned asset).
Every state lives on self (DecisionLoop builds components per episode); the null arm runs this op as shadow.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from . import geometry as geo

PKG = Path(__file__).resolve().parent
ENVELOPES = ("off", "bte", "yawctrl")
ZEROS = ("off", "substitute", "signctrl")
DETECTOR = "first-boundary/hold4-loose"   # the one frozen detector (C2)


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
    identical, dims 0-5 == mean + state, other dims == mean (PITFALLS P-20261101-06; same rule as hs_band_guard)."""
    rows = np.asarray(chunk, dtype=np.float64)
    if not np.all(np.abs(rows - rows[0:1]) <= tol):
        return False
    ref = np.asarray(action_mean, dtype=np.float64).copy()
    ref[:6] += np.asarray(state, dtype=np.float64)[:6]
    return bool(np.all(np.abs(rows[0] - ref) <= tol))


def _bits(data: bytes, n: int) -> list:
    h = hashlib.sha256(data).digest()
    return [1 if ((h[i // 8] >> (i % 8)) & 1) == 0 else -1 for i in range(n)]


def yaw_coin(yaw_seed: int, decision: int) -> int:
    """+/-1 from a fixed public seed and the decision index only (domain C9)."""
    return _bits(b"tilt-guard/yawctrl" + int(yaw_seed).to_bytes(8, "little", signed=False)
                 + int(decision).to_bytes(8, "little", signed=False), 1)[0]


def zero_coins(seed_digest: bytes, decision: int, n: int) -> list:
    """Per-dimension +/-1 coins from public inputs only (general C3): first public state digest + decision index."""
    return _bits(b"tilt-guard/zerosign" + seed_digest + int(decision).to_bytes(8, "little", signed=False), n)


class BoundaryDetector:
    """The one frozen boundary detector (see module docstring)."""

    def __init__(self, hold_decisions: int = 4, width_min: float = 0.004) -> None:
        self.hold = int(hold_decisions)
        self.width_min = float(width_min)
        self.run = 0
        self.boundary: Optional[int] = None

    def update(self, decision: int, g: float, width: float, zero: bool) -> bool:
        """Feed one decision; True exactly at the boundary decision."""
        if self.boundary is not None or zero:
            return False
        if g > 0 and width > self.width_min:
            self.run += 1
        elif g <= 0:
            if self.run >= self.hold:
                self.boundary = int(decision)
                return True
            self.run = 0
        return False


def _import(dotted: str):
    module, _, attr = dotted.partition(":")
    return getattr(importlib.import_module(module), attr)


class TiltGuardActionOp:
    def __init__(self, guard: bool = False, envelope: str = "off", zero: str = "off",
                 guard_factory: str = "hs_band_guard.action_op:BandGuardActionOp",
                 guard_params: Optional[Mapping[str, Any]] = None,
                 exec_rows: int = 5, theta_deg: float = 55.0, window_steps: int = 150, hold_decisions: int = 4,
                 width_min: float = 0.004, kappa_per_rad: float = 0.5, omega_max: float = 0.2,
                 singular_eps: float = 1e-3, yaw_seed: int = 20261101, zero_chunk_tol: float = 1e-4,
                 zero_max_consecutive: int = 2, norm_stats: str = "assets/pi0_fast_libero_norm_stats.json",
                 norm_stats_sha256: Optional[str] = None, state_key: str = "states",
                 detector: str = DETECTOR) -> None:
        if envelope not in ENVELOPES:
            raise ValueError(f"envelope must be one of {ENVELOPES}, got {envelope!r}")
        if zero not in ZEROS:
            raise ValueError(f"zero must be one of {ZEROS}, got {zero!r}")
        if detector != DETECTOR:
            raise ValueError(f"only the frozen detector {DETECTOR!r} is implemented, got {detector!r}")
        self.exec_rows = int(exec_rows)
        if self.exec_rows < 1 or int(window_steps) % self.exec_rows:
            raise ValueError("exec_rows must be >= 1 and divide window_steps")
        if int(zero_max_consecutive) < 1:
            raise ValueError("zero_max_consecutive must be >= 1")
        ns = load_norm_stats(norm_stats, norm_stats_sha256)
        self.action_mean = np.asarray(ns["actions"]["mean"], dtype=np.float64)
        self.action_dim = self.action_mean.size
        self.use_guard = bool(guard)
        self.envelope = envelope
        self.zero = zero
        self.theta = float(np.radians(theta_deg))
        self.window_dec = int(window_steps) // self.exec_rows
        self.kappa = float(kappa_per_rad)
        self.omega_max = float(omega_max)
        self.singular_eps = float(singular_eps)
        self.yaw_seed = int(yaw_seed)
        self.zero_tol = float(zero_chunk_tol)
        self.zero_max = int(zero_max_consecutive)
        self.state_key = state_key
        self.guard = _import(guard_factory)(**dict(guard_params or {})) if self.use_guard else None
        self.detector = BoundaryDetector(hold_decisions, width_min)
        # per-episode memory (fresh object per episode)
        self.t = 0
        self.zero_run = 0
        self.prev_chunk: Optional[np.ndarray] = None     # previous decision's post-G (pre-B) chunk, if not substituted
        self.prev_substituted = False
        self.last_grip: Optional[float] = None          # last executed gripper command (row exec_rows-1 of the output)
        self.prev_k: Optional[np.ndarray] = None
        self._seed: Optional[bytes] = None
        self._diag: dict = {}
        self.stats = {"decisions": 0, "skip_chunk": 0, "zero_detected": 0, "z_tail": 0, "z_hold": 0,
                      "z_passthrough": 0, "z_max_run": 0, "g_active": 0, "b_active": 0, "b_rows_rescaled": 0,
                      "k_fallback": 0, "boundary_decision": None}

    def diagnostics(self) -> dict:
        return dict(self._diag)

    # ---------------------------------------------------------------------------------------------------------------
    def _state(self, obs) -> Optional[np.ndarray]:
        state = obs.get(self.state_key) if isinstance(obs, Mapping) else None
        if state is None:
            return None
        try:
            return np.asarray(state, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None

    def _substitute(self, c: np.ndarray) -> tuple:
        """Return (substitute chunk, mode) for a zero chunk at zero_run <= zero_max (step 1)."""
        n = self.exec_rows
        latched = self.last_grip if self.last_grip is not None else float(c[0, 6])
        sub = np.zeros(c.shape, dtype=np.float64)
        tail_ok = (self.zero_run == 1 and self.prev_chunk is not None and not self.prev_substituted
                   and self.prev_chunk.shape[0] >= 2 * n and self.prev_chunk.shape[1] == c.shape[1])
        mode = "hold"
        if tail_ok:
            m = min(n, c.shape[0])
            sub[:m] = self.prev_chunk[n:n + m]
            mode = "tail"
        sub[:, 6] = latched
        return sub, mode

    def transform(self, chunk, obs, memory):
        t = self.t
        self.t += 1
        self.stats["decisions"] += 1
        s = self._state(obs)
        if self.zero == "signctrl" and self._seed is None:
            self._seed = hashlib.sha256(b"" if s is None else s.tobytes()).digest()
        diag: dict = {"t": t}
        c = np.asarray(chunk)
        if (c.ndim != 2 or c.shape[0] < 1 or c.shape[1] != self.action_dim or not np.all(np.isfinite(c))):
            self.stats["skip_chunk"] += 1
            self.prev_chunk, self.prev_substituted, self.zero_run = None, False, 0
            self._diag = {**diag, "skip": "chunk", "stats": dict(self.stats)}
            return chunk
        s6 = s is not None and s.size >= 6 and bool(np.all(np.isfinite(s[:6])))
        s8 = s is not None and s.size >= 8 and bool(np.all(np.isfinite(s[:8])))
        is_zero = bool(s6 and is_openpi_zero_chunk(c, s, self.action_mean, self.zero_tol))
        self.zero_run = self.zero_run + 1 if is_zero else 0
        if is_zero:
            self.stats["zero_detected"] += 1
            self.stats["z_max_run"] = max(self.stats["z_max_run"], self.zero_run)
        # boundary detector (always fed, for diagnostics in every variant)
        if s8:
            if self.detector.update(t, float(np.sign(c[0, 6])), float(s[6] - s[7]), is_zero):
                self.stats["boundary_decision"] = t
        diag.update({"zero": is_zero, "zero_run": self.zero_run, "boundary": self.detector.boundary,
                     "hold_run": self.detector.run})

        # (1) Z on the native chunk
        out = chunk
        substituted = None
        if self.zero != "off" and is_zero:
            if self.zero_run <= self.zero_max:
                sub, mode = self._substitute(c)
                if self.zero == "signctrl":
                    r = np.asarray(zero_coins(self._seed, t, c.shape[1]), dtype=np.float64)
                    cz = c.astype(np.float64)
                    sub = cz + r[None, :] * (sub - cz)
                    sub[:, 6] = np.clip(sub[:, 6], -1.0, 1.0)
                    diag["z_coins"] = [int(x) for x in r]
                out = sub.astype(c.dtype)
                substituted = mode
                self.stats["z_" + mode] += 1
            else:
                self.stats["z_passthrough"] += 1
                diag["z_mode"] = "passthrough"
        if substituted:
            diag["z_mode"] = substituted

        # (2) G on non-substituted chunks
        if substituted is None and self.guard is not None:
            g_out = self.guard.transform(out, obs, memory)
            gd = self.guard.diagnostics() if callable(getattr(self.guard, "diagnostics", None)) else {}
            diag["guard"] = {k: v for k, v in gd.items() if k != "stats"}
            if g_out is not out:
                self.stats["g_active"] += 1
            out = g_out
        # memory for a later tail: post-G, pre-B
        if substituted is None:
            self.prev_chunk = np.array(out, dtype=np.float64, copy=True)
            self.prev_substituted = False
        else:
            self.prev_chunk = None
            self.prev_substituted = True

        # (3) B / yaw control last
        executed_native_zero = is_zero and substituted is None
        if s8:
            R = geo.rotmat(s[3:6])
            tilt = geo.tilt_of_matrix(R)
            diag["tilt_deg"] = float(np.degrees(tilt))
            b = self.detector.boundary
            in_window = b is not None and b <= t < b + self.window_dec
            if (self.envelope != "off" and in_window and tilt > self.theta and not executed_native_zero):
                k, src = geo.envelope_axis(R, self.prev_k, self.singular_eps)
                self.prev_k = k
                if src != "cross":
                    self.stats["k_fallback"] += 1
                push = geo.push_magnitude(tilt, self.theta, self.kappa, self.omega_max)
                n = min(self.exec_rows, len(out))
                base = np.asarray(out, dtype=np.float64)[:n, 3:6]
                new, removed, rescaled = geo.bte_rotation_rows(base, k, push)
                o2 = np.array(out, copy=True)
                info = {"k": [float(x) for x in k], "k_source": src, "push": push,
                        "removed": [float(x) for x in removed]}
                if self.envelope == "bte":
                    o2[:n, 3:6] = new
                    self.stats["b_rows_rescaled"] += int(rescaled.sum())
                    info["rescaled"] = int(rescaled.sum())
                    info["k_component_after"] = [float(x) for x in (new @ k)]
                else:
                    norms = np.linalg.norm(new - np.clip(base, -1.0, 1.0), axis=1)
                    sign = yaw_coin(self.yaw_seed, t)
                    yaw = geo.yaw_rotation_rows(base, norms, sign)
                    o2[:n, 3:6] = yaw
                    realized = np.linalg.norm(np.clip(yaw, -1.0, 1.0) - np.clip(base, -1.0, 1.0), axis=1)
                    info.update({"yaw_sign": sign, "yaw_norms": [float(x) for x in norms],
                                 "yaw_realized_norms": [float(x) for x in realized]})
                out = o2
                self.stats["b_active"] += 1
                diag["envelope"] = {"active": True, "mode": self.envelope, **info}
            else:
                cross = np.cross(R[:, 2], geo.DOWN)
                if float(np.linalg.norm(cross)) >= self.singular_eps:
                    self.prev_k = cross / float(np.linalg.norm(cross))
        out_arr = np.asarray(out)
        self.last_grip = float(out_arr[min(self.exec_rows, len(out_arr)) - 1, 6])
        diag["active"] = {"z": substituted is not None, "g": bool(diag.get("guard", {}).get("active", False)),
                          "b": bool(diag.get("envelope", {}).get("active", False))}
        self._diag = {**diag, "stats": dict(self.stats)}
        return out
