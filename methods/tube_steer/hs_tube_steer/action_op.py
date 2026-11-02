"""tube-steer: success-tube orientation steering for openpi pi0_fast_libero on vla-eval LIBERO as one DecisionLoop
action_op (round-5 MEMORY P1, runs/rtcfpp-20261101/ideation/r5-memory.json; review r5-proposal-review.json C1-C13;
amendment A5). Self-contained: no import of any other method package (geometry helpers copied here).

Memory (C1): assets/tube_memory.npz (sha256 fcf83383...) + tube_memory.json (00d2e87b...), byte-identical copies of the
frozen artifact built from the 127 SUCCESSFUL S2 memory episodes (init 10-29, seed 7, native collection mode): per episode
the decision-time gripper z-axis (world frame) per decision, keyed by the public instruction. Both sha256 are checked at
construction; a mismatch raises (never a silent abstain). t8 has no success memory -> abstain (bitwise native).

Gate (C3, pinned to ideation/scripts-r5-memory/dev80_tube.py): at decision q (decision index; step 5q) with decision-time
state s = obs['states'], z = R(s[3:6])[:, 2],
  D(q) = min over same-instruction memories m and q' in [q-W, q+W] (clamped to m; a memory shorter than q-W contributes
         its last row) of angle(z_m[q'], z) in degrees; z_ref = the argmin row (first memory, first row on ties).
  arm at q iff not armed, D(q-1) > eps and D(q) > eps and q <= arm_by_decision (39, step <= 195) -- re-arming after a
  disarm uses the same rule, so it is possible only before step 200;
  while armed: if D(q) <= release (7.5 deg) -> disarm (no steer at q); if q > hold_limit_step / 5 (step > 300) ->
  disarm (no steer); else steer.
Steer (T): u = unit(z x z_ref) (world frame; a positive OSC rotation about u moves z toward z_ref), c = clip(D/eps, 0.5,
  1.0); for each executed row r < exec_rows: a_rot' = a_rot - (a_rot . u) u + c u, then clip to [-1, 1] (as specified, no
  pre-clip). If the NATIVE chunk is the openpi zero chunk: rotation = c u (other rotation components zeroed). Translation
  (dims 0-2), gripper (dim 6) and rows >= exec_rows are native. Every non-steered decision returns the input object
  (bitwise native).
Ortho control (C6, development-only): identical arming law and first arming decision; replaces the component along
  u_perp = unit(z x u) with s c (s = +/-1 from sha256('tube-steer/ortho' | decision index)), keeps the native component
  along u; zero-chunk rows: rotation = s c u_perp; steers at most ortho_cap decisions (6) or until D <= 7.5 deg, whichever
  first; then native for the rest of the episode (no re-arm).
Diagnostics per decision (C4): tube_found, D, z, z_ref, u, c, armed, native and executed component along the steered axis
(row mean over executed rows), zero-chunk flag, op wall time. Public inputs only (obs['states'], obs instruction, the
native chunk); no RNG, no simulator state, case id, seed or arm. Memory/state lives on self (built per episode).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

PKG = Path(__file__).resolve().parent
MODES = ("steer", "ortho")


def _read_pinned(path: str, sha256: str) -> bytes:
    p = Path(path)
    p = p if p.is_absolute() else PKG / p
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{path}: sha256 {digest} != pinned {sha256} (refusing to start)")
    return data


def rotmat(aa) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def zaxis(aa) -> np.ndarray:
    return rotmat(aa)[:, 2]


def is_openpi_zero_chunk(chunk: np.ndarray, state: np.ndarray, action_mean: np.ndarray, tol: float) -> bool:
    """openpi/HF FAST decode failure served as Unnormalize(0) (= mean, z-score) + state on dims 0-5 (P-20261101-06)."""
    rows = np.asarray(chunk, dtype=np.float64)
    if not np.all(np.abs(rows - rows[0:1]) <= tol):
        return False
    ref = np.asarray(action_mean, dtype=np.float64).copy()
    ref[:6] += np.asarray(state, dtype=np.float64)[:6]
    return bool(np.all(np.abs(rows[0] - ref) <= tol))


def tube_deviation(z: np.ndarray, memories, q: int, W: int):
    """D(q) in degrees and z_ref (dev80_tube.py convention). memories: list of [n_m, 3] arrays (decision-indexed)."""
    best, ref = np.inf, None
    for zm in memories:
        nm = len(zm)
        lo, hi = max(0, q - W), min(nm, q + W + 1)
        if lo >= hi:
            lo, hi = nm - 1, nm
        ang = np.degrees(np.arccos(np.clip(zm[lo:hi] @ z, -1.0, 1.0)))
        j = int(ang.argmin())
        if ang[j] < best:
            best, ref = float(ang[j]), np.asarray(zm[lo + j], dtype=np.float64)
    return best, ref


def ortho_coin(decision: int) -> int:
    h = hashlib.sha256(b"tube-steer/ortho" + int(decision).to_bytes(8, "little", signed=False)).digest()
    return 1 if (h[0] & 1) == 0 else -1


class TubeMemory:
    """Frozen success tube keyed by instruction (loaded once per process; sha256-checked)."""
    _cache: dict = {}

    @classmethod
    def load(cls, npz: str, npz_sha256: str, meta: str, meta_sha256: str) -> dict:
        key = (npz, npz_sha256, meta, meta_sha256)
        if key not in cls._cache:
            import io
            data = _read_pinned(npz, npz_sha256)
            info = json.loads(_read_pinned(meta, meta_sha256))
            if info.get("npz_sha256") != npz_sha256:
                raise RuntimeError("tube_memory.json does not describe the pinned npz")
            z = np.load(io.BytesIO(data))
            by_instr: dict = {}
            for e in info["episodes"]:
                if not 10 <= int(e["init"]) <= 29:
                    raise RuntimeError(f"memory episode {e['key']} is outside the reserved init 10-29")
                by_instr.setdefault(e["instruction"], []).append(np.asarray(z[e["key"] + "_z"], dtype=np.float64))
            cls._cache[key] = by_instr
        return cls._cache[key]


class TubeSteerActionOp:
    def __init__(self, mode: str = "steer", eps_deg: float = 15.0, consecutive: int = 2, arm_by_decision: int = 39,
                 window_decisions: int = 4, release_deg: float = 7.5, hold_limit_step: int = 300,
                 gain_floor: float = 0.5, gain_cap: float = 1.0, ortho_cap: int = 6, exec_rows: int = 5,
                 decision_steps: int = 5, zero_chunk_tol: float = 1e-4, state_key: str = "states",
                 instruction_key: str = "task_description",
                 memory_npz: str = "assets/tube_memory.npz", memory_npz_sha256: str = "",
                 memory_json: str = "assets/tube_memory.json", memory_json_sha256: str = "",
                 norm_stats: str = "assets/pi0_fast_libero_norm_stats.json", norm_stats_sha256: str = "") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if int(consecutive) != 2:
            raise ValueError("the frozen gate uses 2 consecutive decisions (C3)")
        self.mode = mode
        self.eps, self.release = float(eps_deg), float(release_deg)
        self.arm_by = int(arm_by_decision)
        self.W = int(window_decisions)
        self.hold_q = int(hold_limit_step) // int(decision_steps)
        self.gain_floor, self.gain_cap = float(gain_floor), float(gain_cap)
        self.ortho_cap = int(ortho_cap)
        self.exec_rows = int(exec_rows)
        self.zero_tol = float(zero_chunk_tol)
        self.state_key, self.instruction_key = state_key, instruction_key
        self.memory = TubeMemory.load(memory_npz, memory_npz_sha256, memory_json, memory_json_sha256)
        ns = json.loads(_read_pinned(norm_stats, norm_stats_sha256))
        ns = ns.get("norm_stats", ns)
        self.action_mean = np.asarray(ns["actions"]["mean"], dtype=np.float64)
        # per-episode state
        self.q = 0
        self.prev_D: Optional[float] = None
        self.armed = False
        self.first_arm: Optional[int] = None
        self.ortho_steps = 0
        self.ortho_done = False
        self._diag: dict = {}
        self.stats = {"decisions": 0, "tube_found": None, "armed_decisions": 0, "arm_events": 0, "first_arm": None,
                      "steered_decisions": 0, "zero_rows_steered": 0}

    def diagnostics(self) -> dict:
        return dict(self._diag)

    def gain(self, D: float) -> float:
        return float(np.clip(D / self.eps, self.gain_floor, self.gain_cap))

    def _finish(self, diag, out):
        self._diag = {**diag, "stats": dict(self.stats)}
        return out

    def transform(self, chunk, obs, memory):
        t0 = time.perf_counter()
        q = self.q
        self.q += 1
        self.stats["decisions"] += 1
        diag: dict = {"q": q}
        instr = obs.get(self.instruction_key) if isinstance(obs, Mapping) else None
        mems = self.memory.get(instr) if isinstance(instr, str) else None
        found = mems is not None
        if self.stats["tube_found"] is None:
            self.stats["tube_found"] = found
        diag["tube_found"] = found
        st = obs.get(self.state_key) if isinstance(obs, Mapping) else None
        c = np.asarray(chunk)
        s = None
        if st is not None:
            try:
                s = np.asarray(st, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                s = None
        valid = (found and s is not None and s.size >= 6 and bool(np.all(np.isfinite(s[:6])))
                 and c.ndim == 2 and c.shape[0] >= 1 and c.shape[1] == self.action_mean.size and bool(np.all(np.isfinite(c))))
        if not valid:
            self.prev_D = None
            diag["wall_s"] = time.perf_counter() - t0
            return self._finish({**diag, "skip": "abstain" if not found else "invalid", "armed": self.armed}, chunk)
        z = zaxis(s[3:6])
        D, zref = tube_deviation(z, mems, q, self.W)
        prev = self.prev_D
        self.prev_D = D
        diag.update({"D": D, "z": [float(x) for x in z], "z_ref": [float(x) for x in zref]})
        # arming / disarming (identical law in both modes; ortho never re-arms after its run ends)
        if not self.armed and not (self.mode == "ortho" and self.ortho_done):
            if prev is not None and prev > self.eps and D > self.eps and q <= self.arm_by:
                self.armed = True
                self.stats["arm_events"] += 1
                if self.first_arm is None:
                    self.first_arm = q
                    self.stats["first_arm"] = q
        steer = False
        if self.armed:
            if D <= self.release or q > self.hold_q:
                self.armed = False
                if self.mode == "ortho":
                    self.ortho_done = True
            elif self.mode == "ortho" and self.ortho_steps >= self.ortho_cap:
                self.armed = False
                self.ortho_done = True
            else:
                steer = True
        diag["armed"] = self.armed
        if self.armed:
            self.stats["armed_decisions"] += 1
        if not steer:
            diag["wall_s"] = time.perf_counter() - t0
            return self._finish(diag, chunk)
        cross = np.cross(z, zref)
        nrm = float(np.linalg.norm(cross))
        if nrm < 1e-9:                       # z == z_ref exactly: no restoring direction (cannot happen while D > 7.5)
            diag["wall_s"] = time.perf_counter() - t0
            return self._finish({**diag, "skip": "degenerate_axis"}, chunk)
        u = cross / nrm
        g = self.gain(D)
        if self.mode == "steer":
            axis, target = u, g
        else:
            up = np.cross(z, u)
            up /= float(np.linalg.norm(up))
            sign = ortho_coin(q)
            axis, target = up, sign * g
            diag["ortho_sign"] = sign
        n = min(self.exec_rows, c.shape[0])
        rot = np.asarray(c[:n, 3:6], dtype=np.float64)
        zero = is_openpi_zero_chunk(c, s, self.action_mean, self.zero_tol) if s.size >= 6 else False
        if zero:
            new = np.tile(target * axis, (n, 1))
            self.stats["zero_rows_steered"] += n
        else:
            new = rot - (rot @ axis)[:, None] * axis[None, :] + target * axis[None, :]
        new = np.clip(new, -1.0, 1.0)
        out = np.array(c, copy=True)
        out[:n, 3:6] = new
        self.stats["steered_decisions"] += 1
        if self.mode == "ortho":
            self.ortho_steps += 1
        diag.update({"steer": True, "u": [float(x) for x in u], "axis": [float(x) for x in axis], "c": g,
                     "zero_chunk": zero, "native_component": float(np.mean(np.clip(rot, -1, 1) @ axis)),
                     "executed_component": float(np.mean(new @ axis)), "wall_s": time.perf_counter() - t0})
        return self._finish(diag, out)
