"""STAR core, ported as literally as possible from the authors' released pi0.5 code.

Source: https://github.com/PKU-VaLuE-Lab/STAR @ 0eaa15fc9223734af024e928a82e40966724b739 (Apache-2.0)
  - StarConfig            <- src/openpi/inference/star/config.py:14-56
  - SlidingWindowNorm     <- src/openpi/inference/star/scheduler.py:14-42
  - ProprioPhaseScheduler <- src/openpi/inference/star/scheduler.py:58-327
  - StarLayerMixer        <- src/openpi/inference/star/controller_core_v2.py:12-64 (TorchStarLayerController)
Changes (all behaviour-neutral for the pi0.5 path, checked by runs/star-port-20261104/parity/):
  [c1] no openpi import; the config dataclass is local with the same fields, defaults and resolvers.
  [c2] StarLayerMixer.step takes an optional `positions` selector (None = every position, the released
       behaviour); the OpenVLA port uses it to restrict the EMA to the action-emitting positions (design doc
       port_design.md, unknown U3). With positions=None the arithmetic is the released code's.
  [c3] simpler_state_to_star(): SimplerEnv `states` [pos3, quat_wxyz4, gripper1] -> the released LIBERO layout
       [pos3, axis-angle3, gripper1] with the released client's _quat2axisangle (examples/libero/main.py:399-414).
       New code (the authors released no SimplerEnv client); disclosed re-derivation.
  [c4] TABLE7_OPENVLA_SIMPLER / TABLE7_PI05_LIBEROPRO / README_PI05_LIBEROPRO constant sets.
  [c5] QuatProprioPhaseScheduler: [pos3, quat4, gripper1] layout -> the released geodesic rotation branch.
  [c6] StarLayerMixer(shadow=True): computes beta, Delta and m exactly as the candidate does but returns the layer's
       own output object unchanged (overhead/shadow null arm of SETTING.md section 2; actions bitwise native).
  [c7] StarLayerMixer.activity(): logging-only per-layer edit ratio ||m^k - dh^k|| / ||dh^k|| at the last position
       (review critic-ext-2 P9); computed out-of-place in float32 in step(), moved to the host once per request.
Nothing here reads reward, success, simulator state, seeds or case ids. numpy only at import; torch is imported
lazily by StarLayerMixer (the policy server has it).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import dataclasses
import logging
import math
from typing import Any

import numpy as np


# ----------------------------------------------------------------------------------------------- config [c1]
@dataclasses.dataclass(frozen=True)
class StarConfig:
    """Configuration for STAR inference-time correction (config.py:14-56, same fields and defaults)."""

    beta_min: float = 0.25
    beta_max: float = 0.55
    beta_kappa: float = 1.0
    correction_last_n_layers: int = 6

    control_hz: float = 10.0
    stats_window_sec: float = 2.0
    stats_window_steps: int = 0
    norm_min_count: int = 8

    phase_coeff_c: float = 1.0
    phase_coeff_v: float = 1.0
    phase_coeff_j: float = -1.0
    phase_coeff_w: float = -1.0

    action_expert_index: int = 1

    def resolve_beta_bounds(self) -> tuple[float, float]:
        lo = float(self.beta_min)
        hi = float(self.beta_max)
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"STAR beta bounds must lie in [0, 1], got ({lo}, {hi})")
        return lo, hi

    def resolve_correction_start(self, total_layers: int) -> int:
        """1-based layer index where correction starts (last-N layers)."""
        if total_layers <= 0:
            return 1
        n = max(1, int(self.correction_last_n_layers))
        return max(1, int(total_layers) - n + 1)


def last_n_from_rho(total_layers: int, rho: float = 2.0 / 3.0) -> int:
    """Paper Eq. 2: p = floor(rho n), mixed layers p+1..n -> n - p layers (pi0.5: 18 -> 6; OpenVLA: 32 -> 11)."""
    p = int(math.floor(rho * int(total_layers)))
    return int(total_layers) - p


# [c4] constant sets. Table 7 = arXiv:2610.41009v1 Appendix B. README = the repo's pi0.5 LIBERO-PRO command.
TABLE7_OPENVLA_SIMPLER = dict(beta_min=0.25, beta_max=0.55, beta_kappa=4.75, phase_coeff_c=1.0, phase_coeff_v=1.0,
                              phase_coeff_j=-0.25, phase_coeff_w=-0.75)
TABLE7_PI05_LIBEROPRO = dict(beta_min=0.25, beta_max=0.55, beta_kappa=2.30, phase_coeff_c=0.25, phase_coeff_v=1.0,
                             phase_coeff_j=-0.125, phase_coeff_w=-0.25)
README_PI05_LIBEROPRO = dict(beta_min=0.25, beta_max=0.55, beta_kappa=4.25, phase_coeff_c=1.0, phase_coeff_v=1.0,
                             phase_coeff_j=-0.375, phase_coeff_w=-2.0, correction_last_n_layers=6)


# ------------------------------------------------------------------------------------- scheduler (verbatim logic)
class SlidingWindowNorm:
    """Causal z-score over a fixed-length history window."""

    def __init__(self, window_size: int, *, eps: float = 1e-6, min_count: int = 4):
        self._window_size = max(1, int(window_size))
        self._eps = float(eps)
        self._min_count = max(1, int(min_count))
        self._buffer: deque[float] = deque(maxlen=self._window_size)

    def reset(self) -> None:
        self._buffer.clear()

    def update(self, value: float) -> None:
        self._buffer.append(float(value))

    def normalize(self, value: float) -> float:
        if len(self._buffer) < self._min_count:
            return 0.0
        arr = np.asarray(self._buffer, dtype=np.float32)
        if arr.size < 2:
            return 0.0
        std = float(arr.std(ddof=1))
        if not np.isfinite(std):
            return 0.0
        z = (float(value) - float(arr.mean())) / (std + self._eps)
        if not np.isfinite(z):
            return 0.0
        return float(z)


@dataclasses.dataclass
class _PendingState:
    c_t: float
    v_t: float
    j_t: float
    w_t: float
    u_t: float
    r_t: float
    pos_cur: np.ndarray | None
    rot_cur: np.ndarray | None
    gripper_cur: float | None


class ProprioPhaseScheduler:
    """Compute STAR step context from raw proprioception observations (scheduler.py:58-327)."""

    def __init__(self, config: StarConfig):
        self._config = config
        self._beta_min, self._beta_max = config.resolve_beta_bounds()
        self._beta_kappa = float(config.beta_kappa)
        if config.stats_window_steps > 0:
            window_steps = int(config.stats_window_steps)
        else:
            window_steps = max(1, round(config.control_hz * config.stats_window_sec))

        effective_min_count = max(1, min(int(config.norm_min_count), window_steps))
        if effective_min_count < int(config.norm_min_count):
            logging.warning(
                "STAR norm_min_count=%d exceeds window_steps=%d; clipping min_count to %d.",
                int(config.norm_min_count), window_steps, effective_min_count,
            )

        self._c_norm = SlidingWindowNorm(window_steps, min_count=effective_min_count)
        self._v_norm = SlidingWindowNorm(window_steps, min_count=effective_min_count)
        self._j_norm = SlidingWindowNorm(window_steps, min_count=effective_min_count)
        self._w_norm = SlidingWindowNorm(window_steps, min_count=effective_min_count)
        self._r_window_size = window_steps
        self._r_min_count = effective_min_count
        self._r_buffer: deque[float] = deque(maxlen=self._r_window_size)
        self._r_eps = 1e-6

        self._prev_pos: np.ndarray | None = None
        self._prev2_pos: np.ndarray | None = None
        self._prev_rot: np.ndarray | None = None
        self._prev_gripper: float | None = None
        self._pending: _PendingState | None = None

    @property
    def window_steps(self) -> int:
        return self._r_window_size

    @property
    def min_count(self) -> int:
        return self._r_min_count

    def reset_episode(self) -> None:
        self._prev_pos = None
        self._prev2_pos = None
        self._prev_rot = None
        self._prev_gripper = None
        self._pending = None
        self._c_norm.reset()
        self._v_norm.reset()
        self._j_norm.reset()
        self._w_norm.reset()
        self._r_buffer.clear()

    def prepare_context(self, observation: Mapping[str, Any]) -> dict[str, float]:
        state = self._extract_state(observation)
        pos_cur, rot_cur, gripper_cur = self._decompose_state(state)

        c_t = v_t = j_t = w_t = u_t = 0.0
        dx_t: np.ndarray | None = None
        if pos_cur is not None and self._prev_pos is not None:
            dx_t = pos_cur - self._prev_pos
            v_t = float(np.linalg.norm(dx_t))
        if dx_t is not None and self._prev_pos is not None and self._prev2_pos is not None:
            dx_prev = self._prev_pos - self._prev2_pos
            c_t = self._safe_cos(dx_t, dx_prev)
            j_t = float(np.linalg.norm(dx_t - dx_prev))

        if rot_cur is not None and self._prev_rot is not None:
            w_t = self._rotation_distance(rot_cur, self._prev_rot)
        if gripper_cur is not None and self._prev_gripper is not None:
            u_t = abs(gripper_cur - self._prev_gripper)

        zc = self._clip(self._c_norm.normalize(c_t))
        zv = self._clip(self._v_norm.normalize(v_t))
        zj = self._clip(self._j_norm.normalize(j_t))
        zw = self._clip(self._w_norm.normalize(w_t))
        r_t = (
            float(self._config.phase_coeff_c) * zc
            + float(self._config.phase_coeff_v) * zv
            + float(self._config.phase_coeff_j) * zj
            + float(self._config.phase_coeff_w) * zw
        )
        r_t_tilde = self._normalize_r(r_t)
        beta_t = self._beta_from_r_tilde(r_t_tilde)

        self._pending = _PendingState(
            c_t=float(c_t), v_t=float(v_t), j_t=float(j_t), w_t=float(w_t), u_t=float(u_t), r_t=float(r_t),
            pos_cur=None if pos_cur is None else pos_cur.copy(),
            rot_cur=None if rot_cur is None else rot_cur.copy(),
            gripper_cur=None if gripper_cur is None else float(gripper_cur),
        )
        return {
            "c_t": float(c_t), "v_t": float(v_t), "j_t": float(j_t), "w_t": float(w_t), "u_t": float(u_t),
            "r_t": float(r_t), "r_t_tilde": float(r_t_tilde), "beta_t": float(beta_t), "g_t": float(beta_t),
        }

    def update_after_action(self) -> None:
        if self._pending is None:
            return
        self._c_norm.update(self._pending.c_t)
        self._v_norm.update(self._pending.v_t)
        self._j_norm.update(self._pending.j_t)
        self._w_norm.update(self._pending.w_t)
        self._r_buffer.append(float(self._pending.r_t))

        self._prev2_pos = None if self._prev_pos is None else self._prev_pos.copy()
        self._prev_pos = self._pending.pos_cur
        self._prev_rot = self._pending.rot_cur
        self._prev_gripper = self._pending.gripper_cur
        self._pending = None

    def ingest_state_history(self, state_history: Any, *, current_observation: Mapping[str, Any] | None = None) -> int:
        seq = self._to_state_sequence(state_history)
        if seq is None or seq.shape[0] == 0:
            return 0
        if current_observation is not None:
            current_state = self._extract_state(current_observation)
            if current_state is not None and self._states_close(seq[-1], current_state, atol=1e-4):
                seq = seq[:-1]
        if seq.shape[0] == 0:
            return 0
        replayed = 0
        for state in seq:
            self.prepare_context({"observation/state": state})
            self.update_after_action()
            replayed += 1
        return replayed

    def _normalize_r(self, r_t: float) -> float:
        if len(self._r_buffer) < self._r_min_count:
            return 0.0
        arr = np.asarray(self._r_buffer, dtype=np.float32)
        if arr.size < 2:
            return 0.0
        q25 = float(np.quantile(arr, 0.25))
        q50 = float(np.quantile(arr, 0.50))
        q75 = float(np.quantile(arr, 0.75))
        iqr = q75 - q25
        if not np.isfinite(iqr) or abs(iqr) < self._r_eps:
            return 0.0
        z = (float(r_t) - q50) / (iqr + self._r_eps)
        if not np.isfinite(z):
            return 0.0
        return float(z)

    def _beta_from_r_tilde(self, r_t_tilde: float) -> float:
        sigmoid_input = float(np.clip(self._beta_kappa * float(r_t_tilde), -60.0, 60.0))
        sig = float(1.0 / (1.0 + np.exp(-sigmoid_input)))
        beta = self._beta_min + (self._beta_max - self._beta_min) * sig
        return float(np.clip(beta, self._beta_min, self._beta_max))

    @staticmethod
    def _clip(value: float, clip_val: float = 2.5) -> float:
        return float(np.clip(value, -clip_val, clip_val))

    @staticmethod
    def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an < 1e-8 or bn < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    @staticmethod
    def _rotation_distance(cur: np.ndarray, prev: np.ndarray) -> float:
        cur = cur.reshape(-1).astype(np.float32, copy=False)
        prev = prev.reshape(-1).astype(np.float32, copy=False)
        if cur.size == 4 and prev.size == 4:
            q1 = cur[:4]
            q2 = prev[:4]
            n1 = float(np.linalg.norm(q1))
            n2 = float(np.linalg.norm(q2))
            if n1 < 1e-8 or n2 < 1e-8:
                return 0.0
            q1 = q1 / n1
            q2 = q2 / n2
            cos_sim = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
            return float(2.0 * np.arccos(cos_sim))
        dim = min(cur.size, prev.size)
        return float(np.linalg.norm(cur[:dim] - prev[:dim]))

    @staticmethod
    def _extract_state(observation: Mapping[str, Any]) -> np.ndarray | None:
        if "observation/state" in observation:
            return ProprioPhaseScheduler._to_state_vec(observation["observation/state"])
        if "state" in observation:
            return ProprioPhaseScheduler._to_state_vec(observation["state"])
        obs_obj = observation.get("observation")
        if isinstance(obs_obj, Mapping) and "state" in obs_obj:
            return ProprioPhaseScheduler._to_state_vec(obs_obj["state"])
        return None

    @staticmethod
    def _to_state_vec(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1)
        elif arr.ndim == 2:
            arr = arr[-1].reshape(-1)
        else:
            arr = arr.reshape(arr.shape[0], -1)[-1]
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _to_state_sequence(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(arr.shape[0], -1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _states_close(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-6) -> bool:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        dim = min(a.size, b.size)
        if dim == 0:
            return False
        return bool(np.allclose(a[:dim], b[:dim], atol=atol, rtol=0.0))

    @staticmethod
    def _decompose_state(state: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        if state is None:
            return None, None, None
        if state.size < 3:
            return None, None, None
        pos = state[:3]
        rot: np.ndarray | None = None
        if state.size >= 6:
            rot = state[3:6]
        gripper: float | None = None
        if state.size >= 8:
            gripper = float(np.mean(state[6:8]))
        elif state.size >= 7:
            gripper = float(state[6])
        return pos, rot, gripper


# ------------------------------------------------------------------------------ SimplerEnv state adapter [c3]
def _quat2axisangle(quat):
    """examples/libero/main.py:399-414 (copied from robosuite), quat in (x, y, z, w)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def simpler_state_to_star(states) -> np.ndarray:
    """SimplerEnv `states` = eef pose in the robot-base frame [x, y, z, qw, qx, qy, qz, gripper_open] (vla-eval
    SimplerEnvBenchmark.make_obs, send_state=True; ManiSkill2_real2sim agent.eef_pos) -> the released LIBERO
    layout [x, y, z, axis-angle(3), gripper] that ProprioPhaseScheduler._decompose_state expects."""
    s = np.asarray(states, dtype=np.float64).reshape(-1)
    if s.size != 8:
        raise ValueError(f"expected SimplerEnv states [pos3, quat_wxyz4, gripper1], got shape {s.shape}")
    quat_xyzw = np.array([s[4], s[5], s[6], s[3]], dtype=np.float64)
    aa = _quat2axisangle(quat_xyzw.copy())
    return np.concatenate([s[:3], np.asarray(aa, dtype=np.float64), s[7:8]])


class QuatProprioPhaseScheduler(ProprioPhaseScheduler):
    """[c5] Same scheduler, state layout [pos3, quat4, gripper1] (SimplerEnv `states`): rotation = the 4-D quaternion,
    so the released `_rotation_distance` takes its geodesic branch 2*acos|<q1, q2>| (scheduler.py:240-253).
    Everything else (cues, windows, z/IQR normalization, beta map) is the released code unchanged. The released
    `_decompose_state` cannot reach the 4-D branch (it slices state[3:6]); this override is the only change and is
    the SETTING.md default for U7 (geodesic d_rot). Quaternion order (wxyz vs xyzw) does not affect |<q1, q2>|."""

    @staticmethod
    def _decompose_state(state):
        if state is None or state.size < 8:
            raise ValueError("QuatProprioPhaseScheduler expects [pos3, quat4, gripper1] states")
        return state[:3], state[3:7], float(state[7])


# ------------------------------------------------------------------------------------------- layer mixer [c2]
class StarLayerMixer:
    """TorchStarLayerController (controller_core_v2.py:12-64) with an optional position selector.

    step(h_in, h_out) is called once per decoder layer in order; reset() before each forward pass (the released
    wrapper resets at layer index 0, runtime.py:44-45). `positions`: None = all positions (released behaviour);
    "last" = only the last sequence position of the forward (OpenVLA action-emitting position)."""

    def __init__(self, config: StarConfig, total_layers: int, positions: str | None = None, shadow: bool = False,
                 record_activity: bool = True):
        self._shadow = bool(shadow)
        self._record_activity = bool(record_activity)   # [c7] logging only; a test-only toggle compares on vs off
        self._edit: list = []                            # per forward: list of (num, den) float32 device tensors
        if positions not in (None, "all", "last"):
            raise ValueError(f"positions must be None/'all'/'last', got {positions!r}")
        self._config = config
        self._total_layers = int(total_layers)
        self._correction_start_layer = config.resolve_correction_start(self._total_layers)
        self._beta_min, self._beta_max = config.resolve_beta_bounds()
        self._positions = None if positions in (None, "all") else positions
        mid = 0.5 * (self._beta_min + self._beta_max)
        self._step_ctx = {"c_t": 0.0, "v_t": 0.0, "j_t": 0.0, "w_t": 0.0, "u_t": 0.0, "r_t": 0.0,
                          "r_t_tilde": 0.0, "beta_t": mid, "g_t": mid}
        self.reset()

    @property
    def correction_start_layer(self) -> int:
        return self._correction_start_layer

    def set_step_context(self, ctx: dict) -> None:
        self._step_ctx = dict(ctx)

    def step_context(self) -> dict:
        """The step-level context (beta_t, ...) used for every forward of this request (server R4 check)."""
        return dict(self._step_ctx)

    def reset(self) -> None:
        self.layer_count = 0
        self.am = None

    def activity(self) -> dict | None:
        """[c7] review critic-ext-2 P9 (logging only): edit_ratio[f][j] = ||m^k - dh^k||_2 / ||dh^k||_2 over the hidden
        dimension at forward f's last (edited) position, for mixed layer k = correction_start + j; float32; None for
        a zero denominator. One device -> host move for the whole request."""
        if not self._record_activity or not self._edit:
            return None
        import torch

        flat = [t for fwd in self._edit for pair in fwd for t in pair]
        host = torch.stack(flat).reshape(-1, 2).cpu().tolist()          # the only device -> host transfer
        rows, i = [], 0
        for fwd in self._edit:
            row = []
            for _ in fwd:
                num, den = host[i]
                i += 1
                row.append(None if den == 0.0 else float(num / den))
            rows.append(row)
        return {"edit_ratio": rows,
                "edit_ratio_layers": list(range(self._correction_start_layer, self._total_layers + 1)),
                "edit_ratio_positions": "last", "edit_ratio_dtype": "float32"}

    def step(self, h_in, h_out):
        import torch

        self.layer_count += 1
        if self._positions == "last":
            full_in, full_out = h_in, h_out
            h_in, h_out = h_in[:, -1:, :], h_out[:, -1:, :]
        delta = h_out - h_in

        if self.layer_count < self._correction_start_layer:
            return full_out if self._positions == "last" else h_out

        beta_t = float(self._step_ctx.get("beta_t", self._step_ctx.get("g_t", self._beta_min)))
        beta_t = float(np.clip(beta_t, self._beta_min, self._beta_max))
        beta = torch.full((delta.shape[0], 1, 1), beta_t, dtype=delta.dtype, device=delta.device)
        am_prev = delta if self.am is None else self.am
        self.am = beta * am_prev + (1.0 - beta) * delta
        corrected = h_in + self.am
        if self._record_activity:           # [c7] out-of-place float32 norms of fresh tensors; nothing is modified
            if self.layer_count == self._correction_start_layer:
                self._edit.append([])
            diff_last = (self.am - delta)[:, -1, :].float()
            delta_last = delta[:, -1, :].float()
            num = torch.linalg.vector_norm(diff_last, dim=-1)[0]
            den = torch.linalg.vector_norm(delta_last, dim=-1)[0]
            self._edit[-1].append((num, den))
        if self._shadow:                    # [c6] compute, never write back
            return full_out if self._positions == "last" else h_out
        if self._positions == "last":
            return torch.cat([full_out[:, :-1, :], corrected], dim=1)
        return corrected
