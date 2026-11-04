"""Server-side STAR hook for EXT `openvla-residual-hook` (capability residual_hooks; ext/EXT_openvla-residual-hook.md).

Loaded by the policy server from a `sampler: {factory: "hs_star.server_hook:StarResidualHook", params: {...}}`
request (module prefix `hs_`), one object per (factory, params), one per-episode state per (session, episode,
sampler key), like SCALE's decoder state in openvla_hs.py. Protocol the extension's server calls:

    READS = ("states",)                      the only observation key the server passes (R2: a new dict)
    bind(total_layers)                       once, after loading (32 Llama-2 decoder layers for OpenVLA-7B)
    init_state() -> state                    at the first request of an episode (scheduler with empty history)
    begin(state, obs_view) -> (mixer, info)  per request: reads obs_view["states"]; computes beta_t
    next_state(state, info) -> state         after the request (scheduler.update_after_action)

The server wraps its unchanged native decode (the authors' no-mask greedy generate) in
`harnessstudio.policy_ext.residual_hooks.residual_hooks(model.language_model.model.layers, mixer)` and writes `info`
(beta_t, r_t, r_t_tilde, c/v/j/w, w_t_f64) into its pre-cast action log, one record per request (R1).
w_t is the released float32 geodesic (scheduler.py:240-253, resolution floor ~1e-3 rad per step, review R10);
w_t_f64 is the same angle in float64 (2*atan2(|v_rel|, |w_rel|)), LOGGED ONLY: it never enters beta.
STAR uses no observation key other than `states`, never reward/success/sim state/seed/case ids.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

import numpy as np

from .star_core import (StarConfig, ProprioPhaseScheduler, QuatProprioPhaseScheduler, StarLayerMixer,
                        TABLE7_OPENVLA_SIMPLER, last_n_from_rho, simpler_state_to_star)

# Paper / code defaults for OpenVLA on SimplerEnv (port_design.md section 5; QUESTIONS.md reconciles unknowns).
DEFAULTS = dict(
    TABLE7_OPENVLA_SIMPLER,          # beta (0.25, 0.55), w_c 1, w_v 1, w_j -0.25, w_w -0.75, kappa 4.75
    rho=2.0 / 3.0,                   # paper Eq. 2 -> 11 mixed layers of 32 (layers 22..32)
    control_hz=3.0,                  # SimplerEnv Google Robot control frequency; one inference per env step
    stats_window_sec=2.0,            # released default T_w -> W = round(3 * 2.0) = 6 steps
    stats_window_steps=0,
    norm_min_count=8,                # released default; clipped to W = 6 by the released scheduler
    positions="last",                # U3/U4: action-emitting positions only (alternative "all")
    state_layout="quat_geodesic",    # U7 (SETTING.md): [pos3, quat_wxyz4, g] -> released 4-D geodesic branch;
                                     # alternative "axisangle": convert to the released LIBERO layout (Euclidean)
    shadow=False,                    # True = overhead/shadow null: compute everything, write nothing back
)
LAYOUTS = ("quat_geodesic", "axisangle")

STATE_KEY = "states"
READS = (STATE_KEY,)


def _geodesic_f64(q1, q2) -> float:
    """Rotation angle between unit quaternions (wxyz or xyzw alike), float64 and well-conditioned at small angles."""
    a = np.asarray(q1, dtype=np.float64); b = np.asarray(q2, dtype=np.float64)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    rel = b - d * a                       # component orthogonal to a; |rel| = sin(theta/2), |d| = cos(theta/2)
    return float(2.0 * math.atan2(float(np.linalg.norm(rel)), abs(d)))


@dataclasses.dataclass
class _EpisodeState:
    scheduler: ProprioPhaseScheduler
    requests: int = 0
    prev_quat: Any = None
    pending_quat: Any = None


class StarResidualHook:
    READS = READS

    def __init__(self, **params: Any) -> None:
        p = dict(DEFAULTS)
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown STAR params {sorted(unknown)}")
        p.update(params)
        if p["state_layout"] not in LAYOUTS:
            raise ValueError(f"unsupported state_layout {p['state_layout']!r}; one of {LAYOUTS}")
        self.params = p
        self._total_layers: int | None = None
        self._config: StarConfig | None = None

    def bind(self, total_layers: int) -> None:
        p = self.params
        n_mix = last_n_from_rho(int(total_layers), float(p["rho"]))
        self._total_layers = int(total_layers)
        self._config = StarConfig(
            beta_min=p["beta_min"], beta_max=p["beta_max"], beta_kappa=p["beta_kappa"],
            correction_last_n_layers=n_mix, control_hz=p["control_hz"], stats_window_sec=p["stats_window_sec"],
            stats_window_steps=p["stats_window_steps"], norm_min_count=p["norm_min_count"],
            phase_coeff_c=p["phase_coeff_c"], phase_coeff_v=p["phase_coeff_v"], phase_coeff_j=p["phase_coeff_j"],
            phase_coeff_w=p["phase_coeff_w"])

    @property
    def config(self) -> StarConfig:
        if self._config is None:
            raise RuntimeError("StarResidualHook.bind(total_layers) was not called")
        return self._config

    def init_state(self) -> _EpisodeState:
        cls = QuatProprioPhaseScheduler if self.params["state_layout"] == "quat_geodesic" else ProprioPhaseScheduler
        return _EpisodeState(scheduler=cls(self.config))

    def begin(self, state: _EpisodeState, obs: Mapping[str, Any]):
        """`obs` is the server's view: a new dict with only READS keys."""
        if STATE_KEY not in obs or obs[STATE_KEY] is None:
            raise KeyError("STAR needs the benchmark's proprioceptive 'states' (send_state: true); refusing to "
                           "run unfired (P-20261103-14)")
        raw = np.asarray(obs[STATE_KEY], dtype=np.float64).reshape(-1)
        if raw.size != 8:
            raise ValueError(f"STAR expects 8-D SimplerEnv states [pos3, quat_wxyz4, gripper1], got {raw.shape}")
        star_state = raw if self.params["state_layout"] == "quat_geodesic" else simpler_state_to_star(raw)
        ctx = state.scheduler.prepare_context({"observation/state": star_state})
        mixer = StarLayerMixer(self.config, self._total_layers, positions=self.params["positions"],
                               shadow=bool(self.params["shadow"]))
        mixer.set_step_context(ctx)
        info = dict(ctx)
        quat = raw[3:7].copy()
        info.update(request=state.requests, correction_start_layer=mixer.correction_start_layer,
                    shadow=bool(self.params["shadow"]),
                    w_t_f64=0.0 if state.prev_quat is None else _geodesic_f64(quat, state.prev_quat))
        state.pending_quat = quat
        return mixer, info

    def next_state(self, state: _EpisodeState, info: Mapping[str, Any]) -> _EpisodeState:
        state.scheduler.update_after_action()
        state.prev_quat, state.pending_quat = state.pending_quat, None
        state.requests += 1
        return state
