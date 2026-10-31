"""ChunkTrust adaptive horizon selection (AHS) as HarnessStudio v2 slot components.

Source: ChunkTrust, arXiv:2609.39754v2; code hf618/ChunkTrust@a9fd57412b3a19a0728cd726b0558dd89395fb84
src/chunktrust/gr00t_ahs.py. The numerics are vendored byte for byte (chunktrust_gr00t_ahs.py) and so is
v1's episode-state wrapper (horizon.py: Beta posterior, rng(ts_seed + 0), executed-action history), with
the hashes in PROVENANCE.json. Neither file is edited; this module only adapts them to the v2 slots.

Per decision (one native sample, K = 1):
  proposer  VelocityReadoutProposer: num_samples 1, readouts ("velocity",). The DecisionLoop asks the
            policy server for the flow-velocity readout with the sample, so extras["velocity"] holds
            [denoise steps, H, A_model] float32 (GR00T N1.5: [4, 16, 32], the padded model action space).
  executor  AHSExecutor: k = ChunkTrustAHS.select(velocity, actions) over the candidates {4, 8, 12, 16}
            (spectral high-frequency energy of the velocity trace per candidate prefix, window-RMS over
            the denoising steps, mixed 0.5/0.5 with the speed uniformity of history + prefix, Beta
            posterior with kernel forgetting, expected-round). The executed prefix is appended to the
            history (record_executed).

v1 -> v2 port (HarnessStudio v1 harnessstudio/methods/chunktrust_ahs/modules.py):
  - the velocity comes from extras["velocity"] (v2 policy readout) instead of extras["sampler"]["velocity"]
    (v1 server-side VelocityReadout sampler). Same tensor: the action decoder's pred[:, -H:] at each
    denoising step, all 32 padded dims (velocity_dims="all", as v1 and the upstream overlay).
  - the chunk is an array [H, A] instead of an object with .actions.
  - column order: v1's flat GR00T rows were the sorted GR00T keys (base_motion 4, control_mode 1,
    end_effector_position 3, end_effector_rotation 3, gripper_close 1); the v2 wire order is vla-eval's
    (position, rotation, gripper, base_motion, control_mode). The speed-uniformity term uses row norms of
    action differences, which do not depend on the column order mathematically but do in float32
    summation order, so the executor permutes the columns back to v1's order when the bound action
    layout has all five GR00T groups. Otherwise (e.g. the 7-D fake stack) the native order is used.
  - diagnostics() (new; v1's executor had none): a compact view of the last decision for the v2 trace.
"""
from __future__ import annotations

import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import chunktrust_gr00t_ahs as upstream

# horizon.py (vendored, byte-identical) imports the upstream numerics by top-level name. Alias the
# vendored module for that import only, so horizon binds this package's copy and no global name leaks.
_previous = sys.modules.get("chunktrust_gr00t_ahs")
sys.modules["chunktrust_gr00t_ahs"] = upstream
try:
    from . import horizon
finally:
    if _previous is None:
        del sys.modules["chunktrust_gr00t_ahs"]
    else:
        sys.modules["chunktrust_gr00t_ahs"] = _previous
if horizon.select_exec_k_horizon is not upstream.select_exec_k_horizon:  # pragma: no cover
    raise ImportError("horizon.py must use the vendored chunktrust_gr00t_ahs numerics")

# Identical to v1 AHS_PARAMS (harnessstudio/methods/chunktrust_ahs/__init__.py) and to
# runs/chunktrust-beat-20261026/harnesses/chunktrust_ahs/config/ahs_reference.json (= horizon.REFERENCE_AHS_PARAMS).
AHS_PARAMS = {
    "exec_mode": "expected_round", "expected_temp": 1.0, "horizon_alpha": 4.0,
    "horizon_candidates": [4, 8, 12, 16], "intra_cut_t": 0.25, "signal_mode": "both",
    "speed_beta": 4.0, "speed_fallback": "ehh_only", "speed_weight": 0.5, "speed_window": 20,
    "tau_window": 3, "ts_epsilon": 0.05, "ts_forget_rho": 0.99, "ts_kernel_bandwidth": 10.0,
    "ts_seed": 0, "ts_update_eta": 1.0, "ts_update_mode": "kernel_forget",
    "use_speed_uniformity": True, "z_mode": "window_rms"}
AHS_PARAMS_SHA256 = horizon.canonical_sha256(AHS_PARAMS)
MIN_K = min(AHS_PARAMS["horizon_candidates"])

# v1 flat column order (sorted GR00T keys, harnessstudio v1 policies/gr00t_n15.py ACTION_LAYOUT), as
# role -> accepted action-layout names: vla-eval DimSpec names first, then the GR00T key names.
V1_COLUMN_ORDER = (
    ("base_motion", ("base_motion", "action.base_motion")),
    ("control_mode", ("control_mode", "action.control_mode")),
    ("position", ("position", "action.end_effector_position")),
    ("rotation", ("rotation", "action.end_effector_rotation")),
    ("gripper", ("gripper", "action.gripper_close")),
)

EXECUTOR_PARAMS = {"min_k": MIN_K, "per_step": False, "ahs_params": AHS_PARAMS, "velocity_dims": "all",
                   "column_order": "v1"}


class VelocityReadoutProposer:
    """Proposer slot: one native sample per decision with the policy's flow-velocity readout.
    Replaces v1's server-side VelocityReadout sampler (which recorded v_t and never replaced it)."""

    def __init__(self, num_samples: int = 1, readouts: Sequence[str] = ("velocity",)) -> None:
        if num_samples != 1 or type(num_samples) is not int:
            raise ValueError("chunktrust-ahs runs on the one native sample: num_samples must be 1")
        if list(readouts) != ["velocity"]:
            raise ValueError("chunktrust-ahs needs exactly the velocity readout")
        self.num_samples = 1
        self.readouts = ("velocity",)
        self.identity = {"module": "velocity_readout_proposer", "num_samples": 1, "readouts": ["velocity"]}


class AHSExecutor:
    """Executor slot: k = ChunkTrust AHS over the velocity readout of the executed (native) sample."""

    per_step = False

    def __init__(self, min_k: int = MIN_K, per_step: bool = False, ahs_params: Optional[Mapping[str, Any]] = None,
                 velocity_dims: str = "all", column_order: str = "v1") -> None:
        ahs_params = dict(AHS_PARAMS if ahs_params is None else ahs_params)
        # v1 AHSExecutor checks, unchanged
        if per_step is not False:
            raise ValueError("AHS decides k up front; per_step must be False")
        if min_k != min(ahs_params["horizon_candidates"]):
            raise ValueError("min_k must equal min(horizon_candidates)")
        if column_order not in ("v1", "native"):
            raise ValueError("column_order must be 'v1' (v1/upstream GR00T flat order when available) or 'native'")
        self.ahs = horizon.ChunkTrustAHS(ahs_params, velocity_dims)
        self.min_k, self.per_step = min_k, per_step
        self.column_order = column_order
        self.identity = {"module": "chunktrust_ahs", "velocity_dims": velocity_dims,
                         "ahs_params_sha256": horizon.canonical_sha256(ahs_params), "column_order": column_order}
        self.ahs.start_episode()
        self._perm: Optional[np.ndarray] = None
        self.last_info: Optional[dict] = None
        self._last: dict = {}

    # -- action layout ---------------------------------------------------------------
    def bind(self, embodiment) -> None:
        """Public action layout (name -> (start, end), vector order). Picks v1's column order if possible."""
        layout = {str(k): (int(v[0]), int(v[1])) for k, v in dict(embodiment.action_layout or {}).items()}
        self._perm = None
        if self.column_order != "v1":
            return
        cols = []
        for _, names in V1_COLUMN_ORDER:
            name = next((n for n in names if n in layout), None)
            if name is None:
                return  # not the GR00T panda_omron layout: keep the native order
            cols.extend(range(*layout[name]))
        width = embodiment.action_dim or max(e for _, e in layout.values())
        if sorted(cols) != list(range(int(width))):
            return
        self._perm = np.asarray(cols, dtype=np.intp)

    def _flat(self, actions) -> np.ndarray:
        arr = np.asarray(actions)
        if self._perm is not None:
            if arr.ndim != 2 or arr.shape[1] != len(self._perm):
                raise ValueError(f"chunktrust-ahs: actions {arr.shape} do not match the bound layout")
            arr = arr[:, self._perm]
        # C order, as v1's flats: arr[:, perm] is not C-contiguous, and a strided history makes the float32
        # row norms of the speed-uniformity term sum in a different order (ulp differences from v1).
        return np.ascontiguousarray(arr)

    # -- executor interface ----------------------------------------------------------------
    def plan(self, chunk, extras, obs) -> int:
        velocity = extras.get("velocity")
        if velocity is None:
            raise ValueError("chunktrust-ahs needs extras['velocity'] (proposer readouts=['velocity'] and a "
                             "policy with velocity_readout)")
        k, self.last_info = self.ahs.select(velocity=velocity, actions_flat=self._flat(chunk))
        info = self.last_info
        self._last = {
            "k": int(k), "column_order": "v1" if self._perm is not None else "native",
            "velocity_shape": list(np.shape(velocity)), "chunk_shape": list(np.shape(chunk)),
            **{key: info.get(key) for key in ("choose_mode", "strides", "expected_stride", "z_intra", "q_intra",
                                              "p_inter", "q_mix", "theta_sample", "posterior_alpha_after",
                                              "posterior_beta_after", "update_feedback")}}
        return int(k)

    def continue_after(self, executed, obs) -> bool:
        return True

    def record_executed(self, executed_actions) -> None:
        # v1: self.ahs.record_executed(np.asarray(executed_actions, dtype=np.float32)), in v1's column order
        self.ahs.record_executed(self._flat(np.asarray(executed_actions, dtype=np.float32)))

    def diagnostics(self) -> dict:
        """Compact JSON view of the last decision (v1 traced nothing for this executor)."""
        return dict(self._last)
