"""Wrappers around the vendored ChunkTrust-AHS numerics (byte-identical copies)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # horizon.py imports chunktrust_gr00t_ahs by top-level name


class VelocityReadout:
    """Server-side Sampler: records each step's velocity, never replaces it."""
    def __init__(self):
        self.identity = {"module": "velocity_readout"}
        self._steps = []

    def on_step(self, step, x_t, v_t, ctx):
        self._steps.append(v_t.detach().clone() if hasattr(v_t, "detach") else v_t)
        return None

    def extras(self) -> dict:
        import numpy as np
        if not self._steps:
            return {}
        first = self._steps[0]
        if hasattr(first, "detach"):
            import torch
            # [B, T, H, D] -> batch element 0, float32 [T, H, D] (as VelocityTraceRecorder.take()).
            return {"velocity": torch.stack(self._steps, dim=1)[0].float().cpu().numpy()}
        return {"velocity": np.stack(self._steps).astype(np.float32)}


class AHSExecutor:
    def __init__(self, min_k, per_step, ahs_params, velocity_dims="all"):
        import horizon
        if per_step is not False:
            raise ValueError("AHS decides k up front; per_step must be False")
        if min_k != min(ahs_params["horizon_candidates"]):
            raise ValueError("min_k must equal min(horizon_candidates)")
        self.ahs = horizon.ChunkTrustAHS(ahs_params, velocity_dims)
        self.min_k, self.per_step = min_k, per_step
        self.identity = {"module": "chunktrust_ahs", "velocity_dims": velocity_dims,
                         "ahs_params_sha256": horizon.canonical_sha256(ahs_params)}
        self.ahs.start_episode()
        self.last_info = None

    def plan(self, chunk, extras, obs) -> int:
        k, self.last_info = self.ahs.select(velocity=extras["sampler"]["velocity"], actions_flat=chunk.actions)
        return int(k)

    def continue_after(self, executed, obs) -> bool:
        return True

    def record_executed(self, executed_actions) -> None:
        import numpy as np
        self.ahs.record_executed(np.asarray(executed_actions, dtype=np.float32))
