"""chunk-interrupt (tf_variant of VLA-Corrector truncation; spec §7).

Commanded translation: unnormalized action.end_effector_position (OSC_POSE delta, controller
input units) x output_max/input_max = 0.05 m per unit (robosuite default_pandaomron.json,
input_ref_frame 'base'). Achieved translation: state.end_effector_position_relative (base frame).
No improvement is claimed: the ChunkTrust H1 stall detector did not predict failure
(P(fail|fired) 0.55 vs base 0.53; runs/chunktrust-beat-20261026/reports/final_report.md).
"""
from __future__ import annotations

import math

from ..core.capabilities import caps
from ..core.method import Fidelity, MethodConfig, ModuleRef

CONTROLLER_SCALE_M = 0.05   # output_max[0:3] / input_max of the pinned controller config below
CONTROLLER_CONFIG_REL = "controllers/config/robots/default_pandaomron.json"  # inside the robosuite package
CONTROLLER_CONFIG_SHA256 = "4f91b12b904e8907005393e45dbb5576af470f7f3d16da017de3826fa22e8dcb"


def controller_config_path():
    """The pinned robosuite panda_omron controller config: $HARNESSSTUDIO_ROBOSUITE_ROOT (the robosuite
    package directory) or the importable robosuite package; None if neither is available."""
    import os
    from pathlib import Path
    root = os.environ.get("HARNESSSTUDIO_ROBOSUITE_ROOT")
    if not root:
        try:
            import robosuite
            root = str(Path(robosuite.__file__).resolve().parent)
        except ImportError:
            return None
    path = Path(root) / CONTROLLER_CONFIG_REL
    return path if path.is_file() else None
# Real-data check (2026-10-27, runs/harnessstudio-verify/chunk-interrupt/units-check-20261027, identity
# baseline, OpenDrawer--0002, 750 steps): per-axis correlation of achieved base-frame EEF motion with
# commanded delta x 0.05 m = 0.83 / 0.92 / 0.70; free-motion achieved/commanded ratio ~0.14 per step.
# rho stays 0.05 (frozen, not tuned on this data).
POSITION_KEY = "action.end_effector_position"  # resolved from the policy adapter's action layout
# v2 (2026-10-28): action slice and proprio location come from extras['policy'] (GR00T: 5:8 and
# state.end_effector_position_relative; Pi0.5: 0:3 and observation/state[0:3]); GR00T behaviour unchanged.


def _vec3(value):
    flat = value.tolist() if hasattr(value, "tolist") else value
    while isinstance(flat, (list, tuple)) and flat and isinstance(flat[0], (list, tuple)):
        flat = flat[-1]
    return [float(v) for v in flat[-3:]]


class StallCut:
    def __init__(self, min_k=4, per_step=True, window=4, rho=0.05, scale_m=CONTROLLER_SCALE_M,
                 position_key=POSITION_KEY):
        if not per_step or window < 1 or min_k < window or not 0 < rho < 1:
            raise ValueError("StallCut needs per_step=True, window >= 1, min_k >= window, 0 < rho < 1")
        self.min_k, self.per_step, self.window, self.rho = min_k, per_step, window, rho
        self.scale, self.position_key = float(scale_m), position_key
        self.identity = {"module": "stall_cut", "window": window, "rho": rho, "scale_m": self.scale,
                         "position_key": position_key}
        self._rows, self._pos, self._low = [], [], 0
        self.slice, self.key, self.proprio_slice = None, None, None

    def _read(self, obs):
        value = obs.values[self.key]
        if self.proprio_slice is None:
            return _vec3(value)
        flat = value.tolist() if hasattr(value, "tolist") else value
        while isinstance(flat, (list, tuple)) and flat and isinstance(flat[0], (list, tuple)):
            flat = flat[-1]
        return [float(v) for v in flat[self.proprio_slice[0]:self.proprio_slice[1]]]

    def plan(self, chunk, extras, obs) -> int:
        policy = (extras or {}).get("policy", {})
        layout, proprio = policy.get("action_layout") or {}, policy.get("proprio_eef")
        if self.position_key not in layout or not proprio:
            raise ValueError("policy adapter must declare the end-effector action slice and proprio location")
        self.slice = tuple(layout[self.position_key])
        self.key, self.proprio_slice = proprio[0], (tuple(proprio[1]) if proprio[1] else None)
        self._rows = [list(map(float, r)) for r in chunk.actions]
        self._pos = [self._read(obs)]
        self._low = 0
        return len(self._rows)

    def continue_after(self, executed, obs) -> bool:
        self._pos.append(self._read(obs))
        if executed < self.window:
            return True
        a, b = self.slice
        commanded = [sum(r[a + j] for r in self._rows[executed - self.window:executed]) * self.scale for j in range(3)]
        achieved = [p - q for p, q in zip(self._pos[executed], self._pos[executed - self.window])]
        c_norm = math.sqrt(sum(v * v for v in commanded))
        ratio = 1.0 if c_norm < 1e-6 else math.sqrt(sum(v * v for v in achieved)) / c_norm
        self._low = self._low + 1 if ratio < self.rho else 0
        return self._low < self.window

    def record_executed(self, executed_actions) -> None:
        pass


def config() -> MethodConfig:
    return MethodConfig(
        name="chunk-interrupt", version="2", paper="arXiv:2607.01804v1", fidelity=Fidelity.TF_VARIANT,
        substitutions=("VLA-Corrector trained latent-dynamics corrector -> public commanded-vs-achieved "
                       "end-effector displacement check; truncation only, no gradient recovery",),
        required=caps("action_chunk", "seeded_sampling", "proprio", "variable_execute"),
        slots={"executor": ModuleRef("harnessstudio.methods.chunk_interrupt:StallCut",
                                     {"min_k": 4, "per_step": True, "window": 4, "rho": 0.05})},
        description="Cut a chunk and re-query when the arm stops following its commands.")
