"""chunk-interrupt (tf_variant of VLA-Corrector truncation; spec §7).

Commanded translation: unnormalized action.end_effector_position (OSC_POSE delta, controller
input units) x output_max/input_max = 0.05 m per unit (robosuite default_pandaomron.json,
input_ref_frame 'base'). Achieved translation: state.end_effector_position_relative (base frame).
No improvement is claimed: the ChunkTrust H1 stall detector did not predict failure
(P(fail|fired) 0.55 vs base 0.53; runs/chunktrust-beat-20261026/reports/final_report.md).

v3 (HarnessStudio v2 port): the commanded slice is the "position" block of the public action layout
(bind(embodiment); vla-eval DimSpec names, e.g. RoboCasa365 GR00T N1.5 0:3), and the achieved
position is embodiment.eef_position(obs) on the public observation the DecisionLoop passes to plan()
and continue_after() (RoboCasa365 state["state.end_effector_position_relative"], LIBERO states[0:3],
fake state[0:2]). If either is unavailable the executor never cuts (the chunk runs natively) and
says so in self.diagnostics. Algorithm, window, rho and the 0.05 m scale are v2's.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

CONTROLLER_SCALE_M = 0.05   # output_max[0:3] / input_max of the pinned controller config below
CONTROLLER_CONFIG_REL = "controllers/config/robots/default_pandaomron.json"  # inside the robosuite package
CONTROLLER_CONFIG_SHA256 = "4f91b12b904e8907005393e45dbb5576af470f7f3d16da017de3826fa22e8dcb"


def controller_config_path():
    """The pinned robosuite panda_omron controller config: $HARNESSSTUDIO_ROBOSUITE_ROOT (the robosuite
    package directory) or the importable robosuite package; None if neither is available."""
    import os
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
POSITION_KEY = "position"  # vla-eval DimSpec name in Embodiment.action_layout (v2: "action.end_effector_position")


class StallCut:
    """Per-step executor: cut the chunk (and re-query) after `window` consecutive steps in which the
    achieved end-effector displacement over the last `window` steps is below rho x the commanded one."""
    sees_proposals = False

    def __init__(self, min_k=4, per_step=True, window=4, rho=0.05, scale_m=CONTROLLER_SCALE_M,
                 position_key=POSITION_KEY, slice: Optional[Sequence[int]] = None):
        if not per_step or window < 1 or min_k < window or not 0 < rho < 1:
            raise ValueError("StallCut needs per_step=True, window >= 1, min_k >= window, 0 < rho < 1")
        if slice is not None and (len(slice) != 2 or any(type(v) is not int for v in slice)
                                  or not 0 <= slice[0] < slice[1]):
            raise ValueError("slice must be [start, end) of the commanded position block")
        self.min_k, self.per_step, self.window, self.rho = min_k, per_step, window, rho
        self.scale, self.position_key = float(scale_m), position_key
        self._slice_override = None if slice is None else (int(slice[0]), int(slice[1]))
        self.identity = {"module": "stall_cut", "window": window, "rho": rho, "scale_m": self.scale,
                         "position_key": position_key, "slice": self._slice_override}
        self._rows, self._pos, self._low = [], [], 0
        self._active = False
        self.slice = self._slice_override
        self.embodiment = None
        # Not traced by the framework (slot components have no trace hook); inspect in tests/debugging.
        self.diagnostics = {"eef_available": None, "slice": self.slice, "decisions": 0,
                            "native_decisions": 0, "cut_requests": 0, "reason": None}

    def bind(self, embodiment) -> None:
        """Public action-space facts: Embodiment.action_layout (name -> (start, end)) and eef_position()."""
        self.embodiment = embodiment
        if self._slice_override is None:
            layout = dict(embodiment.action_layout or {})
            self.slice = tuple(layout[self.position_key]) if self.position_key in layout else None
        self.diagnostics["slice"] = self.slice

    def _read(self, obs) -> Optional[list]:
        if self.embodiment is None or obs is None:
            return None
        value = self.embodiment.eef_position(obs)
        if value is None or np.asarray(value).size == 0:
            return None
        return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]

    def plan(self, chunk, extras, obs) -> int:
        self._rows = [list(map(float, r)) for r in np.asarray(chunk)]
        first = self._read(obs)
        self._active = self.slice is not None and first is not None
        self._pos = [first]
        self._low = 0
        d = self.diagnostics
        d["decisions"] += 1
        d["eef_available"] = first is not None
        if not self._active:
            d["native_decisions"] += 1
            d["reason"] = ("no commanded position block in the action layout" if self.slice is None
                           else "benchmark exposes no end-effector position") + ": never cuts (native execution)"
        return len(self._rows)

    def continue_after(self, executed, obs) -> bool:
        if not self._active:
            return True
        current = self._read(obs)
        if current is None:  # eef vanished mid-chunk: stop judging, run the rest natively
            self._active = False
            self.diagnostics["reason"] = "end-effector position disappeared mid-chunk: no cut"
            return True
        self._pos.append(current)
        if executed < self.window:
            return True
        a, b = self.slice
        n = min(b - a, len(current), len(self._pos[executed - self.window]))  # 3 on LIBERO/RoboCasa365, 2 on fake
        commanded = [sum(r[a + j] for r in self._rows[executed - self.window:executed]) * self.scale for j in range(n)]
        achieved = [p - q for p, q in zip(self._pos[executed][:n], self._pos[executed - self.window][:n])]
        c_norm = math.sqrt(sum(v * v for v in commanded))
        ratio = 1.0 if c_norm < 1e-6 else math.sqrt(sum(v * v for v in achieved)) / c_norm
        self._low = self._low + 1 if ratio < self.rho else 0
        wants = self._low < self.window
        if not wants:
            self.diagnostics["cut_requests"] += 1
        # v1's pipeline honoured a cut only once executed >= min_k; v2's DecisionLoop does not, so the
        # executor applies that rule itself (with the defaults the earliest cut is step 7 >= 4 anyway).
        return wants or executed < self.min_k

    def record_executed(self, executed_actions) -> None:
        pass


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "executor": ModuleRef("hs_chunk_interrupt:StallCut", {"min_k": 4, "per_step": True, "window": 4, "rho": 0.05}),
    })
