"""medoid-bon: CycleVLA's Minimum Bayes Risk (MBR) consensus selection over K seeded samples.

Source: CycleVLA, arXiv:2601.02295v2 (28 Jul 2026), Sec. IV-B "Sampling Actions and Ranking via MBR"
and Sec. V-A (see CYCLEVLA_SOURCE.json). Rule as published:
  - each hypothesis -> trajectory by accumulating its end-effector translational and rotational
    deltas, a feature in R^(6H) (position x,y,z and orientation u,v,w at each step);
  - the gripper signal is excluded; no normalization;
  - d = L2 distance; select argmin_i of the mean distance to the other hypotheses; N = 8.
The MBR step is training-free. CycleVLA's progress-aware VLA and VLM failure predictor/backtracking
(trained or VLM-based) are NOT part of this method, hence fidelity 'component'.

v2 (2026-10-28): semantic keys instead of fixed GR00T dims; selection on GR00T is unchanged.
v3 (HarnessStudio v2 port): the end-effector dims are resolved by NAME from the public action layout
the framework passes to bind(embodiment) (vla-eval DimSpec names "position" and "rotation", in
action-vector order, e.g. RoboCasa365 GR00T N1.5 0:3 + 3:6, the fake stack 0:3 + 3:6). base_motion,
control_mode and gripper are not in the paper's feature and are excluded. An explicit dims= list
overrides the layout. Chunks arrive as arrays [H, A] instead of objects with .actions. The algorithm,
constants and arithmetic (pure-Python float64 accumulation, tie-break on the lower index) are v2's.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

EEF_KEYS = ["position", "rotation"]   # vla-eval DimSpec names (v2 used "action.end_effector_position/rotation")
NUM_SAMPLES = 8
HORIZON_STEPS = 16


class SeededPool:
    """Proposer: K seeded samples per decision (sample 0 is the native stream)."""

    def __init__(self, num_samples: int = NUM_SAMPLES):
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        self.num_samples = num_samples
        self.identity = {"module": "seeded_pool", "num_samples": num_samples}


class MBRSelect:
    """Selector: index of the sample with the lowest mean L2 distance between accumulated
    end-effector trajectories (the MBR medoid). Never averages samples."""

    def __init__(self, keys: Optional[Sequence[str]] = None, dims: Optional[Sequence[int]] = None,
                 horizon_steps: int = HORIZON_STEPS):
        if dims is not None and (not dims or any(type(d) is not int or d < 0 for d in dims)):
            raise ValueError("dims must be nonnegative ints")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        self.keys = list(EEF_KEYS if keys is None else keys)
        self.dims = None if dims is None else list(dims)
        self.horizon = int(horizon_steps)
        self.layout: dict = {}
        self.identity = {"module": "cyclevla_mbr", "keys": self.keys, "dims": self.dims, "metric": "l2",
                         "feature": "cumulative end-effector deltas", "horizon_steps": self.horizon}

    def bind(self, embodiment) -> None:
        """Public action-space facts (Embodiment.action_layout: name -> (start, end))."""
        self.layout = dict(embodiment.action_layout or {})

    def _resolve(self) -> list:
        if self.dims is not None:
            return self.dims
        missing = [k for k in self.keys if k not in self.layout]
        if missing:
            raise ValueError(f"policy action layout lacks {missing} (has {sorted(self.layout)}); "
                             f"pass dims= to select the end-effector dims explicitly")
        return [d for k in self.keys for d in range(*self.layout[k])]

    def _feature(self, actions, dims) -> list:
        totals = [0.0] * len(dims)
        feature = []
        for row in list(actions)[:self.horizon]:
            for i, d in enumerate(dims):
                totals[i] += float(row[d])
            feature.extend(totals)
        return feature

    def select(self, chunks, extras, obs) -> int:
        dims = self._resolve()
        features = [self._feature(c, dims) for c in chunks]
        n = len(features)
        if n == 1:
            return 0
        risks = []
        for i in range(n):
            total = sum(math.sqrt(sum((a - b) ** 2 for a, b in zip(features[i], features[j])))
                        for j in range(n) if j != i)
            risks.append(total / (n - 1))
        return min(range(n), key=lambda i: (risks[i], i))


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "proposer": ModuleRef("hs_medoid_bon:SeededPool", {"num_samples": NUM_SAMPLES}),
        "selector": ModuleRef("hs_medoid_bon:MBRSelect", {"keys": EEF_KEYS, "horizon_steps": HORIZON_STEPS}),
    })
