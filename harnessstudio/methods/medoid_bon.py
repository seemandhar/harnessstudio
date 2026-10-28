"""medoid-bon: CycleVLA's Minimum Bayes Risk (MBR) consensus selection over K seeded samples.

Source: CycleVLA, arXiv:2601.02295v2 (28 Jul 2026), Sec. IV-B "Sampling Actions and Ranking via MBR"
and Sec. V-A (see CYCLEVLA_SOURCE.json). Rule as published:
  - each hypothesis -> trajectory by accumulating its end-effector translational and rotational
    deltas, a feature in R^(6H) (position x,y,z and orientation u,v,w at each step);
  - the gripper signal is excluded; no normalization;
  - d = L2 distance; select argmin_i of the mean distance to the other hypotheses; N = 8.
The MBR step is training-free. CycleVLA's progress-aware VLA and VLM failure predictor/backtracking
(trained or VLM-based) are NOT part of this method, hence fidelity 'component'.
The end-effector dims are resolved by NAME from the policy adapter's action layout
(extras['policy']['action_layout']): GR00T 5:8 + 8:11, Pi0.5 0:3 + 3:6. base_motion and
control_mode are not in the paper's action space and are excluded like the gripper.
v2 (2026-10-28): semantic keys instead of fixed GR00T dims; selection on GR00T is unchanged.
"""
from __future__ import annotations

import math

from ..core.capabilities import caps
from ..core.method import Fidelity, MethodConfig, ModuleRef

EEF_KEYS = ["action.end_effector_position", "action.end_effector_rotation"]
NUM_SAMPLES = 8


class SeededPool:
    def __init__(self, num_samples: int = NUM_SAMPLES):
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        self.num_samples = num_samples
        self.identity = {"module": "seeded_pool", "num_samples": num_samples}


class MBRSelect:
    def __init__(self, keys=None, dims=None, horizon_steps: int = 16):
        if dims is not None and (not dims or any(type(d) is not int or d < 0 for d in dims)):
            raise ValueError("dims must be nonnegative ints")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        self.keys = list(EEF_KEYS if keys is None else keys)
        self.dims = None if dims is None else list(dims)
        self.horizon = int(horizon_steps)
        self.identity = {"module": "cyclevla_mbr", "keys": self.keys, "dims": self.dims, "metric": "l2",
                         "feature": "cumulative end-effector deltas", "horizon_steps": self.horizon}

    def _resolve(self, extras):
        if self.dims is not None:
            return self.dims
        layout = (extras or {}).get("policy", {}).get("action_layout") or {}
        missing = [k for k in self.keys if k not in layout]
        if missing:
            raise ValueError(f"policy action layout lacks {missing}")
        return [d for k in self.keys for d in range(*layout[k])]

    def _feature(self, actions, dims) -> list:
        totals = [0.0] * len(dims)
        feature = []
        for row in list(actions)[:self.horizon]:
            for i, d in enumerate(dims):
                totals[i] += float(row[d])
            feature.extend(totals)
        return feature

    def select(self, chunks, extras, obs) -> int:
        dims = self._resolve(extras)
        features = [self._feature(c.actions, dims) for c in chunks]
        n = len(features)
        if n == 1:
            return 0
        risks = []
        for i in range(n):
            total = sum(math.sqrt(sum((a - b) ** 2 for a, b in zip(features[i], features[j])))
                        for j in range(n) if j != i)
            risks.append(total / (n - 1))
        return min(range(n), key=lambda i: (risks[i], i))


def config() -> MethodConfig:
    return MethodConfig(
        name="medoid-bon", version="2", paper="arXiv:2601.02295v2", fidelity=Fidelity.COMPONENT,
        substitutions=(), required=caps("action_chunk", "seeded_sampling", "isolated_streams"),
        slots={"proposer": ModuleRef("harnessstudio.methods.medoid_bon:SeededPool", {"num_samples": NUM_SAMPLES}),
               "selector": ModuleRef("harnessstudio.methods.medoid_bon:MBRSelect",
                                     {"keys": EEF_KEYS, "horizon_steps": 16})},
        description="CycleVLA MBR consensus selection (component): execute the sample with the lowest "
                    "mean L2 distance between accumulated end-effector trajectories.")
