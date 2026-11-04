"""Toy residual hook for EXT openvla-residual-hook (tests only; never an installed method).

Protocol of capability `residual_hooks` (policies/openvla_simpler_star_hs.py): READS, bind, init_state, begin,
next_state; the mixer scales every layer update from `start` on by `gain` (gain 1.0 = native arithmetic h_in + delta).
`seen` records exactly what the server handed to begin() (review critic-ext-1 R2 test). Loaded as
`hs_ext_residual_toy:ToyHook` with tests/toys on sys.path.
"""
from __future__ import annotations

import numpy as np


class ToyMixer:
    def __init__(self, total_layers: int, start: int, gain: float, shadow: bool = False):
        self.total_layers, self.correction_start_layer, self.gain, self.shadow = total_layers, start, gain, shadow
        self.layer_count = 0
        self._step_ctx = {"gain": gain}

    def step_context(self):
        return dict(self._step_ctx)

    def reset(self):
        self.layer_count = 0

    def step(self, h_in, h_out):
        self.layer_count += 1
        if self.layer_count < self.correction_start_layer:
            return h_out
        corrected = h_in + self.gain * (h_out - h_in)
        return h_out if self.shadow else corrected


class ToyHook:
    READS = ("states",)

    def __init__(self, start_from_last: int = 3, gain: float = 0.5, shadow: bool = False, reads=None):
        self.n, self.gain, self.shadow = int(start_from_last), float(gain), bool(shadow)
        if reads is not None:
            self.READS = tuple(reads)
        self.total = None
        self.seen = []

    def bind(self, total_layers: int):
        self.total = int(total_layers)

    def init_state(self):
        return {"requests": 0}

    def begin(self, state, obs_view):
        self.seen.append({k: np.array(v, copy=True) for k, v in obs_view.items()})
        mixer = ToyMixer(self.total, self.total - self.n + 1, self.gain, self.shadow)
        return mixer, {"request": state["requests"], "gain": self.gain, "keys": sorted(obs_view)}

    def next_state(self, state, info):
        state["requests"] += 1
        return state

