"""Executor slot adapters over harnessstudio.runtime.modules."""
from __future__ import annotations

from ..runtime.contracts import PublicObservation
from ..runtime.modules import ProgressMonitor


class PublicStasis:
    """Unified's public-proprioception stasis monitor as a per-step Executor.

    cut=False (default) keeps Unified's semantics: advisory only, the signal is recorded and the
    chunk always continues. cut=True makes the stasis signal cut the chunk and re-query; that is a
    new hypothesis, not a port, and must be reviewed as such. A stationary robot may be correctly
    holding an object; this is never a success or failure detector.
    """
    def __init__(self, key="state.end_effector_position_relative", window=4, tolerance=1e-4,
                 min_k=4, per_step=True, cut=False):
        if per_step is not True or type(min_k) is not int or min_k < 1:
            raise ValueError("PublicStasis is a per-step executor with integer min_k >= 1")
        self.monitor = ProgressMonitor(key, window, tolerance)
        self.key, self.min_k, self.per_step, self.cut = key, min_k, per_step, bool(cut)
        self.identity = dict(self.monitor.identity, adapter="harnessstudio.executor", cut=self.cut, min_k=min_k)
        self.signals = []

    def _flat(self, obs):
        value = obs.values[self.key]
        flat = value.tolist() if hasattr(value, "tolist") else value
        while isinstance(flat, (list, tuple)) and flat and isinstance(flat[0], (list, tuple)):
            flat = flat[-1]
        return PublicObservation.from_mapping({self.key: [float(v) for v in flat]}, (self.key,))

    def plan(self, chunk, extras, obs) -> int:
        self.monitor.reset()
        self.monitor.observe(self._flat(obs))
        return len(chunk.actions)

    def continue_after(self, executed, obs) -> bool:
        signal = self.monitor.observe(self._flat(obs))
        self.signals.append(signal["signal"])
        return not (self.cut and signal["signal"] == "public_stasis")

    def record_executed(self, executed_actions) -> None:
        pass
