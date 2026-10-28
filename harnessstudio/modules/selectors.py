"""Selector slot adapters over harnessstudio.runtime.modules."""
from __future__ import annotations

from ..runtime.modules import MedoidSelector


class TrajectoryMedoid:
    """Unified's scale-explicit trajectory medoid as a HarnessStudio Selector.

    Selects an existing sample (never averages). Per-dimension scales must be fixed positive
    numbers declared before validation (no fitting). A generic ablation, not an A3/MG-Select
    reproduction (same caveat as Unified).
    """
    def __init__(self, scales, horizon_steps: int = 16):
        self._inner = MedoidSelector(scales)
        self.horizon = int(horizon_steps)
        self.identity = dict(self._inner.identity, adapter="harnessstudio.selector", horizon_steps=self.horizon)

    def select(self, chunks, extras, obs) -> int:
        return self._inner.select(list(chunks), self.horizon)
