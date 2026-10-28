"""Policy capabilities (spec §4.1)."""
from __future__ import annotations

from enum import Enum
from typing import Iterable

from .errors import CapabilityError


class Capability(str, Enum):
    ACTION_CHUNK = "action_chunk"
    SEEDED_SAMPLING = "seeded_sampling"
    ISOLATED_STREAMS = "isolated_streams"
    DENOISE_STEP = "denoise_step"
    VELOCITY_READOUT = "velocity_readout"
    PROPRIO = "proprio"
    VARIABLE_EXECUTE = "variable_execute"
    PREDICTED_FUTURE = "predicted_future"


_UNIFIED_ALIASES = {Capability.ISOLATED_STREAMS: ("independent_sampling", "isolated_sampling")}


def caps(*values: str) -> frozenset:
    return frozenset(Capability(v) for v in values)


def unified_names(capabilities: Iterable[Capability]) -> frozenset:
    names = set()
    for cap in capabilities:
        names.add(cap.value)
        names.update(_UNIFIED_ALIASES.get(cap, ()))
    return frozenset(names)


def require(required: Iterable[Capability], available: Iterable[Capability], *, method: str, policy: str) -> None:
    missing = frozenset(required) - frozenset(available)
    if missing:
        raise CapabilityError(f"{method} needs {sorted(m.value for m in missing)}, "
                              f"which {policy} does not provide")
