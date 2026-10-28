"""Policy-facing contracts. No simulator, rewards, resets, or success oracle."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def public_hash(value: Any) -> str:
    """Hash nested public inputs including array dtype/shape, without logging pixels."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, Mapping):
            h.update(b"mapping")
            for key in sorted(item):
                h.update(str(key).encode()); visit(item[key])
        elif hasattr(item, "dtype") and hasattr(item, "tobytes"):
            h.update(str(item.dtype).encode()); h.update(str(item.shape).encode())
            h.update(item.tobytes())
        elif isinstance(item, (tuple, list)):
            h.update(b"sequence")
            for v in item:
                visit(v)
        else:
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        h.update(b"\0")
    visit(value)
    return h.hexdigest()


FORBIDDEN_KEYS = frozenset({"reward", "success", "done", "info", "sim", "sim_state",
    "physics", "qpos", "qvel", "object_states", "privileged", "task_id", "seed",
    "case_id", "episode_id", "validation", "outcome"})


@dataclass(frozen=True)
class PublicObservation:
    """Construct only from the benchmark's reviewed public-input allowlist.

    This is a workflow boundary, not a sandbox for arbitrary Python plugins.
    Native policy preprocessing stays in the adapter.
    """
    values: Mapping[str, Any]
    sha256: str

    @classmethod
    def from_mapping(cls, observation: Mapping[str, Any], allowed_keys: Sequence[str]):
        keys = tuple(allowed_keys)
        if not keys or len(set(keys)) != len(keys) or set(keys) & FORBIDDEN_KEYS:
            raise ValueError("A nonempty unique public-input allowlist is required")
        if any(k not in observation for k in keys):
            raise ValueError("Missing required public observation")
        values = copy.deepcopy({k: observation[k] for k in keys})
        def check(item):
            if isinstance(item, Mapping):
                if set(item) & FORBIDDEN_KEYS:
                    raise ValueError("Evaluator metadata cannot enter public observations")
                for v in item.values(): check(v)
            elif isinstance(item, (tuple, list)):
                for v in item: check(v)
        check(values)
        return cls(MappingProxyType(values), public_hash(values))


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    checkpoint_identity: str
    capabilities: frozenset[str] = frozenset()
    weights_frozen: bool = True

    def __post_init__(self):
        if not self.policy_id or not self.checkpoint_identity or self.weights_frozen is not True:
            raise ValueError("An identified frozen original policy is required")


@dataclass(frozen=True)
class RuntimeContract:
    benchmark: str
    horizon: int
    control_hz: float
    native_chunk_steps: int
    action_dim: int
    public_keys: tuple[str, ...]

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.horizon, self.native_chunk_steps, self.action_dim)):
            raise ValueError("Horizon, native chunk length and action dimension must be positive integers")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("Native control frequency must be finite and positive")
        if not self.benchmark or not self.public_keys or set(self.public_keys) & FORBIDDEN_KEYS:
            raise ValueError("Benchmark and permitted public inputs must be explicit")


@dataclass(frozen=True)
class Budget:
    samples_per_decision: int = 1
    max_policy_calls: int = 1000
    max_operator_calls: int = 0
    max_operator_tokens: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise ValueError("Inference budgets must be nonnegative integers")
        if self.samples_per_decision < 1 or self.max_policy_calls < 1:
            raise ValueError("At least one policy proposal is required")


@dataclass(frozen=True)
class ActionChunk:
    """Keep the native array object/dtype; validation never clips or rescales it."""
    actions: Any

    def validate(self, action_dim: int, min_steps: int):
        if len(self.actions) < min_steps:
            raise ValueError("Policy output shorter than its native execution contract")
        for row in self.actions:
            if len(row) != action_dim or any(not isinstance(v, Real) or not math.isfinite(v) for v in row):
                raise ValueError("Policy output has invalid shape or nonfinite actions")
        return self


class PolicyPort(Protocol):
    spec: PolicySpec

    def infer(self, observation: PublicObservation, instruction: str) -> ActionChunk:
        """One native frozen-policy proposal; no environment interaction."""

    def infer_alternative(self, observation: PublicObservation, instruction: str, index: int) -> ActionChunk:
        """Optional: isolate RNG AND native queues/caches from the primary stream."""


class InstructionOperator(Protocol):
    required_capabilities: frozenset[str]
    identity: Mapping[str, Any]

    def rewrite(self, observation: PublicObservation, instruction: str,
                memories: tuple) -> tuple[str, int]:
        """Return instruction and actual charged tokens, only at episode start."""
