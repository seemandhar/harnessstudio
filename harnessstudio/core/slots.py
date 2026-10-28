"""Slot protocols (spec §4.2). Modules never receive case identity, seeds or the arm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence


@dataclass(frozen=True)
class DecisionCtx:
    decision_index: int
    step_index: int
    public_input_sha256: str
    instruction_sha256: str
    policy_calls_used: int
    policy_calls_budget: int


class Sampler(Protocol):
    identity: dict
    def on_step(self, step: int, x_t: Any, v_t: Any, ctx: DecisionCtx) -> Optional[Any]: ...
    def extras(self) -> dict: ...


class Proposer(Protocol):
    identity: dict
    num_samples: int


class Selector(Protocol):
    identity: dict
    def select(self, chunks: Sequence[Any], extras: dict, obs: Any) -> int: ...


class Executor(Protocol):
    identity: dict
    min_k: int
    per_step: bool
    def plan(self, chunk: Any, extras: dict, obs: Any) -> int: ...
    def continue_after(self, executed: int, obs: Any) -> bool: ...
    def record_executed(self, executed_actions: Any) -> None: ...


class EpisodeOp(Protocol):
    identity: dict
    def start_episode(self, instruction: str, obs: Any) -> str: ...
