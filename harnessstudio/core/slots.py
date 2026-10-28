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


class GuidedSampler(Protocol):
    """Server-slot sampler for policies with Capability.DIFFERENTIABLE_DENOISER (Kinetix RTC extension).

    guide() is called PER ROLLOUT inside the backend's vmap over the batch, once per flow step:
    x_t, v_t, x1_hat, prev: [H, A] float32; t: scalar; vjp(cot [H, A]) -> [H, A] is the pullback of
    x1_hat(x_t) = x_t + (1 - t) * v(obs, x_t, t); d, s, H: Python ints (protocol constants); xp: the
    array namespace (jax.numpy on kinetix-flow, numpy in tier 1). Must be a pure function: no side
    effects, no I/O, no randomness, no host callbacks. Returns the velocity to integrate, [H, A].
    """
    identity: dict
    def guide(self, xp: Any, x_t: Any, t: Any, v_t: Any, x1_hat: Any, vjp: Any, prev: Any,
              d: int, s: int, H: int) -> Any: ...
    def extras(self) -> dict: ...


def sampler_has_guide(ref) -> bool:
    """True when a sampler ModuleRef's factory exposes guide(). Imports only that sampler's module."""
    import importlib
    module, _, attr = ref.factory.partition(":")
    return callable(getattr(getattr(importlib.import_module(module), attr), "guide", None))
