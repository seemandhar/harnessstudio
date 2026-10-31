"""Component interfaces (spec 5.2). Components never receive the case, seeds, arm, reward or success.

Category mapping (RoboMetaHarness IDs):
  observation_grounding    -> ObservationOp
  language_alignment       -> InstructionOp
  proposal_selection       -> Proposer, Selector
  sampler_guidance         -> Sampler (runs inside the policy server; see policy_ext)
  uncertainty_compute      -> Gate
  execution_monitoring     -> Executor, Monitor
  contact_feedback         -> Monitor
  memory_retrieval         -> ActionOp (+ ExperienceStore, M4)

  agentic_orchestration    -> Controller
  skill_composition        -> skills (M4)
  world_model_verification -> Selector over predicted futures (T3)
  experience_evolution     -> Preparer (M4, offline)
Any component may define bind(embodiment) to receive public action-space facts
(Embodiment.action_layout: name -> (start, end), from the policy action_spec, in vector order).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np


class ObservationOp(Protocol):
    def transform(self, obs: dict) -> dict:
        """Return replacement values for public keys (e.g. edited images). Keys not returned are unchanged."""


class InstructionOp(Protocol):
    def rewrite(self, instruction: str, obs: dict, memory: dict) -> str: ...


class Gate(Protocol):
    def plan(self, obs: dict, history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Return {'num_samples': int} (and other method-specific switches) for this decision."""


class Proposer(Protocol):
    num_samples: int


class Selector(Protocol):
    def select(self, chunks: Sequence[np.ndarray], extras: Sequence[Mapping[str, Any]], obs: dict) -> int: ...


class ActionOp(Protocol):
    def transform(self, chunk: np.ndarray, obs: dict, memory: dict) -> np.ndarray: ...


class Executor(Protocol):
    """How much of the chosen chunk runs (v1 interface).
    plan() returns k >= min_k; with per_step=True, continue_after() is asked after each step.
    Set sees_proposals = True to get extras['proposals'] (every sample of the decision, seed order)."""
    min_k: int
    per_step: bool

    def plan(self, chunk: np.ndarray, extras: Mapping[str, Any], obs: dict) -> int: ...
    def continue_after(self, executed: int, obs: dict) -> bool: ...
    def record_executed(self, executed_actions: np.ndarray) -> None: ...


class Monitor(Protocol):
    def check(self, obs: dict, history: Sequence[Mapping[str, Any]]) -> str:
        """'continue' or 'interrupt' (ends the current chunk; the next decision re-plans)."""


class Controller(Protocol):
    """Owns the episode. Acts only through ctx (execute, decide, policy, tools, skills)."""
    async def run(self, ctx) -> None: ...
