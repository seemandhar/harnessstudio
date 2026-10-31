"""FastHarness review scheduler and recovery state machine (fast_harness/scheduler.py, recovery.py @ dfe070f).

Ported line for line; only the data types are local. The decision ("review this chunk, and why" or
"zero-call fast path") depends only on the experience store, the fresh proposal and the episode's own
review history.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from .memory import Match, Memory, Statistics


@dataclass(frozen=True)
class Observation:
    episode_id: str
    step: int
    features: tuple
    payload: dict
    images: Any = None          # public camera images (not serialized into the review JSON)


@dataclass(frozen=True)
class Proposal:
    request_id: str
    observation: Observation
    payload: dict
    actions: Any = None         # the fresh policy chunk [H, A]
    max_steps: int = 15
    warnings: tuple = ()
    blocked: bool = False


@dataclass(frozen=True)
class Transition:
    before: Observation
    after: Observation
    request_id: str
    stage: Optional[str]
    mode: str
    source: str
    steps: int


@dataclass(frozen=True)
class Checkpoint:
    id: str
    observation: Observation
    stage: str
    goal: str
    prerequisites: tuple


@dataclass(frozen=True)
class Config:
    max_interval: int = 6
    min_samples: int = 4
    cooldown_successes: int = 3
    max_unreviewed_steps: int = 60
    match_threshold: float = 0.06
    ambiguity_margin: float = 0.01
    error_radius: float = 0.04
    audit_probability: float = 0.05
    stall_chunks: int = 3
    attempt_chunks: int = 8
    recovery_chunks: int = 24
    max_checkpoints: int = 64
    max_episode_chunks: int = 1000
    always_review: bool = False
    episode_number: Optional[int] = None
    confidence_cap: float = 1.0
    relax_known_stage_change: bool = False
    contact_gating: bool = False
    contact_low_z: float = 0.95
    contact_descent_m: float = 0.05
    gate_stage_change_on_contact_only: bool = False
    commit_gating: bool = False
    commit_cap: int = 12
    commit_min_confidence: float = 0.9
    persist_error_review: bool = False
    seed: int = 0


def lower_success_bound(stats: Statistics) -> float:
    n = stats.successes + stats.failures
    if n == 0:
        return 0.0
    p, z = stats.successes / n, 1.96
    numerator = p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, numerator / (1 + z * z / n))


def interval(stats: Statistics, config: Config) -> int:
    if stats.total < config.min_samples:
        return 1
    if stats.ever_failed and stats.clean_streak < config.cooldown_successes:
        return 1
    confidence = min(config.confidence_cap, stats.total / config.min_samples)
    return 1 + int((config.max_interval - 1) * lower_success_bound(stats) ** 2 * confidence)


def contact_imminent(proposal: Proposal, config: Config) -> Optional[str]:
    """Grasp/release (gripper command crosses 0.5) or a descent onto the work surface in the proposal."""
    payload = getattr(proposal, "payload", None) or {}
    trajectory = payload.get("student_eef_trajectory") or []
    current = payload.get("current_eef") or {}
    for arm in ("left", "right"):
        openings = []
        start = (current.get(arm) or {}).get("gripper_opening_command") if isinstance(current.get(arm), dict) else None
        if isinstance(start, (int, float)) and not isinstance(start, bool):
            openings.append(float(start))
        heights = []
        for waypoint in trajectory:
            arm_state = waypoint.get(arm) if isinstance(waypoint, dict) else None
            if not isinstance(arm_state, dict):
                continue
            value = arm_state.get("gripper_opening_command")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                openings.append(float(value))
            position = arm_state.get("position")
            if isinstance(position, (list, tuple)) and len(position) == 3 and \
                    isinstance(position[2], (int, float)) and not isinstance(position[2], bool):
                heights.append(float(position[2]))
        if any((a - 0.5) * (b - 0.5) < 0 for a, b in zip(openings, openings[1:])):
            return "contact_gripper"
        if len(heights) >= 3 and (heights[0] - heights[-1]) >= config.contact_descent_m and \
                heights[-1] <= config.contact_low_z:
            return "contact_approach"
    return None


class Scheduler:
    def __init__(self, config: Config):
        self.config = config
        self.random = random.Random(config.seed)

    def reason(self, memory: Memory, match: Match, observation: Observation, proposal: Proposal, *,
               last_review_step: Optional[int], chunks_since_review: int, previous_stage: Optional[str],
               recovery_active: bool, force_review: bool, stalled: bool) -> tuple:
        cfg = self.config
        stats = memory.statistics(match.stage) if match.stage else None
        gap = interval(stats, cfg) if stats is not None else 1
        step_limit = cfg.max_unreviewed_steps
        if cfg.commit_gating and stats is not None and gap > 1 \
                and lower_success_bound(stats) >= cfg.commit_min_confidence \
                and not memory.near_error(match.stage, observation.features, cfg.error_radius):
            span = memory.span(match.stage)
            horizon = min(cfg.commit_cap, max(1, round(span))) if span else min(cfg.commit_cap, gap)
            step_limit = max(cfg.max_unreviewed_steps, horizon * 15 + 15)
        if cfg.always_review:
            return "baseline", 1, step_limit
        if last_review_step is None:
            return "episode_start", gap, step_limit
        if recovery_active:
            return "recovery", 1, step_limit
        if match.stage is None:
            return match.reason, 1, step_limit
        contact = contact_imminent(proposal, cfg) if cfg.contact_gating else None
        if contact:
            return contact, 1, step_limit
        if match.stage != previous_stage and not (gap > 1 and (
                cfg.relax_known_stage_change or cfg.gate_stage_change_on_contact_only)):
            return "stage_changed", 1, step_limit
        if force_review:
            return "previous_uncertainty", 1, step_limit
        if proposal.warnings or stalled:
            return "monitor_alert", 1, step_limit
        if memory.near_error(match.stage, observation.features, cfg.error_radius) and (
                cfg.persist_error_review or stats.clean_streak < cfg.cooldown_successes * 3):
            return "error_memory", 1, step_limit
        if observation.step - last_review_step >= step_limit:
            return "physical_step_limit", gap, step_limit
        if chunks_since_review >= gap:
            return "scheduled", gap, step_limit
        if self.random.random() < cfg.audit_probability:
            return "random_audit", gap, step_limit
        return None, gap, step_limit


class MotionMonitor:
    def __init__(self, limit: int):
        self.limit, self.still = limit, 0

    def observe(self, transition: Transition) -> None:
        before = transition.before.payload.get("current_state", [])
        after = transition.after.payload.get("current_state", [])
        stationary = bool(before and len(before) == len(after)) and all(
            abs(a - b) < 1e-5 for a, b in zip(before, after))
        self.still = self.still + 1 if stationary else 0

    @property
    def stalled(self) -> bool:
        return self.still >= self.limit


@dataclass
class Recovery:
    attempt_limit: int
    total_limit: int
    phase: str = "normal"
    attempts_failed: int = 0
    attempt_steps: int = 0
    total_steps: int = 0
    target: Optional[Checkpoint] = None
    excluded: set = field(default_factory=set)
    error_cases: set = field(default_factory=set)
    failure_step: int = 0
    failure_goal: str = "Recover task progress and student intent"

    @property
    def active(self) -> bool:
        return self.phase not in ("normal", "exhausted")

    def candidates(self, checkpoints: list) -> tuple:
        return tuple(point for point in checkpoints
                     if point.id not in self.excluded and point.observation.step < self.failure_step)

    def context(self) -> dict:
        return dict(phase=self.phase, failed_attempts=self.attempts_failed,
                    attempt_chunks=self.attempt_steps, total_chunks=self.total_steps,
                    target_checkpoint_id=self.target.id if self.target else None,
                    goal=self.target.goal if self.target and self.phase.startswith("return_") else self.failure_goal,
                    remaining_attempt_chunks=self.attempt_limit - self.attempt_steps,
                    remaining_total_chunks=self.total_limit - self.total_steps)

    def start(self, step: int, goal: str = "Recover task progress and student intent") -> None:
        self.failure_goal = goal
        self.phase, self.failure_step = "local", step
        self.attempts_failed = self.attempt_steps = 0
        self.target = None
        self.excluded.clear()
        if self.total_steps >= self.total_limit:
            self.phase = "exhausted"

    def executed(self) -> None:
        if self.active:
            self.attempt_steps += 1
            self.total_steps += 1

    def update(self, review, checkpoints: list) -> bool:
        previous_phase = self.phase
        if self.phase == "select":
            options = {point.id: point for point in self.candidates(checkpoints)}
            if review.selected_checkpoint_id is None:
                self.phase = "exhausted"
            elif review.selected_checkpoint_id not in options:
                raise ValueError("Reviewer selected an ineligible checkpoint")
            else:
                self.target = options[review.selected_checkpoint_id]
                self.excluded.add(self.target.id)
                self.phase, self.attempt_steps = "return_selected", 0
            return True
        if not self.active or not self.attempt_steps:
            return False
        status = review.recovery_status
        attempt_ended = status in ("succeeded", "failed") or self.attempt_steps >= self.attempt_limit
        if status == "succeeded" and review.last_outcome != "ok":
            raise ValueError("A failed transition cannot complete a recovery goal")
        if status == "succeeded":
            if self.phase.startswith("return_"):
                self.phase, self.attempts_failed = "after_return", 0
                self.attempt_steps = 0
            else:
                self.phase, self.target = "normal", None
                self.attempt_steps = self.attempts_failed = 0
        elif status == "failed" or self.attempt_steps >= self.attempt_limit:
            self.attempt_steps = 0
            if self.phase.startswith("return_"):
                self.phase = "select"
            else:
                self.attempts_failed += 1
                if self.attempts_failed >= 2:
                    options = self.candidates(checkpoints)
                    if self.phase == "local" and options:
                        self.target = max(options, key=lambda point: point.observation.step)
                        self.excluded.add(self.target.id)
                        self.phase = "return_latest"
                    else:
                        self.phase = "select"
                    self.attempts_failed = 0
        if self.active and self.total_steps >= self.total_limit:
            self.phase = "exhausted"
        if self.phase == "select" and not self.candidates(checkpoints):
            self.phase = "exhausted"
        return self.phase != previous_phase or attempt_ended
