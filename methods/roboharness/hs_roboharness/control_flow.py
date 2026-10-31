"""Control-flow state machines ported verbatim from src/soma_control_flow.py (RoboHarness repo, MIT, 716defc).

RollbackState     "Encore": replay -accumulated/N for N steps, then a stabilization buffer (zero motion, gripper
                  open), then one zero action, then control returns to the policy.
KeyStepRetryState timeout-triggered rollback when the step passes key_steps.end + timeout_grace.
TaskDecomposeState subtask queue; advances only on a success signal held for 2 consecutive steps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np


@dataclass
class RollbackState:
    active: bool = False
    phase: str = ""  # "reverse" | "buffer" | "done"
    reverse_steps_total: int = 100
    buffer_steps_total: int = 25
    reverse_step_idx: int = 0
    buffer_step_idx: int = 0
    reverse_action: np.ndarray | None = None  # (B, D)
    reason: str = ""
    need_hard_reset: bool = False
    action_dim: int = 7

    def start_from_accumulated(self, accumulated_actions: np.ndarray, *, reverse_steps: int = 100,
                               buffer_steps: int = 25, reason: str = ""):
        self.active = True
        self.phase = "reverse"
        self.reverse_steps_total = int(reverse_steps)
        self.buffer_steps_total = int(buffer_steps)
        self.reverse_step_idx = 0
        self.buffer_step_idx = 0
        if accumulated_actions.ndim > 1:
            self.action_dim = accumulated_actions.shape[-1]
        self.reverse_action = -accumulated_actions / max(1, self.reverse_steps_total)
        self.reason = reason
        logging.warning(f"[CTRL] Rollback Initiated | Reason: {reason} | Total Steps: {self.reverse_steps_total} "
                        f"| Action Dim: {self.action_dim}")

    def step_action(self, num_envs: int) -> np.ndarray:
        if not self.active or self.reverse_action is None:
            raise RuntimeError("RollbackState.step_action was called, but the rollback state is not active.")
        if self.phase == "reverse":
            self.reverse_step_idx += 1
            if self.reverse_step_idx >= self.reverse_steps_total:
                self.phase = "buffer" if self.buffer_steps_total > 0 else "done"
            return self.reverse_action.astype(np.float32)
        if self.phase == "buffer":
            self.buffer_step_idx += 1
            dummy_action = np.zeros(self.action_dim, dtype=np.float32)
            if self.action_dim >= 1:
                dummy_action[-1] = -1.0
            if self.buffer_step_idx >= self.buffer_steps_total:
                self.phase = "done"
            return np.array([dummy_action] * num_envs, dtype=np.float32)
        self.need_hard_reset = True
        self.reset()
        return np.zeros((num_envs, self.action_dim), dtype=np.float32)

    def reset(self):
        self.active = False
        self.phase = "done"
        self.reverse_action = None
        self.reason = ""


@dataclass
class KeyStepRetryState:
    key_steps: dict = field(default_factory=lambda: {"start": 0, "end": 0, "timeout_grace": 0})
    armed: bool = False
    triggered: bool = False

    def update_from_control_flags(self, control_flags: dict[str, Any]):
        ks = control_flags.get("key_steps")
        if isinstance(ks, dict) and "start" in ks and "end" in ks:
            self.key_steps = {"start": int(ks.get("start", 0)), "end": int(ks.get("end", 0)),
                              "timeout_grace": int(ks.get("timeout_grace", 0))}
            self.armed = True
            self.triggered = False
        if control_flags.get("key_step_retry") is True:
            self.armed = True

    def should_trigger(self, *, step: int, success_any: bool) -> bool:
        if not self.armed or self.triggered:
            return False
        if success_any:
            self.armed = False
            return False
        end = self.key_steps.get("end", 0)
        grace = self.key_steps.get("timeout_grace", 0)
        return step >= (end + grace) and end > 0

    def mark_triggered(self):
        self.triggered = True
        logging.warning(f"[CTRL] KeyStepRetry Triggered | Configuration: {self.key_steps}")

    def reset(self):
        self.armed = False
        self.triggered = False
        self.key_steps = {"start": 0, "end": 0, "timeout_grace": 0}


@dataclass
class TaskDecomposeState:
    subtasks: List[str] = field(default_factory=list)
    subtask_idx: int = 0
    success_hold_steps: int = 0

    def update_from_control_flags(self, control_flags: dict[str, Any]):
        sts = control_flags.get("subtasks")
        if isinstance(sts, list) and all(isinstance(x, str) for x in sts) and len(sts) > 0:
            if sts != self.subtasks:
                self.subtasks = sts
                self.subtask_idx = 0
                self.success_hold_steps = 0
                logging.info(f"[CTRL] TaskDecompose Sequence Updated | Steps: {len(sts)} | Initial: {sts[0]}")

    def current_task(self, default_task: str) -> str:
        if self.subtasks and 0 <= self.subtask_idx < len(self.subtasks):
            return self.subtasks[self.subtask_idx]
        return default_task

    def maybe_advance(self, *, success_any: bool):
        if not self.subtasks:
            return
        if success_any:
            self.success_hold_steps += 1
            if self.success_hold_steps >= 2:
                if self.subtask_idx < len(self.subtasks) - 1:
                    self.subtask_idx += 1
                    self.success_hold_steps = 0
                    logging.info(f"[CTRL] Task Advanced >>> Step {self.subtask_idx}: {self.subtasks[self.subtask_idx]}")
        else:
            self.success_hold_steps = 0

    def reset(self):
        self.subtasks = []
        self.subtask_idx = 0
        self.success_hold_steps = 0
