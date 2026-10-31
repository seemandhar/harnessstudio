"""Synthetic benchmark for tier 1: deterministic, CPU-only, runs in-process through vla-eval.

The observation includes a private key ('secret_goal') that the 'fake' contract does not expose, so
tier 1 can check that methods never see it while the policy still receives the raw observation.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import DimSpec


class FakeBenchmark(StepBenchmark):
    def __init__(self, n_tasks: int = 2, horizon: int = 12, action_dim: int = 7, task_order: str = "forward",
                 instruction: str = "reach the goal of {name}") -> None:
        super().__init__()
        self.n_tasks, self.horizon, self.action_dim = n_tasks, horizon, action_dim
        self.task_order, self.instruction = task_order, instruction

    def get_tasks(self) -> list[dict[str, Any]]:
        tasks = [{"name": f"fake_task_{i}"} for i in range(self.n_tasks)]
        return tasks[::-1] if self.task_order == "reverse" else tasks

    def reset(self, task: dict[str, Any]) -> Any:
        key = f"{task['name']}|{task.get('episode_idx', 0)}"
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        self._pos = rng.uniform(-1.0, 1.0, 2)
        self._goal = rng.uniform(-1.5, 1.5, 2)
        self._t = 0
        return {"pos": self._pos.copy()}

    def step(self, action: Any) -> StepResult:
        a = np.asarray(action["actions"], dtype=np.float64).reshape(-1)
        if a.shape != (self.action_dim,):
            raise ValueError(f"FakeBenchmark expects a {self.action_dim}-D action, got {a.shape}")
        self._pos = self._pos + 0.4 * a[:2]
        self._t += 1
        success = bool(np.linalg.norm(self._pos - self._goal) < 0.6)
        return StepResult(obs={"pos": self._pos.copy()}, reward=float(success),
                          done=success or self._t >= self.horizon, info={"success": success})

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        level = int((raw_obs["pos"][0] + 5.0) * 25.0) % 256
        return {"images": {"cam": np.full((8, 8, 3), level, dtype=np.uint8)},
                "state": raw_obs["pos"].astype(np.float32),
                "task_description": self.instruction.format(name=task["name"]),
                "secret_goal": self._goal.astype(np.float32)}

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        return {"success": bool(step_result.info.get("success", False))}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self.horizon}

    def get_action_spec(self) -> dict[str, DimSpec]:
        from ..policy_ext.fake import fake_action_spec
        return fake_action_spec(self.action_dim)

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {}
