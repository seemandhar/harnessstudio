"""SYNTHETIC fixtures for tier-1 checks. Never written to run evidence paths; not robot results.

The observation and action layouts mirror GR00T N1.5 panda_omron (9 public keys, 12 flat
action dims in sorted-key order), using plain Python lists so the core needs no NumPy.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from ..core.pipeline import StepResult
from ..core.runtime import ProposeResult

PUBLIC_KEYS = (
    "annotation.human.task_description", "state.base_position", "state.base_rotation",
    "state.end_effector_position_relative", "state.end_effector_rotation_relative",
    "state.gripper_qpos", "video.robot0_agentview_left", "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
ACTION_SHAPES = {"action.base_motion": 4, "action.control_mode": 1, "action.end_effector_position": 3,
                 "action.end_effector_rotation": 3, "action.gripper_close": 1}
CHUNK = 16
ACTION_LAYOUT = {"action.base_motion": (0, 4), "action.control_mode": (4, 5), "action.end_effector_position": (5, 8),
                 "action.end_effector_rotation": (8, 11), "action.gripper_close": (11, 12)}
PROPRIO_EEF = ("state.end_effector_position_relative", None)
ACTION_DIM = 12
INSTRUCTION = "pick up the synthetic cube"


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def input_hashes(observation) -> dict:
    result = {k: _sha(observation[k]) for k in sorted(observation)}
    result["__aggregate__"] = _sha(result)
    return result


def flatten(native) -> list:
    return [sum((list(native[key][t]) for key in sorted(ACTION_SHAPES)), []) for t in range(CHUNK)]


def action_hashes(native) -> str:
    return _sha(flatten(native))


def instruction_of(observation) -> str:
    return observation["annotation.human.task_description"][0]


def with_instruction(observation, text):
    out = dict(observation)
    out["annotation.human.task_description"] = [text]
    return out


def _unflatten(flat) -> dict:
    native, offset = {}, 0
    for key in sorted(ACTION_SHAPES):
        width = ACTION_SHAPES[key]
        native[key] = [list(row[offset:offset + width]) for row in flat]
        offset += width
    return native


class SyntheticPolicy:
    """Deterministic in (state, seed). Binary dims (control_mode=4, gripper_close=11) are 0/1."""
    def __call__(self, observation, seed: int):
        state = observation["state.end_effector_position_relative"]
        rng = random.Random(seed * 1000003 + int(_sha(state)[:8], 16))
        flat = []
        for _ in range(CHUNK):
            row = [round(rng.uniform(-1, 1), 6) for _ in range(ACTION_DIM)]
            row[4] = float(rng.random() < 0.5)
            row[11] = float(rng.random() < 0.5)
            row[5:8] = [round(0.5 + 0.5 * rng.random(), 6) for _ in range(3)]
            flat.append(row)
        return _unflatten(flat), flat


class SyntheticClient:
    """Implements the PolicyClient protocol over SyntheticPolicy."""
    def __init__(self, policy=None, velocity=False):
        self.policy = policy or SyntheticPolicy()
        self.velocity = velocity
        self.calls = []

    def get_action_seeded(self, observation, seed):
        self.calls.append(("seeded", seed))
        return self.policy(observation, seed)

    def propose(self, observation, *, arm, decision_index, step_index, seeds, want):
        self.calls.append(("propose", arm, decision_index, step_index, tuple(seeds)))
        natives, flats = zip(*(self.policy(observation, s) for s in seeds))
        extras = {}
        if "sampler" in want:
            extras["sampler"] = {"velocity": self._velocity(seeds[0])}
        return ProposeResult(list(natives), list(flats), extras, {"policy_calls": len(seeds)})

    @staticmethod
    def _velocity(seed):
        import numpy as np
        return np.random.default_rng(seed).standard_normal((4, CHUNK, 32)).astype(np.float32)


@dataclass
class SyntheticEnv:
    """Integrates commanded EEF deltas (0.05 m per unit); success never occurs by default.

    stall_steps: global step indices where the arm does not move (pinned contact).
    render_noise_steps: global step indices where one camera value flips (class-S stand-in).
    """
    horizon: int = 64
    goal: tuple = (9.0, 9.0, 9.0)
    stall_steps: frozenset = frozenset()
    render_noise_steps: frozenset = frozenset()
    t: int = 0
    pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def observation(self):
        noise = 1 if self.t in self.render_noise_steps else 0
        return {
            "annotation.human.task_description": [INSTRUCTION],
            "state.base_position": [[0.0, 0.0, 0.0]],
            "state.base_rotation": [[0.0, 0.0, 0.0, 1.0]],
            "state.end_effector_position_relative": [[round(v, 6) for v in self.pos]],
            "state.end_effector_rotation_relative": [[0.0, 0.0, 0.0]],
            "state.gripper_qpos": [[0.0, 0.0]],
            "video.robot0_agentview_left": [[self.t % 7 + noise]],
            "video.robot0_agentview_right": [[0]],
            "video.robot0_eye_in_hand": [[0]],
        }

    def execute(self, native, start, count):
        flat = flatten(native)
        executed, success, terminated = 0, False, False
        for i in range(start, start + count):
            row = flat[i]
            if self.t not in self.stall_steps:
                self.pos = [p + 0.05 * row[5 + j] for j, p in enumerate(self.pos)]
            self.t += 1
            executed += 1
            success = sum((p - g) ** 2 for p, g in zip(self.pos, self.goal)) < 0.02 ** 2
            if success or self.t >= self.horizon:
                terminated = True
                break
        return StepResult(self.observation(), success, executed, terminated)


# Adversarial executors for the budget check (tier 1, check 6).
class MinKExecutor:
    identity = {"module": "fixture.min_k"}

    def __init__(self, min_k=1, per_step=True):
        self.min_k, self.per_step = min_k, per_step

    def plan(self, chunk, extras, obs):
        return self.min_k

    def continue_after(self, executed, obs):
        return True

    def record_executed(self, executed_actions):
        pass


class CutEverywhereExecutor(MinKExecutor):
    identity = {"module": "fixture.cut_everywhere"}

    def plan(self, chunk, extras, obs):
        return CHUNK

    def continue_after(self, executed, obs):
        return False


class BelowMinKExecutor(MinKExecutor):
    identity = {"module": "fixture.below_min_k"}

    def plan(self, chunk, extras, obs):
        return self.min_k - 1


class UpperCaseOp:
    identity = {"module": "fixture.uppercase"}

    def start_episode(self, instruction, obs):
        return instruction.upper()
