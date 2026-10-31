"""Analytic skills for end-effector-delta control (spec 5.4).

Each skill is `async def skill(ctx, ...) -> SkillResult`, closed loop on the public end-effector position
(ctx.embodiment.eef_position(obs)). Constants come from the benchmark contract: eef_delta_scale (metres
per unit position action) and the gripper open/close commands. Actions are zero except the dims named
"position" and "gripper" in the action layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class SkillResult:
    ok: bool
    steps: int
    reason: str = ""
    final: dict = field(default_factory=dict)


def _dims(ctx, name: str) -> list:
    return ctx.embodiment.dims(name)


def _blank(ctx) -> np.ndarray:
    dim = ctx.embodiment.action_dim
    if not dim:
        raise ValueError("Skills need the action dimension (policy action_spec or contract action_dim)")
    return np.zeros(int(dim), dtype=np.float32)


def _eef(ctx) -> np.ndarray:
    pos = ctx.embodiment.eef_position(ctx.observe())
    if pos is None:
        raise ValueError("This benchmark does not expose the end-effector position (contract eef_position)")
    return pos


def _gripper_value(ctx, gripper) -> Optional[float]:
    if gripper is None:
        return None
    if gripper in ("open", "close"):
        return ctx.embodiment.gripper_open if gripper == "open" else ctx.embodiment.gripper_close
    return float(gripper)


async def move_to(ctx, target: Sequence[float], *, tolerance: float = 0.01, gain: float = 1.0,
                  max_steps: int = 60, gripper=None, max_action: float = 1.0) -> SkillResult:
    """Move the end effector to an absolute position (benchmark frame) with a proportional controller."""
    scale = ctx.embodiment.eef_delta_scale
    if not scale:
        raise ValueError("move_to needs contract eef_delta_scale")
    pdims = _dims(ctx, "position")
    gval = _gripper_value(ctx, gripper)
    target = np.asarray(target, dtype=np.float64)
    steps = 0
    while steps < max_steps and not ctx.ended:
        pos = _eef(ctx)
        n = min(len(pos), len(target), len(pdims))
        err = target[:n] - pos[:n]
        if float(np.linalg.norm(err)) <= tolerance:
            return SkillResult(True, steps, "reached", {"error": float(np.linalg.norm(err))})
        action = _blank(ctx)
        action[pdims[:n]] = np.clip(gain * err / scale, -max_action, max_action)
        if gval is not None:
            action[_dims(ctx, "gripper")] = gval
        steps += await ctx.execute(action[None, :])
    pos = _eef(ctx)
    n = min(len(pos), len(target))
    err = float(np.linalg.norm(target[:n] - pos[:n]))
    return SkillResult(err <= tolerance, steps, "reached" if err <= tolerance else "max_steps", {"error": err})


async def move_by(ctx, delta: Sequence[float], **kwargs) -> SkillResult:
    """Move the end effector by a relative offset (benchmark frame)."""
    pos = _eef(ctx)
    delta = np.asarray(delta, dtype=np.float64)
    n = min(len(pos), len(delta))
    return await move_to(ctx, pos[:n] + delta[:n], **kwargs)


async def set_gripper(ctx, gripper="close", *, steps: int = 5) -> SkillResult:
    """Hold position and command the gripper for `steps` steps."""
    action = _blank(ctx)
    action[_dims(ctx, "gripper")] = _gripper_value(ctx, gripper)
    done = 0
    for _ in range(steps):
        if ctx.ended:
            break
        done += await ctx.execute(action[None, :])
    return SkillResult(done == steps, done, "done" if done == steps else "episode_ended")


async def release(ctx, *, steps: int = 5) -> SkillResult:
    return await set_gripper(ctx, "open", steps=steps)


async def vla_act(ctx, instruction: Optional[str] = None, *, steps: int = 80) -> SkillResult:
    """The frozen policy as a skill: whole decisions with this prompt until `steps` are executed."""
    done = 0
    while done < steps and not ctx.ended:
        done += await ctx.decide(instruction=instruction)
    return SkillResult(True, done, "done" if done >= steps else "episode_ended")
