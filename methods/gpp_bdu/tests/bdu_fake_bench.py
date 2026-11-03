"""Fake gripper benchmark + scripted spy policy for gpp-bdu seam tests (copy of methods/gpp/tests/gpp_fake_bench.py
with a `press` policy mode: blocked descent with a lateral command; CPU, in-process vla-eval).

GripperBench exposes LIBERO-like public `states` = [x, y, z, 0, 0, 0, qL, qR] (finger width = qL - qR) so the gpp
triggers see real proprio through the HarnessServer. Physics (per step): xyz += 0.0082 * a[:3] (a_z 0.5 -> 4.1 mm, the
measured OpenVLA/LIBERO gain); z >= floor; gripper close (a[6] >= 0) moves the width 0.03 toward 0.05 if the fingers
are at/below the object top (custody: the object then moves with the EE) else toward 0; open moves it toward 0.08.
`stuck`: a held object cannot rise (blocked lift). The policy is Markov in states plus a per-session count of
consecutive held observations: approach down until z <= 1.03, close, and after `hold_obs` held observations propose an
open (a settled, never-lifted clamp), then keep opening.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import DimSpec

from harnessstudio.policy_ext.fake import FakePolicyServer, fake_action_spec

Z0, CLOSE_Z = 1.05, 1.03


class GripperBench(StepBenchmark):
    def __init__(self, obj_top: float = 1.005, floor: float = 0.90, stuck: bool = False, horizon: int = 80,
                 n_tasks: int = 1, task_prefix: str = "grip_task_", gain: float = 0.0082, ramp: float = 1.0,
                 **_: Any) -> None:
        super().__init__()
        self.obj_top, self.floor, self.stuck, self.horizon = obj_top, floor, stuck, horizon
        # gain: metres per unit action; ramp: realized fraction of the first moving step after >= 2 zero-motion
        # steps (OSC ramp-up from rest; 1.0 = no ramp)
        self.gain, self.ramp = gain, ramp
        self.n_tasks, self.task_prefix = n_tasks, task_prefix

    def get_tasks(self):
        return [{"name": f"{self.task_prefix}{i}"} for i in range(self.n_tasks)]

    def reset(self, task):
        self.p = np.array([0.0, 0.0, Z0])
        self.w = 0.08
        self.held = False
        self.t = 0
        self.still = 0
        return self._raw()

    def _raw(self):
        return {"p": self.p.copy(), "w": self.w}

    def step(self, action):
        a = np.asarray(action["actions"], dtype=np.float64).reshape(-1)
        dp = self.gain * a[:3]
        moving = bool(np.abs(a[:3]).max() > 0)
        if moving and self.still >= 2:
            dp = dp * self.ramp
        self.still = 0 if moving else self.still + 1
        if self.held and self.stuck:
            dp[2] = min(dp[2], 0.0)
        self.p = self.p + dp
        self.p[2] = max(self.p[2], self.floor)
        if a[6] >= 0:
            target = 0.05 if (self.held or self.p[2] <= self.obj_top) else 0.0
            if self.w > 0.06 and self.p[2] <= self.obj_top:
                self.held = True
                target = 0.05
            self.w = max(target, self.w - 0.03) if self.w > target else target
        else:
            self.w = min(0.08, self.w + 0.03)
            if self.w > 0.06:
                self.held = False
        self.t += 1
        return StepResult(obs=self._raw(), reward=0.0, done=self.t >= self.horizon, info={"success": False})

    def make_obs(self, raw, task):
        p, w = raw["p"], raw["w"]
        states = np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, w / 2, -w / 2], dtype=np.float64)
        level = int(abs(p[2]) * 1000 + w * 1000) % 256
        return {"images": {"cam": np.full((8, 8, 3), level, dtype=np.uint8)}, "states": states,
                "state": states[:2].astype(np.float32), "task_description": f"do {task['name']}"}

    def get_step_result(self, step_result):
        return {"success": False}

    def get_metadata(self):
        return {"max_steps": self.horizon}

    def get_action_spec(self):
        return fake_action_spec(7)

    def get_observation_spec(self):
        return {}


class GripperPolicy(FakePolicyServer):
    def __init__(self, hold_obs: int = 12, frozen: bool = False, press: float = 0.0, close_at=None) -> None:
        super().__init__(chunking="harness", seeded=True, chunk_len=1)
        self.hold_obs, self.frozen = hold_obs, frozen
        # press: keep commanding a_z -0.5 with a_x = press and the gripper open (a blocked descent against the floor);
        # close_at: from that request index on (per session) command a close instead
        self.press, self.close_at = press, close_at
        self.requests: dict = {}
        self.log: list = []
        self.held_count: dict = {}

    def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
        s = np.asarray(obs["states"], dtype=np.float64)
        z, w = s[2], s[6] - s[7]
        sid = ctx.session_id
        held = 0.01 < w < 0.06
        self.held_count[sid] = self.held_count.get(sid, 0) + 1 if held else 0
        n = self.requests[sid] = self.requests.get(sid, -1) + 1
        if self.press:
            a = ([self.press, 0.0, -0.5, 0.0, 0.0, 0.0, -1.0] if self.close_at is None or n < self.close_at
                 else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        elif self.frozen:
            a = [0.002, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]          # start-of-episode fixed point (null action)
        elif w > 0.06 and z > CLOSE_Z:
            a = [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -1.0]
        elif self.held_count[sid] > self.hold_obs:
            a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]          # release a never-lifted clamp
        elif w > 0.06 and self.held_count[sid] == 0 and self._released(sid):
            a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        else:
            a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        if self.held_count[sid] > self.hold_obs:
            self.released = getattr(self, "released", set()) | {sid}
        self.log.append({"session": sid, "seeds": list(seeds), "sampler": sampler, "z": float(z), "w": float(w),
                         "action": a})
        return [(np.array([a], dtype=np.float32), {"seed": int(x)}) for x in seeds]

    def _released(self, sid) -> bool:
        return sid in getattr(self, "released", set())
