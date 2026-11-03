"""Fake RGB-D gripper benchmark + scripted spy policy for witnessed-rescue seam tests (CPU, in-process vla-eval).

Adapted from harnessstudio/methods/gpp/tests/gpp_fake_bench.py (GripperBench / GripperPolicy, SCALE++): same
measured OpenVLA/LIBERO position gain (0.0082 m per unit action) and finger-width model, plus
- an object box (footprint, top height) that can be grasped where it actually is, carried, and released onto a tray;
- yaw integration (a[5]) in the published axis-angle;
- a top-down pinhole camera rendering agentview RGB + metric depth with robosuite-convention calibration
  (world_to_pixel maps [x, y, z, 1] to [col*z, row*z, z, 1]) published mirrored horizontally exactly like
  harnessstudio.benchmarks.libero_rgbd (`pixel_flip: horizontal`), and a small wrist RGB image.
Public observation keys: images {agentview, wrist}, depth {agentview}, camera {agentview}, states, task_description.
The spy policy is Markov in states plus per-session counters; it ignores samplers (like FakePolicyServer) and logs
every request (session, seeds, sampler).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult

from harnessstudio.policy_ext.fake import FakePolicyServer, fake_action_spec

TABLE = 0.90
HC, F, N = 1.6, 110.0, 128          # camera height, focal length (px), image size
CX = CY = (N - 1) / 2.0
OBJ = {"center": (0.10, 0.05), "size": (0.05, 0.04), "top": 0.98}
TRAY = {"center": (-0.10, 0.10), "size": (0.12, 0.12)}
GAIN, YAW_GAIN = 0.0082, 0.08


def world_to_pixel() -> np.ndarray:
    return np.array([[F, 0.0, -CX, CX * HC], [0.0, -F, -CY, CY * HC], [0.0, 0.0, -1.0, HC], [0.0, 0.0, 0.0, 1.0]])


def pixel_of(xyz, scale: int = 2) -> tuple:
    """Calibration-frame display pixel (row, col) of a world point (what a perfect planner would point at)."""
    p = world_to_pixel() @ np.append(np.asarray(xyz, dtype=np.float64), 1.0)
    return int(round(p[1] / p[2] * scale)), int(round(p[0] / p[2] * scale))


def _inside(x, y, center, size, margin=0.0):
    return (np.abs(x - center[0]) <= size[0] / 2 + margin) & (np.abs(y - center[1]) <= size[1] / 2 + margin)


class RGBDGripperBench(StepBenchmark):
    def __init__(self, horizon: int = 200, n_tasks: int = 1, task_prefix: str = "wr_task_",
                 instruction: str = "put the red block on the blue tray", slip_lift: float = 0.0,
                 wrist: bool = True, **_: Any) -> None:
        super().__init__()
        self.wrist = wrist              # False: no wrist RGB, like the libero_rgbd pilot benchmark
        self.slip_lift = slip_lift      # > 0: a held object slips out once lifted this far (first time only)
        self.horizon, self.n_tasks, self.task_prefix, self.instruction = horizon, n_tasks, task_prefix, instruction

    def get_tasks(self):
        return [{"name": f"{self.task_prefix}{i}"} for i in range(self.n_tasks)]

    def reset(self, task):
        self.p = np.array([0.0, 0.0, 1.05])
        self.yaw = 0.0
        self.w = 0.08
        self.obj_c = np.array(OBJ["center"], dtype=np.float64)
        self.obj_top = OBJ["top"]
        self.held = False
        self.offset = np.zeros(3)
        self.placed = False
        self.slipped = False
        self.t = 0
        return self._raw()

    def _raw(self):
        return {"p": self.p.copy(), "w": self.w, "yaw": self.yaw, "obj_c": self.obj_c.copy(), "obj_top": self.obj_top}

    def step(self, action):
        a = np.asarray(action["actions"], dtype=np.float64).reshape(-1)
        self.p = self.p + GAIN * a[:3]
        self.p[2] = max(self.p[2], TABLE + 0.01)
        self.yaw += YAW_GAIN * a[5]
        h = OBJ["top"] - TABLE
        if a[6] > 0:
            over = bool(_inside(self.p[0], self.p[1], self.obj_c, OBJ["size"], 0.01))
            if not self.held and self.w > 0.06 and over and TABLE < self.p[2] <= self.obj_top:
                self.held = True
                self.offset = np.array([self.obj_c[0] - self.p[0], self.obj_c[1] - self.p[1], self.obj_top - self.p[2]])
            target = 0.04 if self.held else 0.0
            self.w = max(target, self.w - 0.03)
        else:
            self.w = min(0.08, self.w + 0.03)
            if self.held and self.w > 0.06:
                self.held = False
                self.obj_top = TABLE + h        # drops onto the table / tray (same height)
                self.placed = bool(_inside(self.obj_c[0], self.obj_c[1], TRAY["center"], TRAY["size"]))
        if self.held and self.slip_lift > 0 and not self.slipped and self.obj_top - OBJ["top"] > self.slip_lift:
            self.held, self.slipped = False, True
            self.obj_top = TABLE + h
            self.w = 0.0
        if self.held:
            self.obj_c = self.p[:2] + self.offset[:2]
            self.obj_top = self.p[2] + self.offset[2]
        self.t += 1
        return StepResult(obs=self._raw(), reward=0.0, done=self.t >= self.horizon, info={"success": self.placed})

    @staticmethod
    def render(raw):
        rows, cols = np.mgrid[0:N, 0:N].astype(np.float64)
        zc_t = HC - TABLE
        xt, yt = (cols - CX) * zc_t / F, -(rows - CY) * zc_t / F
        depth = np.full((N, N), zc_t)
        rgb = np.zeros((N, N, 3), dtype=np.uint8) + 128
        tray = _inside(xt, yt, TRAY["center"], TRAY["size"])
        rgb[tray] = (40, 60, 200)
        zc_o = HC - raw["obj_top"]
        xo, yo = (cols - CX) * zc_o / F, -(rows - CY) * zc_o / F
        box = _inside(xo, yo, raw["obj_c"], OBJ["size"])
        depth[box] = zc_o
        rgb[box] = (200, 30, 30)
        return rgb, depth.astype(np.float32)

    def make_obs(self, raw, task):
        rgb, depth = self.render(raw)
        p, w = raw["p"], raw["w"]
        states = np.array([p[0], p[1], p[2], 0.0, 0.0, raw["yaw"], w / 2, -w / 2], dtype=np.float64)
        wrist = np.full((16, 16, 3), int(w * 1000) % 256, dtype=np.uint8)
        cam = {"world_to_pixel": world_to_pixel(), "height": N, "width": N, "pixel_flip": "horizontal"}
        images = {"agentview": np.ascontiguousarray(rgb[:, ::-1])}
        if self.wrist:
            images["wrist"] = wrist
        return {"images": images,
                "depth": {"agentview": np.ascontiguousarray(depth[:, ::-1])}, "camera": {"agentview": cam},
                "states": states, "task_description": self.instruction}

    def get_step_result(self, step_result):
        return {"success": bool(step_result.info.get("success"))}

    def get_metadata(self):
        return {"max_steps": self.horizon}

    def get_action_spec(self):
        return fake_action_spec(7)

    def get_observation_spec(self):
        return {}


class SpyPolicy(FakePolicyServer):
    """scenario 'h2': descend at the start xy, then close on air for 15 steps (H2), then open and idle.
    scenario 'h1': go over the object, descend, close on it, hold 12 held observations, then propose open (H1).
    scenario 'idle': small open-gripper motions only (never triggers)."""

    def __init__(self, scenario: str = "h2", dtype=np.float32) -> None:
        super().__init__(chunking="harness", seeded=True, chunk_len=1)
        self.scenario, self.dtype = scenario, dtype
        self.log: list = []
        self.counts: dict = {}

    def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
        s = np.asarray(obs["states"], dtype=np.float64)
        x, y, z, w = s[0], s[1], s[2], s[6] - s[7]
        sid = ctx.session_id
        c = self.counts.setdefault(sid, {"close": 0, "held": 0, "n": 0})
        c["n"] += 1
        if self.scenario == "idle":
            a = [0.1 * math.sin(c["n"] / 5.0), 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        elif self.scenario == "h2":
            if c["close"] == 0 and z > 0.95:
                a = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]
            elif c["close"] < 15:
                c["close"] += 1
                a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        elif self.scenario == "drop":
            tx, ty = OBJ["center"]
            if c["close"] > 0:
                c["close"] += 1
                a = [0.0, 0.0, 1.0 if c["close"] > 6 else 0.0, 0.0, 0.0, 0.0, 1.0]   # close, then lift while closed
            elif abs(x - tx) > 0.01 or abs(y - ty) > 0.01:
                a = [float(np.clip((tx - x) / 0.02, -1, 1)), float(np.clip((ty - y) / 0.02, -1, 1)), 0.0, 0, 0, 0, -1.0]
            elif z > 0.955:
                a = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]
            else:
                c["close"] += 1
                a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        else:  # h1
            tx, ty = OBJ["center"]
            held = 0.01 < w < 0.06
            c["held"] = c["held"] + 1 if held else c["held"]
            if c["held"] > 12 or c.get("released"):
                c["released"] = True
                a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            elif held or c["close"] > 0:
                c["close"] += 1
                a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            elif abs(x - tx) > 0.01 or abs(y - ty) > 0.01:
                a = [float(np.clip((tx - x) / 0.02, -1, 1)), float(np.clip((ty - y) / 0.02, -1, 1)), 0.0, 0, 0, 0, -1.0]
            elif z > 0.955:
                a = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]
            else:
                c["close"] += 1
                a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.log.append({"session": sid, "seeds": list(seeds), "sampler": sampler, "action": a})
        return [(np.array([a], dtype=self.dtype), {"seed": int(x)}) for x in seeds]
