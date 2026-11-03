"""Fake RGB-D two-object gripper benchmark + scripted spy policy for reseat-reentry seam tests (CPU, in-process vla-eval).

Adapted from the witnessed-rescue seam bench (tests/wr_fake_bench.py there, sha256 7a606c71..., itself adapted from
the gpp seam bench): measured OpenVLA/LIBERO position gain 0.0082 m per unit action, finger-width model, top-down
pinhole camera with robosuite-convention calibration published mirrored horizontally (libero_rgbd `pixel_flip`).
Two blocks A and B and a tray; a block is held when the gripper closes with the eef inside its footprint and at/below
its top; it moves with the eef and drops onto the table/tray on release. Success = both blocks on the tray.
Public keys: images {agentview}, depth {agentview}, camera {agentview}, states, task_description (no wrist RGB, like
the libero_rgbd pilot).

SpyPolicy (Markov in states + per-session mode, ignores samplers like FakePolicyServer, logs session/seeds/sampler):
  'gap'      pick A -> tray, release, then idle (open gripper, no motion); if the open eef hovers within 2 cm (xy) of
             B below z 1.10 it picks B -> tray (the "policy executes the grasp after a re-seat" path)
  'gap_late' as 'gap' but idles 100 extra steps at the start (the fire comes too late: steps_left < 150)
  'gap2'     pick A -> tray, idle 170 steps (fire 1), pick B and put it down OFF the tray, idle (fire 2)
  'idle'     small open-gripper motions only (never a carry, never fires)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult

from harnessstudio.policy_ext.fake import FakePolicyServer, fake_action_spec

TABLE = 0.90
HC, F, N = 1.6, 110.0, 128
CX = CY = (N - 1) / 2.0
SIZE = (0.05, 0.04)
TOP = 0.98
OBJS = {"A": (0.10, 0.05), "B": (-0.06, 0.0)}
TRAY = {"center": (-0.12, 0.13), "size": (0.12, 0.12)}
GAIN = 0.0082


def world_to_pixel() -> np.ndarray:
    return np.array([[F, 0.0, -CX, CX * HC], [0.0, -F, -CY, CY * HC], [0.0, 0.0, -1.0, HC], [0.0, 0.0, 0.0, 1.0]])


def pixel_of(xyz, scale: int = 2) -> tuple:
    p = world_to_pixel() @ np.append(np.asarray(xyz, dtype=np.float64), 1.0)
    return int(round(p[1] / p[2] * scale)), int(round(p[0] / p[2] * scale))


def _inside(x, y, center, size, margin=0.0):
    return (np.abs(x - center[0]) <= size[0] / 2 + margin) & (np.abs(y - center[1]) <= size[1] / 2 + margin)


class RGBDTwoBlockBench(StepBenchmark):
    def __init__(self, horizon: int = 400, n_tasks: int = 1, task_prefix: str = "rr_task_",
                 instruction: str = "put both the red block and the green block on the blue tray", **_: Any) -> None:
        super().__init__()
        self.horizon, self.n_tasks, self.task_prefix, self.instruction = horizon, n_tasks, task_prefix, instruction

    def get_tasks(self):
        return [{"name": f"{self.task_prefix}{i}"} for i in range(self.n_tasks)]

    def reset(self, task):
        self.p = np.array([0.0, 0.0, 1.05])
        self.w = 0.08
        self.obj = {k: {"c": np.array(v, dtype=np.float64), "top": TOP, "held": False, "off": np.zeros(3)}
                    for k, v in OBJS.items()}
        self.t = 0
        return self._raw()

    def _raw(self):
        return {"p": self.p.copy(), "w": self.w,
                "obj": {k: {"c": v["c"].copy(), "top": v["top"]} for k, v in self.obj.items()}}

    def _placed(self, k):
        o = self.obj[k]
        return not o["held"] and bool(_inside(o["c"][0], o["c"][1], TRAY["center"], TRAY["size"]))

    def step(self, action):
        a = np.asarray(action["actions"], dtype=np.float64).reshape(-1)
        self.p = self.p + GAIN * a[:3]
        self.p[2] = max(self.p[2], TABLE + 0.01)
        held = [k for k, o in self.obj.items() if o["held"]]
        if a[6] >= 0:
            if not held and self.w > 0.06:
                for k, o in self.obj.items():
                    if _inside(self.p[0], self.p[1], o["c"], SIZE, 0.01) and TABLE < self.p[2] <= o["top"]:
                        o["held"] = True
                        o["off"] = np.array([o["c"][0] - self.p[0], o["c"][1] - self.p[1], o["top"] - self.p[2]])
                        held = [k]
                        break
            self.w = max(0.04 if held else 0.0, self.w - 0.03)
        else:
            self.w = min(0.08, self.w + 0.03)
            if held and self.w > 0.06:
                o = self.obj[held[0]]
                o["held"] = False
                o["top"] = TOP
        for o in self.obj.values():
            if o["held"]:
                o["c"] = self.p[:2] + o["off"][:2]
                o["top"] = self.p[2] + o["off"][2]
        self.t += 1
        success = all(self._placed(k) for k in self.obj)
        return StepResult(obs=self._raw(), reward=0.0, done=success or self.t >= self.horizon,
                          info={"success": success})

    @staticmethod
    def render(raw):
        rows, cols = np.mgrid[0:N, 0:N].astype(np.float64)
        zc_t = HC - TABLE
        xt, yt = (cols - CX) * zc_t / F, -(rows - CY) * zc_t / F
        depth = np.full((N, N), zc_t)
        rgb = np.zeros((N, N, 3), dtype=np.uint8) + 128
        rgb[_inside(xt, yt, TRAY["center"], TRAY["size"])] = (40, 60, 200)
        colors = {"A": (200, 30, 30), "B": (30, 180, 30)}
        for k, o in raw["obj"].items():
            zc = HC - o["top"]
            xo, yo = (cols - CX) * zc / F, -(rows - CY) * zc / F
            box = _inside(xo, yo, o["c"], SIZE) & (zc < depth)
            depth[box] = zc
            rgb[box] = colors[k]
        return rgb, depth.astype(np.float32)

    def make_obs(self, raw, task):
        rgb, depth = self.render(raw)
        p, w = raw["p"], raw["w"]
        states = np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, w / 2, -w / 2], dtype=np.float64)
        cam = {"world_to_pixel": world_to_pixel(), "height": N, "width": N, "pixel_flip": "horizontal"}
        return {"images": {"agentview": np.ascontiguousarray(rgb[:, ::-1])},
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


def _toward(src, dst, gain=1.0 / 0.02):
    return [float(np.clip((dst[i] - src[i]) * gain, -1, 1)) for i in range(len(dst))]


class SpyPolicy(FakePolicyServer):
    def __init__(self, scenario: str = "gap", dtype=np.float32) -> None:
        super().__init__(chunking="harness", seeded=True, chunk_len=1)
        self.scenario, self.dtype = scenario, dtype
        self.log: list = []
        self.sessions: dict = {}

    def _pick(self, s, c, obj_xy, dest=TRAY["center"]):
        """One step of a pick(obj) -> tray -> release routine from the current state; returns (action, done)."""
        x, y, z, w = s[0], s[1], s[2], s[6] - s[7]
        holding = 0.01 < w < 0.06
        if c.get("released"):
            return [0, 0, 0, 0, 0, 0, -1.0], True
        if not holding and not c.get("closing"):
            if math.hypot(x - obj_xy[0], y - obj_xy[1]) > 0.006:
                dz = 1.0 if z < 1.03 else 0.0
                return [*_toward((x, y), obj_xy), dz, 0, 0, 0, -1.0], False
            if z > 0.955:
                return [0, 0, -1.0, 0, 0, 0, -1.0], False
            c["closing"] = True
        if c.get("closing") and not holding and w > 0.004:
            return [0, 0, 0, 0, 0, 0, 1.0], False
        if holding:
            c["closing"] = False
            if z < 1.05 and not c.get("lifted"):
                return [0, 0, 1.0, 0, 0, 0, 1.0], False
            c["lifted"] = True
            if math.hypot(x - dest[0], y - dest[1]) > 0.006:
                return [*_toward((x, y), dest), 0, 0, 0, 0, 1.0], False
            c["released"] = True
            return [0, 0, 0, 0, 0, 0, -1.0], False
        return [0, 0, 0, 0, 0, 0, -1.0], False

    def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
        s = np.asarray(obs["states"], dtype=np.float64)
        sid = ctx.session_id
        st = self.sessions.setdefault(sid, {"mode": "wait" if self.scenario == "gap_late" else "A", "n": 0,
                                            "idle": 0, "A": {}, "B": {}})
        st["n"] += 1
        a = [0, 0, 0, 0, 0, 0, -1.0]
        if self.scenario == "idle":
            a = [0.1 * math.sin(st["n"] / 5.0), 0, 0, 0, 0, 0, -1.0]
        else:
            if st["mode"] == "wait":
                if st["n"] > 100:
                    st["mode"] = "A"
            elif st["mode"] == "A":
                a, done = self._pick(s, st["A"], OBJS["A"])
                if done:
                    st["mode"] = "idle"
            elif st["mode"] == "idle":
                st["idle"] += 1
                near_b = math.hypot(s[0] - OBJS["B"][0], s[1] - OBJS["B"][1]) < 0.02 and s[2] < 1.10
                if (self.scenario in ("gap", "gap_late") and near_b) or (self.scenario == "gap2" and st["idle"] > 170):
                    st["mode"] = "B"
            elif st["mode"] == "B":
                # gap2 puts B down off the tray (no success), so a second post-release gap can fire
                a, done = self._pick(s, st["B"], OBJS["B"], (0.05, -0.10) if self.scenario == "gap2" else TRAY["center"])
                if done:
                    st["mode"] = "done"
        self.log.append({"session": sid, "seeds": list(seeds), "sampler": sampler, "action": a})
        return [(np.array([a], dtype=self.dtype), {"seed": int(x)}) for x in seeds]
