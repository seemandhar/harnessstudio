"""LIBERO / LIBERO-Pro with metric depth and camera calibration (spec 16.11).

Adds two observation keys:
  depth:  {camera: float32 [H, W] metres}, flipped on both axes exactly like vla-eval's RGB images
  camera: {camera: {"world_to_pixel": [4, 4], "intrinsics": [3, 3], "camera_to_world": [4, 4],
                    "height": H, "width": W, "pixel_flip": "horizontal"}}
Matrices are robosuite's (camera_utils). robosuite pixel (col, row) appears in the published image at
(W-1-col, row): OpenGL renders upside down and vla-eval flips both axes, so the net difference is a
horizontal flip (checked on real LIBERO frames: the projected end effector lands on the fingertips, and
back-projection through depth is within 3-11 mm while the gripper is closed on an object).
harnessstudio.tools.geometry applies the mapping. No object poses or other simulator state are added.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark

try:  # LIBERO-Pro shares LIBERO's renderer path in vla-eval
    from vla_eval.benchmarks.libero_pro.benchmark import LIBEROProBenchmark
except Exception:  # pragma: no cover - optional
    LIBEROProBenchmark = None


class _RGBDMixin:
    def __init__(self, *args: Any, depth_cameras: tuple = ("agentview",), send_camera: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.depth_cameras = tuple(depth_cameras)
        self.send_camera = send_camera

    def reset(self, task):
        import libero.libero.envs as envs
        original = envs.OffScreenRenderEnv
        cameras = self.depth_cameras

        class _WithDepth(original):  # type: ignore[misc, valid-type]
            def __init__(self, **kw):
                kw.setdefault("camera_depths", True)
                kw.setdefault("camera_names", sorted({"agentview", "robot0_eye_in_hand", *cameras}))
                super().__init__(**kw)

        envs.OffScreenRenderEnv = _WithDepth
        try:
            return super().reset(task)
        finally:
            envs.OffScreenRenderEnv = original

    def _sim(self):
        env = self._env
        return getattr(env, "sim", None) or env.env.sim

    def make_obs(self, raw_obs: Any, task) -> dict:
        obs = super().make_obs(raw_obs, task)
        from robosuite.utils import camera_utils as cu
        sim = self._sim()
        depth, camera = {}, {}
        for cam in self.depth_cameras:
            raw = raw_obs.get(f"{cam}_depth")
            if raw is None:
                continue
            metric = np.asarray(cu.get_real_depth_map(sim, raw), dtype=np.float32)
            metric = metric[..., 0] if metric.ndim == 3 else metric
            h, w = metric.shape
            depth[cam] = np.ascontiguousarray(metric[::-1, ::-1])
            if self.send_camera:
                camera[cam] = {
                    "world_to_pixel": np.asarray(cu.get_camera_transform_matrix(sim, cam, h, w), dtype=np.float64),
                    "intrinsics": np.asarray(cu.get_camera_intrinsic_matrix(sim, cam, h, w), dtype=np.float64),
                    "camera_to_world": np.asarray(cu.get_camera_extrinsic_matrix(sim, cam), dtype=np.float64),
                    "height": h, "width": w, "pixel_flip": "horizontal",
                }
        if depth:
            obs["depth"] = depth
        if camera:
            obs["camera"] = camera
        return obs


class LIBERORGBDBenchmark(_RGBDMixin, LIBEROBenchmark):
    pass


if LIBEROProBenchmark is not None:
    class LIBEROProRGBDBenchmark(_RGBDMixin, LIBEROProBenchmark):
        pass
