"""SimplerEnv Google Robot official grids (visual matching + variant-aggregation base scenes) with proprioceptive
`states` and the OpenVLA SimplerEnv post-processing (EXT openvla-residual-hook, docs/extensions/openvla-residual-hook.md).

Cases = the official SimplerEnv enumeration of simpler-env/SimplerEnv@eddb5690749a1e34267807c0f7b5eede323918d7
`scripts/rt1_{pick_coke_can,move_near,drawer,put_in_drawer}_{visual_matching,variant_agg}.sh` (VA: only the
"# base setup" block of each script), evaluated as the official evaluator does
(simpler_env/evaluation/maniskill2_evaluator.py:40-131, argparse.py:120-130, env_builder.py:5-18, utils/io.py:66-78):

  - env: gym.make(env_name, **build_kwargs, obs_mode="rgbd", robot="google_robot_static", sim_freq=513,
    control_mode=<google static planner mode>, control_freq=3, max_episode_steps=H, scene_name,
    camera_cfgs={"add_segmentation": True}, rgb_overlay_path=<overlay or None>) plus rgb_overlay_cameras=
    ["overhead_camera"] when an overlay is used; `--enable-raytracing` puts shader_dir="rt" FIRST in the build kwargs;
    `key=value` build kwargs keep the DictAction types (the string "None", booleans True).
  - reset(options={"robot_init_options": {"init_xy", "init_rot_quat"}, "obj_init_options": {"init_xy"} |
    {"episode_id"}}), no seed; init_rot_quat = (Pose(q=euler2quat(0, 0, yaw)) * Pose(q=(0, 0, 0, 1))).q (wxyz).
  - case order = the script loops, inside one invocation robot x -> robot y -> yaw -> object x -> object y (or episode).
  - one fresh env per case; the episode runs to the TimeLimit (OpenVLA never predicts termination); the instruction
    is re-read after every step (Place in Drawer force-advances its subtask when `_elapsed_steps >= 100` BEFORE a step,
    place_in_closed_drawer_in_scene.py:244-247, so the place instruction is first observed after step 101) and the
    OpenVLA gripper state is reset on a change (inherited from SimplerSeededBenchmark.step).

Inherited unchanged from harnessstudio.benchmarks.simpler_seeded: GoogleAuthorsPostprocess (euler2axangle + relative
15-step sticky gripper), the per-reset GPU-UUID/NVIDIA-ICD isolation check, step(), check_done() (never stop on
success). Metrics: `final_success` (last step; the official evaluator's score, SETTING.md primary), `success` (any
step), `first_success_step`, `steps`, `horizon_ok` (steps == the official horizon; an episode with horizon_ok False
was truncated by a config `max_steps` and is invalid, review critic-ext-1 R13c).

Grids (case ids "vm<index>" / "va<index>"):
  vm_pick_coke_can 300, vm_move_near_v0 240, vm_open_drawer 108, vm_place_in_drawer 108,
  va_open_drawer 27, va_close_drawer 27, va_place_in_drawer 27, va_move_near 60, va_pick_coke_can 75.
REFUSED: vm_close_drawer (its 108 configurations are the pool of the PCD discovery run's seeded Close Drawer protocol).
`states` = agent.eef_pos [pos3 robot-base frame, quat_wxyz4, gripper1] (vla-eval make_obs with send_state=True).
The constructor has no **kwargs (server observation_params cannot inject a seed, P-20261103-07). Needs only vla-eval
and numpy at import; the simulator modules are imported lazily.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepResult
from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark
from vla_eval.render import assert_lavapipe_vulkan
from vla_eval.types import EpisodeResult, Task

from harnessstudio.benchmarks.simpler_seeded import GoogleAuthorsPostprocess, SimplerSeededBenchmark

PROTOCOL = "simplerenv_eddb569_official_grids_openvla_post"
GOOGLE_CONTROL_MODE = ("arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_"
                       "interpolate_by_planner")                         # env_builder.py:20-24 (google_robot_static)
# urdf "None" is the STRING the official DictAction yields for `urdf_version=None` (utils/io.py:66-78); base_env.py
# maps None and "None" to "" alike. Kept as the string for byte-level fidelity of the build kwargs.
URDF_COKE_MOVE = ("None", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2",
                  "recolor_cabinet_visual_matching_1")                     # rt1_pick_coke_can/move_near VM order
URDF_DRAWER = ("recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1",
               "recolor_tabletop_visual_matching_2", "None")              # rt1_drawer/put_in_drawer VM order
DRAWER_OVERLAYS = (  # overlay name, robot x, robot y, yaw (rt1_drawer_visual_matching.sh EvalOverlay, in order)
    ("open_drawer_a0", 0.644, -0.179, -0.03), ("open_drawer_a1", 0.765, -0.182, -0.02),
    ("open_drawer_a2", 0.889, -0.203, -0.06), ("open_drawer_b0", 0.652, 0.009, 0.0),
    ("open_drawer_b1", 0.752, 0.009, 0.0), ("open_drawer_b2", 0.851, 0.035, 0.0),
    ("open_drawer_c0", 0.665, 0.224, 0.0), ("open_drawer_c1", 0.765, 0.222, -0.025),
    ("open_drawer_c2", 0.865, 0.222, -0.025))
PLACE_OVERLAYS = (DRAWER_OVERLAYS[0], DRAWER_OVERLAYS[3], DRAWER_OVERLAYS[6])   # rt1_put_in_drawer_visual_matching.sh
DRAWER_VM_KWARGS = {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True}
COKE_OPTIONS = ("lr_switch", "upright", "laid_vertically")
OPEN_ENVS = ("OpenTopDrawerCustomInScene-v0", "OpenMiddleDrawerCustomInScene-v0", "OpenBottomDrawerCustomInScene-v0")
CLOSE_ENVS = ("CloseTopDrawerCustomInScene-v0", "CloseMiddleDrawerCustomInScene-v0", "CloseBottomDrawerCustomInScene-v0")

HORIZONS = {"vm_pick_coke_can": 80, "vm_move_near_v0": 80, "vm_open_drawer": 113, "vm_place_in_drawer": 200,
            "va_open_drawer": 113, "va_close_drawer": 113, "va_place_in_drawer": 200, "va_move_near": 80,
            "va_pick_coke_can": 80}
GRIDS = tuple(HORIZONS)
REFUSED = {"vm_close_drawer": "overlaps the PCD discovery run's seeded Close Drawer pool (runs/star-port-20261104 "
                              "SETTING.md section 3.4 b)"}


def _linspace(a, b, n):
    return [float(v) for v in np.linspace(a, b, int(n))]


def _qmult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def robot_quat(yaw: float) -> list:
    """(Pose(q=euler2quat(0, 0, yaw)) * Pose(q=[0, 0, 0, 1])).q in wxyz (argparse.py:123-126); euler2quat(0, 0, y)
    (transforms3d 'sxyz') = (cos y/2, 0, 0, sin y/2); sapien Pose products compose the quaternions q1 * q2."""
    return [float(v) for v in _qmult([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], [0.0, 0.0, 0.0, 1.0])]


def _case(env_name, scene, build, robot_xy, yaw, obj, overlay=None):
    return {"env_name": env_name, "scene_name": scene, "overlay": overlay, "build_kwargs": build,
            "robot_xy": [float(robot_xy[0]), float(robot_xy[1])], "robot_yaw": float(yaw), "obj": obj}


def enumerate_grid(grid: str) -> list[dict]:
    """The official cases of `grid` in script order; each case is a plain JSON-serialisable dict."""
    if grid in REFUSED:
        raise ValueError(f"{grid} refused: {REFUSED[grid]}")
    cases: list[dict] = []
    coke_xy = list(itertools.product(_linspace(-0.35, -0.12, 5), _linspace(-0.02, 0.42, 5)))
    apple_xy = list(itertools.product(_linspace(-0.08, -0.02, 3), _linspace(-0.02, 0.08, 3)))
    if grid == "vm_pick_coke_can":                     # rt1_pick_coke_can_visual_matching.sh
        for urdf, option in itertools.product(URDF_COKE_MOVE, COKE_OPTIONS):
            for ox, oy in coke_xy:
                cases.append(_case("GraspSingleOpenedCokeCanInScene-v0", "google_pick_coke_can_1_v4",
                                   {option: True, "urdf_version": urdf}, (0.35, 0.20), 0.0, {"init_xy": [ox, oy]},
                                   "google_coke_can_real_eval_1"))
    elif grid == "vm_move_near_v0":                    # rt1_move_near_visual_matching.sh
        for urdf in URDF_COKE_MOVE:
            for ep in range(60):
                cases.append(_case("MoveNearGoogleBakedTexInScene-v0", "google_pick_coke_can_1_v4",
                                   {"urdf_version": urdf}, (0.35, 0.21), -0.09, {"episode_id": ep},
                                   "google_move_near_real_eval_1"))
    elif grid == "vm_open_drawer":                     # rt1_drawer_visual_matching.sh (Open* only)
        for urdf in URDF_DRAWER:
            for env_name in OPEN_ENVS:
                for name, x, y, yaw in DRAWER_OVERLAYS:
                    cases.append(_case(env_name, "dummy_drawer", {"shader_dir": "rt", **DRAWER_VM_KWARGS,
                                       "urdf_version": urdf}, (x, y), yaw, {"init_xy": [0.0, 0.0]}, name))
    elif grid == "vm_place_in_drawer":                 # rt1_put_in_drawer_visual_matching.sh
        for urdf in URDF_DRAWER:
            for name, x, y, yaw in PLACE_OVERLAYS:
                for ox, oy in apple_xy:
                    cases.append(_case("PlaceIntoClosedTopDrawerCustomInScene-v0", "dummy_drawer",
                                       {"shader_dir": "rt", **DRAWER_VM_KWARGS, "urdf_version": urdf,
                                        "model_ids": "baked_apple_v2"}, (x, y), yaw, {"init_xy": [ox, oy]}, name))
    elif grid in ("va_open_drawer", "va_close_drawer"):  # rt1_drawer_variant_agg.sh "# base setup"
        for env_name in (OPEN_ENVS if grid == "va_open_drawer" else CLOSE_ENVS):
            for x, y in itertools.product(_linspace(0.65, 0.85, 3), _linspace(-0.2, 0.2, 3)):
                cases.append(_case(env_name, "frl_apartment_stage_simple", {"shader_dir": "rt"}, (x, y), 0.0,
                                   {"init_xy": [0.0, 0.0]}))
    elif grid == "va_place_in_drawer":                 # rt1_put_in_drawer_variant_agg.sh "# base setup"
        for y in _linspace(-0.2, 0.2, 3):
            for ox, oy in apple_xy:
                cases.append(_case("PlaceIntoClosedTopDrawerCustomInScene-v0", "frl_apartment_stage_simple",
                                   {"shader_dir": "rt", "model_ids": "apple"}, (0.65, y), 0.0, {"init_xy": [ox, oy]}))
    elif grid == "va_move_near":                       # rt1_move_near_variant_agg.sh "# base setup"
        for ep in range(60):
            cases.append(_case("MoveNearGoogleInScene-v0", "google_pick_coke_can_1_v4", {}, (0.35, 0.21), -0.09,
                               {"episode_id": ep}))
    elif grid == "va_pick_coke_can":                   # rt1_pick_coke_can_variant_agg.sh "# base setup"
        for option in COKE_OPTIONS:
            for ox, oy in coke_xy:
                cases.append(_case("GraspSingleOpenedCokeCanInScene-v0", "google_pick_coke_can_1_v4",
                                   {option: True}, (0.35, 0.20), 0.0, {"init_xy": [ox, oy]}))
    else:
        raise ValueError(f"unknown grid {grid!r}; one of {GRIDS}")
    for i, c in enumerate(cases):
        c["index"] = i
        c["robot_quat_wxyz"] = robot_quat(c["robot_yaw"])
    return cases


def _overlay_dir() -> Path:
    import mani_skill2_real2sim

    return Path(mani_skill2_real2sim.__file__).resolve().parents[1] / "data" / "real_inpainting"


def _gym_make(env_name: str, **kwargs):          # module-level seam so CPU tests can substitute a fake env
    import gymnasium as gym
    import simpler_env  # noqa: F401  (registers the ManiSkill2_real2sim envs)

    return gym.make(env_name, **kwargs)


class SimplerVMBenchmark(SimplerSeededBenchmark):
    """One case per official configuration: task_id "vm<index>" / "va<index>", episodes_per_task: 1."""

    def __init__(self, grid: str, cases: list | None = None, gpu_uuid: str | None = None) -> None:
        all_cases = enumerate_grid(grid)          # raises for refused / unknown grids
        idx = list(range(len(all_cases))) if cases is None else [int(i) for i in cases]
        if not idx or len(set(idx)) != len(idx) or any(i < 0 or i >= len(all_cases) for i in idx):
            raise ValueError(f"cases must be unique indices in [0, {len(all_cases)})")
        # SimplerSeededBenchmark.__init__ validates the seeded protocol; keep its step/post-processing/isolation/metrics
        # and call the vla-eval base constructor directly, with send_state=True.
        SimplerEnvBenchmark.__init__(self, task_name=f"google_robot_{grid}", max_episode_steps=HORIZONS[grid],
                                     success_mode="accumulate", send_state=True, seed=None,
                                     deterministic_episodes=False, control_mode=None, gripper_mode="binary")
        self.grid = grid
        self.prefix = grid[:2]
        self.cases = idx
        self._grid_cases = all_cases
        self.seeds: list = []                     # not a seeded protocol
        self.action_post = "openvla_google_authors"
        self.gpu_uuid = gpu_uuid
        self._isolation: dict = {}
        self._post = GoogleAuthorsPostprocess()
        self._trace: list = []
        self._case_now: dict | None = None
        self._reset_kwargs: dict = {}

    def get_tasks(self) -> list[Task]:
        return [{"name": self.task_name, "task_name": self.task_name, "task_id": f"{self.prefix}{i:03d}"}
                for i in self.cases]

    def _index_of(self, task: Task) -> int:
        tid = str(task.get("task_id", ""))
        if not tid.startswith(self.prefix) or not tid[2:].isdigit():
            raise ValueError(f"simpler_vm task without a {self.prefix}<index> task_id: {task!r}")
        return int(tid[2:])

    def make_kwargs(self, case: dict) -> dict:
        kw: dict[str, Any] = dict(case["build_kwargs"])        # shader_dir first (maniskill2_evaluator.py:54-58)
        kw.update(obs_mode="rgbd", robot="google_robot_static", sim_freq=513, control_mode=GOOGLE_CONTROL_MODE,
                  control_freq=3, max_episode_steps=HORIZONS[self.grid], scene_name=case["scene_name"],
                  camera_cfgs={"add_segmentation": True},
                  rgb_overlay_path=None if case["overlay"] is None else str(_overlay_dir() / f"{case['overlay']}.png"))
        if case["overlay"] is not None:
            kw["rgb_overlay_cameras"] = ["overhead_camera"]      # env_builder.py:7-11
        return kw

    @staticmethod
    def reset_options(case: dict) -> dict:
        return {"options": {"robot_init_options": {"init_xy": np.array(case["robot_xy"], dtype=np.float64),
                                                   "init_rot_quat": np.array(case["robot_quat_wxyz"])},
                            "obj_init_options": {k: (np.array(v, dtype=np.float64) if k == "init_xy" else v)
                                                 for k, v in case["obj"].items()}}}

    def reset(self, task: Task) -> Any:
        if self._render_mode == "cpu":
            assert_lavapipe_vulkan(type(self).__name__)
        else:
            self._isolation = self._check_render_isolation()
        i = self._index_of(task)
        if i not in self.cases:
            raise ValueError(f"case {i} is not in this benchmark's case list")
        if int(task.get("episode_idx", 0)) != 0:
            raise ValueError("simpler_vm runs one episode per case task (episodes_per_task: 1)")
        case = self._grid_cases[i]
        kwargs = self.make_kwargs(case)
        if kwargs["rgb_overlay_path"] is not None and not Path(kwargs["rgb_overlay_path"]).is_file():
            raise FileNotFoundError(f"overlay {kwargs['rgb_overlay_path']} missing in this simulator image")
        self._success_seen = False
        self._trace = []
        self._post.reset()
        if self._env is not None:
            self._env.close()
        self._env = _gym_make(case["env_name"], **kwargs)       # fresh env per case
        self._case_now = case
        self._reset_kwargs = self.reset_options(case)
        obs, _ = self._env.reset(**self._reset_kwargs)
        self._task_description = self._instruction()
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        out = dict(super().get_step_result(step_result))
        out["horizon_ok"] = out["steps"] == HORIZONS[self.grid]
        return out

    def get_metric_keys(self) -> dict[str, str]:
        return {"final_success": "mean", "success": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        meta = SimplerEnvBenchmark.get_metadata(self)
        meta.update({"protocol": PROTOCOL, "grid": self.grid, "cases": list(self.cases), "n_grid": len(self._grid_cases),
                     "render_isolation": dict(self._isolation), "action_post": self.action_post,
                     "official_horizon": HORIZONS[self.grid], "send_state": True})
        return meta
