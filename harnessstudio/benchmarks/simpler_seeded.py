"""Seeded SIMPLER benchmark for the PCD authors' protocol (EXT simpler-seeded, docs/extensions/simpler-seeded.md).

Reproduces the evaluation loop of Koorye/PCD @ 31b027e (parallel_inference.py + simpler_env/policies/openvla/
openvla_model.py) on top of vla-eval 0.8.0's SimplerEnvBenchmark:

  (d) reset:       simpler_env.make(task) (prepackaged_config, rgbd), env.reset(seed=s), no options, one case per seed
  (c) actions:     the policy sends the RAW un-normalized OpenVLA action [x, y, z, roll, pitch, yaw, open_gripper]
                   (float64, no gripper sign map). This class applies the authors' post-processing:
                   rot = euler2axangle(roll, pitch, yaw) -> axis * angle; Google relative gripper
                   prev_open - cur_open (0 on the first step) with 15-step stickiness; action_scale 1.0.
  (e) termination: run to the TimeLimit (truncated); success never stops the episode. Metrics: success = any step
                   (Table 1), final_success = last step (Table 9), first_success_step, steps.

Only Google Robot tasks whose instruction is constant are accepted. The constructor deliberately has no **kwargs:
vla-eval forwards a model server's observation_params only to parameters the benchmark constructor names
(orchestrator._merge_observation_params), so the OpenVLA server's {"seed": 0} is dropped instead of collapsing every
case onto one scene.

Needs only vla-eval, numpy and transforms3d (all present in the vla-eval simpler image).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepResult
from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark
from vla_eval.render import assert_lavapipe_vulkan
from vla_eval.specs import GRIPPER_RAW, POSITION_DELTA, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Task

PROTOCOL = "pcd_authors_31b027e_seeded"
# Official horizons (ManiSkill2_real2sim register_env max_episode_steps); the env's TimeLimit enforces them.
GOOGLE_TASKS = {
    "google_robot_pick_coke_can": 80,      # GraspSingleOpenedCokeCanInScene-v0
    "google_robot_move_near": 80,          # MoveNearGoogleBakedTexInScene-v1
    "google_robot_open_drawer": 113,       # OpenDrawerCustomInScene-v0
    "google_robot_close_drawer": 113,      # CloseDrawerCustomInScene-v0
}
STICKY_GRIPPER_NUM_REPEAT = 15            # openvla_model.py:31-33 (google_robot)
ACTION_SCALE = 1.0                        # properties.py OPENVLA_CONFIG action_scale


def seeds_sha256(seeds) -> str:
    return hashlib.sha256(json.dumps([int(s) for s in seeds]).encode()).hexdigest()


class GoogleAuthorsPostprocess:
    """Literal re-derivation of openvla_model.py:128-169 / openvla_contrast.py:61-104 (google_robot branch).

    State is per episode; ``reset`` matches OpenVLAInference.reset (sticky off, counter 0, previous None).
    Input: raw 7-D action (float64). Output: env action [world_vector(3), rot_axangle(3), gripper(1)].
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        from transforms3d.euler import euler2axangle

        raw_actions = np.asarray(raw)[None]
        world_vector = np.array(raw_actions[0, :3])
        rotation_delta = np.array(raw_actions[0, 3:6])
        open_gripper = np.array(raw_actions[0, 6:7])

        world = world_vector * ACTION_SCALE
        roll, pitch, yaw = np.asarray(rotation_delta, dtype=np.float64)
        ax, angle = euler2axangle(roll, pitch, yaw)
        rot = (ax * angle) * ACTION_SCALE

        current = open_gripper
        if self.previous_gripper_action is None:
            relative = np.array([0])
        else:
            relative = self.previous_gripper_action - current
        self.previous_gripper_action = current
        if np.abs(relative) > 0.5 and (not self.sticky_action_is_on):
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative = self.sticky_gripper_action
        if self.gripper_action_repeat == STICKY_GRIPPER_NUM_REPEAT:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        return np.concatenate([world, rot, relative])


class SimplerSeededBenchmark(SimplerEnvBenchmark):
    """One case per seed: tasks are {name: task_name, task_id: "seed<S>"}; run with episodes_per_task: 1."""

    def __init__(self, task_name: str, seeds: list, action_post: str = "openvla_google_authors",
                 gpu_uuid: str | None = None) -> None:
        if task_name not in GOOGLE_TASKS:
            raise ValueError(f"simpler_seeded supports {sorted(GOOGLE_TASKS)}, not {task_name!r}")
        if action_post != "openvla_google_authors":
            raise ValueError(f"unknown action_post {action_post!r}")
        seeds = [int(s) for s in seeds]
        if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
            raise ValueError("seeds must be a non-empty list of unique non-negative ints")
        super().__init__(task_name=task_name, max_episode_steps=None, success_mode="accumulate", send_state=False,
                         seed=None, deterministic_episodes=False, control_mode=None, gripper_mode="binary")
        self.seeds = seeds
        self.action_post = action_post
        self.gpu_uuid = gpu_uuid          # render: gpu only: the leased GPU's UUID (host nvidia-smi), checked per reset
        self._isolation: dict = {}
        self._post = GoogleAuthorsPostprocess()
        self._trace: list = []
        self._seed_now: int | None = None
        self._reset_kwargs: dict = {}

    # -- cases -----------------------------------------------------------------------------------------------
    def get_tasks(self) -> list[Task]:
        return [{"name": self.task_name, "task_name": self.task_name, "task_id": f"seed{s}"} for s in self.seeds]

    @staticmethod
    def _seed_of(task: Task) -> int:
        tid = str(task.get("task_id", ""))
        if not tid.startswith("seed") or not tid[4:].isdigit():
            raise ValueError(f"simpler_seeded task without a seed task_id: {task!r}")
        return int(tid[4:])

    def _make_env(self) -> Any:
        import simpler_env

        return simpler_env.make(self.task_name)

    # -- episode ---------------------------------------------------------------------------------------------
    def _check_render_isolation(self) -> dict:
        """[S-6] render: gpu must see exactly the leased GPU, and Vulkan must be pinned to the NVIDIA ICD only (the
        image also ships lavapipe). Runs on every reset, i.e. on every container launch; any deviation aborts."""
        if not self.gpu_uuid:
            raise RuntimeError("simpler_seeded render: gpu needs the leased GPU's UUID in params.gpu_uuid")
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60)
        uuids = [u.strip() for u in out.stdout.splitlines() if u.strip()]
        if out.returncode != 0 or uuids != [self.gpu_uuid]:
            raise RuntimeError(f"GPU isolation failed: visible {uuids} (rc {out.returncode}), leased {self.gpu_uuid}")
        icds = {k: os.environ.get(k, "") for k in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES")}
        entries = [e for v in icds.values() for e in v.split(os.pathsep) if e]
        if not entries or any(os.path.basename(e) != "nvidia_icd.json" or not os.path.isfile(e) for e in entries):
            raise RuntimeError(f"Vulkan ICD must be pinned to nvidia_icd.json only, got {icds}")
        for e in entries:
            with open(e) as f:
                text = f.read().lower()
            if "lvp" in text or "lavapipe" in text or "nvidia" not in text:
                raise RuntimeError(f"Vulkan ICD {e} is not the NVIDIA driver: {text[:200]}")
        return {"gpu_uuids": uuids, "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"), "icd": icds}

    def reset(self, task: Task) -> Any:
        if self._render_mode == "cpu":
            assert_lavapipe_vulkan(type(self).__name__)
        else:
            self._isolation = self._check_render_isolation()
        seed = self._seed_of(task)
        if seed not in self.seeds:
            raise ValueError(f"seed {seed} is not in this benchmark's seed list")
        if int(task.get("episode_idx", 0)) != 0:
            raise ValueError("simpler_seeded runs one episode per seed task (episodes_per_task: 1)")
        self._success_seen = False
        self._trace = []
        self._post.reset()
        if self._env is not None:
            self._env.close()
        self._env = self._make_env()  # fresh env per case (obs0 equals the authors' reused env, readiness sec. 7)
        self._seed_now = seed
        self._reset_kwargs = {"seed": seed}
        obs, _ = self._env.reset(**self._reset_kwargs)
        self._task_description = self._instruction()
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def _instruction(self) -> str:
        try:
            return self._env.unwrapped.get_language_instruction()
        except AttributeError:
            return self._env.get_wrapper_attr("get_language_instruction")()

    def step(self, action: Action) -> StepResult:
        raw = action.get("actions", action.get("action"))
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw.shape != (7,):
            raise ValueError(f"simpler_seeded expects a raw 7-D OpenVLA action, got shape {raw.shape}")
        env_action = self._post(raw)
        obs, reward, terminated, truncated, info = self._env.step(env_action)
        info["truncated"] = truncated
        # The authors read info["success"] (utils.py summarize); terminated == info["success"] (sapien_env.py:579-580).
        success = bool(info["success"]) if "success" in info else bool(terminated)
        self._trace.append(success)
        if success:
            self._success_seen = True
        # The authors re-read the instruction after every step and reset the policy (sticky state) when it
        # changes (parallel_inference.py:242-246, openvla_model.py:94-96). Supported tasks never change it.
        new_instruction = self._instruction()
        if new_instruction != self._task_description:
            self._task_description = new_instruction
            self._post.reset()
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(reward), done=success, terminated=success,
                                   truncated=bool(truncated), success=success)
        return StepResult(obs=obs, reward=reward, done=success, info=info)

    def check_done(self, step_result: StepResult) -> bool:
        return bool(step_result.info.get("truncated", False))   # never stop on success

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        trace = self._trace
        first = trace.index(True) if True in trace else -1
        return {"success": bool(any(trace)), "final_success": bool(trace[-1]) if trace else False,
                "first_success_step": first, "steps": len(trace)}

    def get_action_spec(self) -> dict[str, DimSpec]:
        """The RAW OpenVLA action this benchmark post-processes: delta xyz, Euler roll/pitch/yaw, raw gripper."""
        return {"position": POSITION_DELTA, "rotation": ROTATION_EULER, "gripper": GRIPPER_RAW}

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "final_success": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta.update({"protocol": PROTOCOL, "seeds": list(self.seeds), "seeds_sha256": seeds_sha256(self.seeds),
                     "render_isolation": dict(self._isolation),
                     "action_post": self.action_post, "official_horizon": GOOGLE_TASKS[self.task_name]})
        return meta
