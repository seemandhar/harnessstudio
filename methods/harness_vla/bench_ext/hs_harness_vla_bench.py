"""LIBERO-Pro benchmark for the harness-vla package: RGB-D + camera calibration + the BDDL instruction fix.

Runs inside the vla-eval LIBERO-Pro container (Python 3.8). It only composes two HarnessStudio benchmark
classes, so it adds sensor data only (metric depth and camera calibration; no object poses):
  - harnessstudio.benchmarks.libero_rgbd._RGBDMixin (depth + camera keys, pixel convention validated on LIBERO)
  - harnessstudio.benchmarks.libero_pro_instructions.LIBEROProBDDLInstruction (task/lan instruction fix)

Use in a vla-eval config:
  benchmark: "hs_harness_vla_bench:LIBEROProRGBDInstruction"
  docker.volumes: <harnessstudio-v2>:/hs:ro ; docker.env: PYTHONPATH=/hs:/hs/methods/harness_vla/bench_ext
  params: {depth_cameras: [agentview, robot0_eye_in_hand], ...}
"""
from __future__ import annotations

from harnessstudio.benchmarks.libero_pro_instructions import LIBEROProBDDLInstruction
from harnessstudio.benchmarks.libero_rgbd import _RGBDMixin


class LIBEROProRGBDInstruction(_RGBDMixin, LIBEROProBDDLInstruction):
    """LIBERO-Pro + metric depth/camera calibration, sending the BDDL instruction for task/lan perturbations.

    Two optional episode-selection parameters for the few-shot (memory) mode (both off by default, so the
    zero-shot configs behave exactly as before):
      episode_offset: N   -> run k = 0..episodes_per_task-1 as episode_idx N+k (init state N+k). Used to evaluate
                             on episodes 1-4 only, so the reference scene (episode 0) is never run by construction.
      pin_init_index: I   -> every episode starts from init state I, whatever its episode_idx (repeated attempts
                             on one reference scene, emulating RPent's RESET during exploration). The task payload
                             then carries init_state_index = I.
      match_sequential_resets: true -> with episode_offset, reproduce the scene of a plain sequential run exactly:
                             LIBERO places the jointless fixtures (stove, cabinet, rack) at every env.reset() from
                             the env's seeded RNG, so a sequential run's episode k has the fixture placement of the
                             (k+1)-th reset since the env was built. When a task's env is (re)built, the env is
                             reset (episode_idx - 1) extra times before the real reset, so episode k again gets reset
                             k+1 (the burned resets load no other init state and send nothing). Found 2026-10-31: the
                             first scored few-shot run used offsets without it (see the results doc).
    vla-eval's sync runner calls benchmark.start_episode(task) -> reset(task) before it builds the EPISODE_START
    payload from the same task dict, so the episode_idx written here is the one HarnessStudio keys the case by.
    """

    def __init__(self, *args, depth_cameras=("agentview", "robot0_eye_in_hand"), episode_offset=0,
                 pin_init_index=None, match_sequential_resets=False, **kwargs):
        super().__init__(*args, depth_cameras=tuple(depth_cameras), **kwargs)
        self.episode_offset = int(episode_offset or 0)
        self.pin_init_index = None if pin_init_index is None else int(pin_init_index)
        self.match_sequential_resets = bool(match_sequential_resets)

    def reset(self, task):
        if self.episode_offset:
            task["episode_idx"] = int(task.get("episode_idx", 0)) + self.episode_offset
        if self.pin_init_index is None:
            if self.episode_offset:
                task["init_state_index"] = int(task["episode_idx"])
                if self.match_sequential_resets and (getattr(self, "_env", None) is None
                                                     or getattr(self, "_current_task_id", None) != task.get("task_id")):
                    super().reset(task)            # builds the env: reset 1 (with this episode's own init state)
                    for _ in range(int(task["episode_idx"]) - 1):
                        self._env.reset()          # resets 2..k
                    task["sequential_resets_matched"] = True
            return super().reset(task)             # reset k+1, as episode k of a sequential run
        real = int(task.get("episode_idx", 0))
        task["episode_idx"] = self.pin_init_index  # LIBEROBenchmark.reset indexes init states by episode_idx
        try:
            obs = super().reset(task)
        finally:
            task["episode_idx"] = real
        task["init_state_index"] = self.pin_init_index
        return obs
