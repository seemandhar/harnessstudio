"""LIBERO benchmark for the rtcf package: stock vla-eval LIBEROBenchmark plus two episode/task selectors.

Runs inside the vla-eval LIBERO container (Python 3.8). Both parameters default to off, so with neither set the
benchmark behaves exactly like vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark.

  episode_offset: N  -> run k = 0..episodes_per_task-1 as episode_idx N+k (LIBERO init state N+k). The rtcf
                        development memory uses N = 10 with 40 episodes (init states 10-49), so it never touches the
                        evaluation init states 0-9. HarnessStudio keys the case by this episode_idx (vla-eval's
                        sync runner calls reset(task) before it builds the EPISODE_START payload from the same dict).
  match_sequential_resets: true -> with episode_offset, reproduce the scene of a plain sequential run exactly
                        (same logic as methods/harness_vla/bench_ext): LIBERO places the jointless fixtures (stove,
                        cabinet, rack) at every env.reset() from the env's seeded RNG, so a sequential run's episode k
                        gets the fixture placement of the (k+1)-th reset since the env was built. When a task's env is
                        (re)built, it is reset (episode_idx - 1) extra times before the real reset, so episode k again
                        gets reset k+1 (the burned resets send nothing). Required for the rtcf development configs
                        (critic review B3), so the memory scenes equal those of a stock seed-7 sequential run.
  task_ids: [i, ...] -> keep only these suite task ids (task sharding without spelling out instructions).

Use in a vla-eval config:
  benchmark: "hs_rtcf_bench:LIBEROSelect"
  docker.volumes: <harnessstudio>:/hs:ro ; docker.env: PYTHONPATH=/hs:/hs/methods/rtcf/bench_ext
"""
from __future__ import annotations

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark

_BASE = "_rtcf_base_episode_idx"


class LIBEROSelect(LIBEROBenchmark):
    def __init__(self, *args, episode_offset=0, task_ids=None, match_sequential_resets=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_offset = int(episode_offset or 0)
        self.match_sequential_resets = bool(match_sequential_resets)
        self.task_ids = None if task_ids is None else [int(t) for t in task_ids]
        if self.episode_offset < 0:
            raise ValueError("episode_offset must be >= 0")

    def get_tasks(self):
        tasks = super().get_tasks()
        if self.task_ids is not None:
            missing = sorted(set(self.task_ids) - {t["task_id"] for t in tasks})
            if missing:
                raise ValueError("task_ids %s not in suite %s" % (missing, self.suite))
            tasks = [t for t in tasks if t["task_id"] in self.task_ids]
        return tasks

    def reset(self, task):
        if self.episode_offset:
            base = int(task.setdefault(_BASE, int(task.get("episode_idx", 0))))   # idempotent on a repeated reset
            task["episode_idx"] = base + self.episode_offset
            task["init_state_index"] = task["episode_idx"]
            if task["episode_idx"] >= 50:
                raise ValueError("LIBERO has 50 init states per task; episode_idx %d" % task["episode_idx"])
            if self.match_sequential_resets and (getattr(self, "_env", None) is None
                                                 or getattr(self, "_current_task_id", None) != task.get("task_id")):
                super().reset(task)            # builds the env: reset 1
                for _ in range(int(task["episode_idx"]) - 1):
                    self._env.reset()          # resets 2..k
                task["sequential_resets_matched"] = True
        return super().reset(task)             # reset k+1, as episode k of a sequential run
