"""LIBERO-RoboHarness (the paper's custom OOD suite, "libero_soma" in the authors' repo) for allenai vla-eval.

The released repo ships the BDDL task files and pre-sampled init states (libero-modified/bddl_files/libero_soma,
init_files/libero_soma) and a patched LIBERO benchmark registry; it does not ship the LIBERO object class for
`red_bowl` (the asset is part of stock LIBERO: assets/stable_scanned_objects/red_bowl). This module adds, at
runtime and only inside the benchmark container:
  - RedBowl, registered exactly like stock LIBERO's WhiteBowl (GoogleScannedObject, obj_name "red_bowl")
  - the "libero_soma" benchmark with the authors' pi0.5 task list (libero_suite_task_map.py)
and serves it through vla-eval's LIBEROBenchmark. The instruction sent to the policy is the BDDL `(:language ...)`
text (quotes stripped): LIBERO's file-name-derived Task.language would be e.g. "soma noisy prompt challenge".
Horizon: LeRobot's default for suites it does not list (500 steps), as the authors' LeRobot eval would use.

Mount the two folders at /app/libero/libero/libero/{bddl_files,init_files}/libero_soma and put this folder on
PYTHONPATH (see methods/roboharness/configs/libero_roboharness.yaml).
"""
from __future__ import annotations

from pathlib import Path

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark

# libero-modified/benchmark/libero_suite_task_map.py, "libero_soma" (pi0.5 entries; smolvla/pi0 ones commented out)
SOMA_TASKS = ["soma_visual_origin", "soma_visual_challenge", "soma_distractor_challenge",
              "soma_noisy_prompt_challenge", "soma_chain_step_challenge"]
SUITE = "libero_soma"


def register() -> None:
    from libero.libero.envs.base_object import OBJECTS_DICT, register_object
    if "red_bowl" not in OBJECTS_DICT:
        from libero.libero.envs.objects.google_scanned_objects import GoogleScannedObject

        @register_object
        class RedBowl(GoogleScannedObject):
            def __init__(self, name="red_bowl", obj_name="red_bowl"):
                super().__init__(name, obj_name)

    from libero.libero import benchmark as B
    if SUITE in B.BENCHMARK_MAPPING:
        return
    B.task_maps[SUITE] = {t: B.Task(name=t, language=B.grab_language_from_filename(t + ".bddl"), problem="Libero",
                                    problem_folder=SUITE, bddl_file=f"{t}.bddl", init_states_file=f"{t}.init")
                          for t in SOMA_TASKS}

    class LIBERO_SOMA(B.Benchmark):  # noqa: N801 - LIBERO registers by lower-cased class name
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = SUITE
            self.tasks = list(B.task_maps[SUITE].values())
            self.n_tasks = len(self.tasks)

    B.register_benchmark(LIBERO_SOMA)


class LIBERORoboHarness(LIBEROBenchmark):
    def __init__(self, suite: str = SUITE, max_steps: int = 500, **kwargs) -> None:
        super().__init__(suite=suite, max_steps=max_steps, **kwargs)

    def _init_libero(self) -> None:
        if self._task_suite is None:
            register()
        super()._init_libero()

    def get_tasks(self):
        tasks = super().get_tasks()
        from libero.libero import get_libero_path
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        for task in tasks:
            obj = task["task_obj"]
            bddl = Path(get_libero_path("bddl_files")) / obj.problem_folder / obj.bddl_file
            words = robosuite_parse_problem(str(bddl))["language_instruction"]
            text = " ".join(words) if isinstance(words, (list, tuple)) else str(words)
            text = text.strip().strip('"').strip()
            if text:
                task["name"] = text
        return tasks
