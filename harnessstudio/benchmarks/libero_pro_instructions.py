"""LIBERO-Pro with the perturbed instruction actually sent to the policy (vla-eval 0.8.0 workaround).

vla-eval 0.8.0's LIBEROBenchmark sends task["name"] = LIBERO's Task.language, which LIBERO-PRO derives from
the BDDL *file name*. LIBERO-PRO's `lan` (paraphrase) and `task` perturbations keep the original file names and
write the new instruction only into the BDDL `(:language ...)` block, so with the stock class the policy never
receives the paraphrase or the new task. For those two perturbations this subclass sends the BDDL instruction
(parsed by LIBERO's own robosuite_parse_problem); all other suites are unchanged. Found during the 2026-10-31
LIBERO-Pro validation run (hs2-gpu/libero_pro/README.md, P4).

Use in a vla-eval config: benchmark: "harnessstudio.benchmarks.libero_pro_instructions:LIBEROProBDDLInstruction",
with this checkout mounted in the container and on PYTHONPATH (see harnessstudio/benchmarks/__init__.py).
"""
from __future__ import annotations

from pathlib import Path

from vla_eval.benchmarks.libero_pro.benchmark import LIBEROProBenchmark

BDDL_INSTRUCTION_PERTURBATIONS = ("lan", "task")


class LIBEROProBDDLInstruction(LIBEROProBenchmark):
    def get_tasks(self):
        tasks = super().get_tasks()
        if getattr(self, "_perturbation", None) not in BDDL_INSTRUCTION_PERTURBATIONS:
            return tasks
        from libero.libero import get_libero_path
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        for task in tasks:
            obj = task["task_obj"]
            bddl = Path(get_libero_path("bddl_files")) / obj.problem_folder / obj.bddl_file
            words = robosuite_parse_problem(str(bddl))["language_instruction"]
            text = " ".join(words) if isinstance(words, (list, tuple)) else str(words)
            if text.strip():
                task["filename_instruction"] = task["name"]
                task["name"] = text.strip()
        return tasks
