"""In-process vla-eval stack for tier 1 and tests: fake benchmark -> (harness server) -> fake policy."""
from __future__ import annotations

import contextlib
import io
from typing import Any, Optional

from vla_eval.api import run, serve_background

FAKE_BENCHMARK = "harnessstudio.verify.fixtures:FakeBenchmark"


def benchmark_config(url: str, *, n_tasks: int = 2, episodes: int = 2, horizon: int = 12,
                     task_order: str = "forward", timeout: float = 60.0,
                     instruction: str = "reach the goal of {name}", task_prefix: str = "fake_task_",
                     duplicate_names: bool = False) -> dict[str, Any]:
    return {
        "server": {"url": url, "timeout": timeout},
        "output_dir": "/tmp/harnessstudio-tier1-unused",
        "benchmarks": [{
            "benchmark": FAKE_BENCHMARK, "mode": "sync", "episodes_per_task": episodes,
            "params": {"n_tasks": n_tasks, "horizon": horizon, "task_order": task_order,
                       "instruction": instruction, "task_prefix": task_prefix,
                       "duplicate_names": duplicate_names},
        }],
    }


def run_fake(url: str, **kwargs) -> list[dict[str, Any]]:
    """Run the fake benchmark against a server URL; returns vla-eval's per-benchmark results."""
    import tempfile
    cfg = benchmark_config(url, **kwargs)
    # Recording on (vla-eval's default): its session ids are what real runs see.
    with tempfile.TemporaryDirectory(prefix="hs-tier1-") as tmp, contextlib.redirect_stdout(io.StringIO()):
        cfg["output_dir"] = tmp
        return run(cfg, docker=False, no_save=False)


def episodes(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten vla-eval results into per-episode dicts."""
    out = []
    for bench in results:
        for task in bench.get("tasks", []):
            for ep in task.get("episodes", []):
                out.append({"task": task.get("task"), **ep})
    return out


def serve(server, port: int = 0):
    return serve_background(server, port=port)
