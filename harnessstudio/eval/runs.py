"""Launch HarnessStudio backend runs for method arms (GPU-ledger bound)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .. import config

DEFAULT_POLICY = "gr00t-n1.5"
DEFAULT_LEDGER = config.ledger_path()


def backend(policy=DEFAULT_POLICY):
    """(policy module, backend adapter script) for a policy name or alias."""
    from ..policies import adapter
    pa = adapter(policy)
    return pa, config.PACKAGE_DIR / pa.BACKEND_ADAPTER


def default_table(policy=DEFAULT_POLICY):
    """The policy's bundled canonical reset table, or the string 'none'."""
    pa, _ = backend(policy)
    name = getattr(pa, "CANONICAL_TABLE", None)
    return config.DATA_DIR / "canonical_tables" / name if name else "none"


BACKEND_ADAPTER = backend()[1]   # GR00T default (kept for callers that predate multi-policy support)
DEFAULT_TABLE = default_table()


def case_flags(groups, tasks=None, num_tasks=None, trials=None) -> list:
    flags = ["--groups", *groups]
    if tasks:
        flags += ["--tasks", *tasks]
    if num_tasks is not None:
        flags += ["--num-tasks", str(num_tasks)]
    if trials:
        flags += ["--trials", *[str(t) for t in trials]]
    return flags


def write_config(path, method, arm, baseline_run=None, stepping=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "harnessstudio-run/1", "method": method, "arm": arm,
                                "baseline_run": str(baseline_run) if baseline_run else None,
                                "stepping_override": stepping}))
    return path


def evaluate(out, flags, gpus, *, config_path, policy=DEFAULT_POLICY, table=None, extra=(), ledger=DEFAULT_LEDGER,
             projected_hours=0.25, max_hours=1.0, workers_per_gpu=1) -> dict:
    """Run one arm through the policy's backend adapter (table=None: the policy's default table)."""
    from .ledger import Ledger
    led = Ledger(ledger)
    led.require(projected_hours)
    out = Path(out)
    pa, script = backend(policy)
    table = default_table(policy) if table is None else table
    argv = [sys.executable, str(script), "--policy", pa.NAME, "--benchmark", pa.BENCHMARK,
            "--harness", "harnessstudio", "--harness-config", str(config_path), "--canonical-table", str(table),
            *flags, "--gpus", gpus, "--workers-per-gpu", str(workers_per_gpu), "--max-hours", str(max_hours),
            "--output", str(out), *extra]
    proc = subprocess.run(argv, cwd=config.workspace(), capture_output=True, text=True,
                          timeout=int(max_hours * 3600) + 600)
    if out.exists():
        led.add(out)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").is_file() else {}
    return {"out": str(out), "argv": argv, "returncode": proc.returncode, "status": summary.get("status"),
            "summary": {k: summary.get(k) for k in ("status", "successes", "completed_episodes", "expected_episodes")},
            "stdout_tail": proc.stdout[-800:], "stderr_tail": proc.stderr[-800:]}


def records(run_dir) -> list:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    out = []
    for case in manifest["cases"]:
        path = run_dir / "results" / case["id"] / "result.json"
        if path.is_file() and (run_dir / "results" / case["id"] / "completed.json").is_file():
            out.append(json.loads(path.read_text()))
    return out
