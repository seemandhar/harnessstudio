"""`rh reproduce`: compare a HarnessStudio method with a reference implementation's run (M/R/X).

The reference run is any completed run directory produced by another implementation of the same
method in the same trace format (`manifest.json`, `results/CASE/{trace.jsonl,result.json,
completed.json,episode.mp4}`), e.g. the original ChunkTrust v2.2 runner. HarnessStudio runs its
method as the candidate on the reference's exact case list, canonical reset table and GPU placement,
then classifies each pair with the first-divergence rule (reference = parent).

Acceptance: 0 X; R <= 20 per 32 pairs (scaled); at least half of the pairs M. M and R both imply
identical per-decision action/selected_k on the verified prefix.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .. import config
from .runs import DEFAULT_LEDGER, evaluate, write_config


def acceptance(report: dict) -> dict:
    n = report["cases"]
    r_bound = math.floor(20 * n / 32)
    checks = {"zero_X": report["X"] == 0, "R_within_bound": report["R"] <= r_bound,
              "at_least_half_M": report["M"] * 2 >= n, "ri_covers_all_cases": not report.get("ri_missing")}
    return {"passed": all(checks.values()), "checks": checks, "R_bound": r_bound,
            "counts": {k: report[k] for k in ("M", "R", "X", "cases")}}


def reference_flags(reference_run) -> tuple:
    """Case-selection flags, GPUs, workers and canonical table that reproduce the reference's case
    list and placement exactly."""
    m = json.loads((Path(reference_run) / "manifest.json").read_text())
    tasks = list(dict.fromkeys(c["task"] for c in m["cases"]))
    trials = sorted({int(c["trial"]) for c in m["cases"]})
    flags = ["--groups", *m["groups"], "--tasks", *tasks, "--trials", *[str(t) for t in trials]]
    table = m.get("canonical_table_path") or str(config.DEFAULT_TABLE)
    gpus = ",".join(str(g) for g in m["gpu_ids"])
    return flags, gpus, int(m.get("workers_per_gpu", 1)), table, [c["id"] for c in m["cases"]]


def reproduce(method, reference_run, *, gpus=None, ledger=DEFAULT_LEDGER, root=None) -> dict:
    from .attribution import classify_runs
    reference_run = Path(reference_run).resolve()
    flags, ref_gpus, workers, table, ref_cases = reference_flags(reference_run)
    root = Path(root or config.runs_dir() / "verify" / method /
                ("reproduce-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    studio = evaluate(root / "studio", flags, gpus or ref_gpus, ledger=ledger, projected_hours=0.6,
                      config_path=write_config(root / "studio.config.json", method, "candidate"), table=table,
                      workers_per_gpu=workers)
    report = {"method": method, "reference_run": str(reference_run), "studio": studio}
    studio_cases = [c["id"] for c in json.loads((Path(studio["out"]) / "manifest.json").read_text())["cases"]] \
        if (Path(studio["out"]) / "manifest.json").is_file() else []
    if studio["status"] != "complete" or studio_cases != ref_cases:
        report.update(passed=False, error="studio run incomplete or case list differs from the reference")
    else:
        attribution = classify_runs(reference_run, studio["out"], root / "attribution")
        report["attribution"] = {k: attribution[k] for k in ("M", "R", "X", "cases", "ri_missing")}
        report["acceptance"] = acceptance(attribution)
        report["passed"] = report["acceptance"]["passed"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "reproduce_report.json").write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(root / "reproduce_report.json")
    return report
