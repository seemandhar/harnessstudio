"""`rh eval --quick`: baseline, null and candidate arms on one frozen case list (development evidence only).

No new statistics: records are checked with harnessstudio.bench.evaluation.validate_episode and compared
with harnessstudio.bench.evaluation.assess (descriptive settings); null vs baseline uses the M/R/X rule.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config
from .runs import DEFAULT_LEDGER, DEFAULT_TABLE, case_flags, evaluate, records, write_config

LABEL = "development evidence only: standalone subset screen, not a Campaign gate result"


def expected_cases(run_dir) -> list:
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    return [{"task": c["task"], "trial": c["trial"], "seed": c["seed"], "horizon": c["horizon"]}
            for c in manifest["cases"]]


def describe(candidate, baseline, cases) -> dict:
    """Descriptive only: thresholds are non-binding (min_net_wins=0, inference ratio unbounded);
    'passed' is deliberately dropped. Promotion decisions belong to the Campaign gates."""
    from ..bench.evaluation import assess
    result = assess(candidate, baseline, cases, min_net_wins=0, max_inference_ratio=1e9)
    keep = ("complete", "comparable", "matched_episodes", "contract_mismatches", "reasons",
            "candidate_successes", "baseline_successes", "matched_wins", "matched_losses", "net_wins",
            "paired_p", "inference_ratio", "wall_time_ratio")
    return {k: result.get(k) for k in keep}


def quick(method, *, groups, tasks=None, num_tasks=None, trials=None, gpus="0", table=None,
          ledger=DEFAULT_LEDGER, root=None, policy="gr00t-n1.5") -> dict:
    from ..bench.evaluation import validate_episode
    from .attribution import classify_runs
    root = Path(root or config.runs_dir() / "verify" / method /
                ("quick-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    flags = case_flags(groups, tasks, num_tasks, trials)
    runs = {}
    runs["baseline"] = evaluate(root / "baseline", flags, gpus, ledger=ledger, table=table, policy=policy,
                                config_path=write_config(root / "baseline.config.json", method, "baseline"))
    if runs["baseline"]["status"] == "complete":
        runs["null"] = evaluate(root / "null", flags, gpus, ledger=ledger, table=table, policy=policy,
                                config_path=write_config(root / "null.config.json", method, "null", runs["baseline"]["out"]))
        runs["candidate"] = evaluate(root / "candidate", flags, gpus, ledger=ledger, table=table, policy=policy,
                                     config_path=write_config(root / "candidate.config.json", method, "candidate"))
    report = {"method": method, "label": LABEL, "runs": runs}
    if any(r["status"] != "complete" for r in runs.values()) or len(runs) != 3:
        report.update(complete=False, error="not every arm completed; no comparison reported")
    else:
        recs = {arm: records(r["out"]) for arm, r in runs.items()}
        invalid = []
        for arm, rows in recs.items():
            for row in rows:
                try:
                    validate_episode(row)
                except (ValueError, TypeError, KeyError) as exc:
                    invalid.append({"arm": arm, "case": row.get("case_id"), "error": str(exc)})
        cases = expected_cases(runs["baseline"]["out"])
        report.update(complete=not invalid, invalid_records=invalid,
                      null_vs_baseline=classify_runs(runs["baseline"]["out"], runs["null"]["out"],
                                                     root / "null-attribution"),
                      candidate_vs_baseline=describe(recs["candidate"], recs["baseline"], cases),
                      candidate_vs_null=describe(recs["candidate"], recs["null"], cases),
                      cost={arm: {"policy_calls": sum(r.get("policy_calls", 0) for r in rows),
                                  "policy_seconds": round(sum(r.get("policy_seconds", 0.0) for r in rows), 2),
                                  "successes": sum(bool(r.get("success")) for r in rows), "episodes": len(rows)}
                            for arm, rows in recs.items()})
        report["null_vs_baseline"] = {k: report["null_vs_baseline"][k] for k in ("M", "R", "X", "cases")}
    root.mkdir(parents=True, exist_ok=True)
    (root / "quick_report.json").write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(root / "quick_report.json")
    return report
