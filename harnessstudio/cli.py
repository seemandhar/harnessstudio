"""`rh`: list, verify, run, eval, reproduce, probe-resets (spec §8)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path



def _cmd_list(args) -> int:
    from .core import registry
    from .core.capabilities import Capability
    from .policies import adapter, names
    rows = []
    for name, (config, origin) in sorted(registry.all_methods().items()):
        compatible = {p: frozenset(config.required) <= adapter(p).CAPABILITIES for p in names()}
        if args.policy and not compatible.get(adapter(args.policy).NAME, False):
            continue
        rows.append({"name": name, "version": config.version, "fidelity": config.fidelity.value,
                     "paper": config.paper, "required": sorted(Capability(c).value for c in config.required),
                     "slots": sorted(config.slots), "origin": origin, "compatible": compatible})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            ok = ",".join(p for p, v in r["compatible"].items() if v) or "-"
            print(f"{r['name']:<18} {r['fidelity']:<11} {r['origin']['kind']:<8} policies={ok}  paper={r['paper']}")
    return 0


def _cmd_verify(args) -> int:
    from .verify.tier1 import run_tier1
    report = {"tier1": run_tier1(args.method, args.policy)}
    passed = report["tier1"]["passed"]
    if args.tier == 2 and passed:
        from .verify.tier2 import run_tier2
        report["tier2"] = run_tier2(args.method, args.policy, gpu=args.gpu, ledger=args.ledger)
        passed = report["tier2"]["passed"]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    from . import config
    out = Path(args.report_root or config.runs_dir() / "verify") / args.method / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(dict(report, passed=passed), indent=2, default=str))
    for tier, rep in report.items():
        for c in rep["checks"]:
            print(f"[{tier}] {'PASS' if c['passed'] else 'FAIL'} {c['name']}: {c['detail']}")
    print(("PASSED" if passed else "FAILED") + f" -> {out / 'report.json'}")
    return 0 if passed else 1


def _add_case_flags(s):
    s.add_argument("--groups", nargs="+", default=["atomic_seen"])
    s.add_argument("--tasks", nargs="+")
    s.add_argument("--num-tasks", type=int)
    s.add_argument("--trials", nargs="+", type=int)
    s.add_argument("--gpus", default="0", help="physical GPU ids with no foreign processes, e.g. 0,1,2,3")
    s.add_argument("--canonical-table", type=Path, help="canonical reset table (default: bundled RoboCasa365 atomic_seen dev table)")
    s.add_argument("--ledger", type=Path)


def _table(args):
    return getattr(args, "canonical_table", None)  # None: the policy's default table


def _ledger(args):
    from .eval.runs import DEFAULT_LEDGER
    return args.ledger or DEFAULT_LEDGER


def _cmd_run(args) -> int:
    from . import config
    from .eval.runs import case_flags, evaluate, write_config
    if args.arm == "null" and not args.baseline_run:
        raise SystemExit("--arm null needs --baseline-run (a completed baseline run on the same case)")
    root = config.runs_dir() / "verify" / args.method / ("run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    cfg = write_config(root / "config.json", args.method, args.arm, args.baseline_run)
    result = evaluate(root / args.arm, case_flags(["atomic_seen"] if not args.group else [args.group],
                      [args.task], None, [args.trial]), str(args.gpu), config_path=cfg, table=_table(args),
                      policy=args.policy,
                      ledger=_ledger(args), projected_hours=0.1, max_hours=0.5)
    print(json.dumps({k: result[k] for k in ("out", "status", "summary", "returncode")}, indent=2))
    video = Path(result["out"]) / "results" / f"{args.task}--{args.trial:04d}" / "episode.mp4"
    print("video:", video if video.exists() else "(none)")
    return 0 if result["status"] == "complete" else 1


def _cmd_eval(args) -> int:
    from .eval.quick import quick
    report = quick(args.method, groups=args.groups, tasks=args.tasks, num_tasks=args.num_tasks, trials=args.trials,
                   gpus=args.gpus, table=_table(args), ledger=_ledger(args), policy=args.policy)
    shown = {k: report.get(k) for k in ("label", "complete", "null_vs_baseline", "candidate_vs_baseline",
                                         "candidate_vs_null", "cost", "invalid_records", "error")}
    print(json.dumps(shown, indent=2, default=str))
    print("report:", report["report_path"])
    return 0 if report.get("complete") else 1


def _cmd_reproduce(args) -> int:
    from .eval.reproduce import reproduce
    report = reproduce(args.method, args.reference_run, gpus=args.gpus, ledger=_ledger(args))
    print(json.dumps({k: report.get(k) for k in ("attribution", "acceptance", "passed", "error")}, indent=2, default=str))
    print("report:", report["report_path"])
    return 0 if report.get("passed") else 1


def _cmd_probe_resets(args) -> int:
    """Build a canonical reset table for a new case list (policy-free reset probe)."""
    import subprocess
    from . import config
    from .backends.gr00t_robocasa365 import adapter as ad
    from .bench.registry import csv_words, load_task_registry
    _, cases = ad.build_cases(load_task_registry(config.policybench()), csv_words(args.groups),
                              csv_words(args.tasks), args.num_tasks, None, args.trials)
    out = config.runs_dir() / "probes" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out.parent.mkdir(parents=True, exist_ok=True)
    cases_json = out.parent / (out.name + ".cases.json")
    cases_json.write_text(json.dumps(cases))
    probe = Path(__file__).parent / "backends/gr00t_robocasa365/reset_probe.py"
    code = subprocess.run([sys.executable, str(probe), "--cases-json", str(cases_json), "--gpus", args.gpus,
                           "--out", str(out), "--phase", "development"], cwd=config.workspace()).returncode
    print("canonical table:", out / "canonical_table.json")
    return code


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rh", description="HarnessStudio: training-free harnesses around frozen robot policies")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("list", help="list methods and policy compatibility")
    s.add_argument("--policy")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_list)
    s = sub.add_parser("verify", help="tier-1 (CPU) and optional tier-2 (one GPU) checks")
    s.add_argument("method")
    s.add_argument("--policy", required=True)
    s.add_argument("--tier", type=int, choices=(1, 2), default=1)
    s.add_argument("--gpu", type=int)
    s.add_argument("--ledger", type=Path)
    s.add_argument("--report-root", type=Path)
    s.set_defaults(fn=_cmd_verify)
    s = sub.add_parser("run", help="one episode with video and per-decision trace (development inspection)")
    s.add_argument("method")
    s.add_argument("--policy", default="gr00t-n1.5")
    s.add_argument("--task", required=True)
    s.add_argument("--trial", type=int, required=True)
    s.add_argument("--group")
    s.add_argument("--arm", choices=("candidate", "null", "baseline"), default="candidate")
    s.add_argument("--baseline-run", type=Path)
    s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--canonical-table", type=Path)
    s.add_argument("--ledger", type=Path)
    s.set_defaults(fn=_cmd_run)
    s = sub.add_parser("eval", help="paired baseline / null / candidate screen on a case list")
    s.add_argument("method")
    s.add_argument("--policy", default="gr00t-n1.5")
    s.add_argument("--quick", action="store_true", help="accepted for compatibility; eval is always the quick screen")
    _add_case_flags(s)
    s.set_defaults(fn=_cmd_eval)
    s = sub.add_parser("reproduce", help="compare a method with a reference implementation's run (M/R/X)")
    s.add_argument("method")
    s.add_argument("--reference-run", type=Path, required=True, help="completed run directory of the reference implementation")
    s.add_argument("--gpus", help="default: the reference run's GPUs (keeps placement identical)")
    s.add_argument("--ledger", type=Path)
    s.set_defaults(fn=_cmd_reproduce)
    s = sub.add_parser("probe-resets", help="canonical reset table for a new case list (policy-free)")
    _add_case_flags(s)
    s.set_defaults(fn=_cmd_probe_resets)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, KeyError, RuntimeError) as exc:
        print(f"rh: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
