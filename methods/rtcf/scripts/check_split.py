#!/usr/bin/env python3
"""Split, protocol and config checks for rtcf runs (critic review B1, B2, B3, B7).

    python3 methods/rtcf/scripts/check_split.py \
        [--prepared PREP_DIR_OR_JSON] [--dev-run DEV_DIR ...] \
        [--eval RUN_DIR=CONFIG.yaml ...] [--config CONFIG.yaml ...] [--out R.json]
    python3 methods/rtcf/scripts/check_split.py --self-test

Checks (exit 1 if any fails):
- development init states (prepared.json dev_cases) are within [10, 49] and evaluation init states (traces) are
  within [0, 9] for every (suite, task), ignoring namespace and seed; no (task, init state) pair is in both.
  HarnessStudio's own refusal compares case keys, which differ by namespace ('libero' vs 'libero@seed10X'), so it
  cannot catch a same-init-state collision under another seed;
- every config sets quat_no_antipodal: true; dev configs use episode_offset >= 10 with offset+episodes <= 50 and
  match_sequential_resets: true; eval configs use episode_offset 0 and <= 10 episodes per task;
- every run (dev and eval) used execute_horizon 5 (run manifest profile);
- every eval run's case namespace is 'seed<S>' with S == the config's params.seed, and eval output directories are
  distinct.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

CADENCE = 5


def bench_params(config: Path) -> tuple[dict, int]:
    cfg = yaml.safe_load(Path(config).read_text())
    b = (cfg.get("benchmarks") or [{}])[0]
    return dict(b.get("params") or {}), int(b.get("episodes_per_task", 0))


def lint_config(config: Path, problems: list) -> dict:
    params, episodes = bench_params(config)
    name = Path(config).name
    if params.get("quat_no_antipodal") is not True:
        problems.append(f"{name}: quat_no_antipodal is not true")
    off = int(params.get("episode_offset") or 0)
    if off:   # development
        if off < 10 or off + episodes > 50:
            problems.append(f"{name}: development init states {off}..{off + episodes - 1} not within [10, 49]")
        if params.get("match_sequential_resets") is not True:
            problems.append(f"{name}: development config without match_sequential_resets: true")
    else:
        if episodes > 10:
            problems.append(f"{name}: evaluation config runs {episodes} > 10 init states")
    return params


def manifest(run: Path, arm_glob: str = "*") -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(run.glob(f"{arm_glob}/manifest.json"))]


def check_cadence(run: Path, problems: list) -> None:
    mans = manifest(run)
    if not mans:
        problems.append(f"{run}: no manifest.json found")
    for m in mans:
        eh = (m.get("profile") or {}).get("execute_horizon")
        if eh != CADENCE:
            problems.append(f"{run} ({m.get('arm')}): execute_horizon {eh} != {CADENCE}")


def dev_pairs(prepared: Path) -> set:
    p = prepared / "prepared.json" if prepared.is_dir() else prepared
    return {(str(t), int(e)) for (_b, t, e) in json.loads(p.read_text())["dev_cases"]}


def eval_records(run: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(run.glob("*/traces/*.json"))]


def check_pairs(dev: set, ev: set, problems: list) -> dict:
    overlap = sorted(dev & ev)
    if overlap:
        problems.append(f"{len(overlap)} (task, init state) pairs are in development and evaluation, e.g. {overlap[:3]}")
    bad_dev = sorted(e for _, e in dev if not 10 <= e <= 49)
    bad_ev = sorted(e for _, e in ev if not 0 <= e <= 9)
    if bad_dev:
        problems.append(f"development init states outside [10, 49]: {sorted(set(bad_dev))[:10]}")
    if bad_ev:
        problems.append(f"evaluation init states outside [0, 9]: {sorted(set(bad_ev))[:10]}")
    return {"dev_pairs": len(dev), "eval_pairs": len(ev), "overlap": len(overlap),
            "dev_init_states": sorted({e for _, e in dev}), "eval_init_states": sorted({e for _, e in ev})}


def run_checks(a) -> dict:
    problems: list = []
    report: dict = {}
    for c in a.config or []:
        lint_config(Path(c), problems)
    dev = dev_pairs(Path(a.prepared)) if a.prepared else set()
    for d in a.dev_run or []:
        check_cadence(Path(d), problems)
    ev, outs = set(), []
    for spec in a.eval or []:
        run_s, _, cfg_s = spec.partition("=")
        run = Path(run_s).resolve()
        if run in outs:
            problems.append(f"evaluation output directory reused: {run}")
        outs.append(run)
        params = lint_config(Path(cfg_s), problems) if cfg_s else {}
        check_cadence(run, problems)
        recs = eval_records(run)
        if not recs:
            problems.append(f"{run}: no traces")
        for r in recs:
            ns = str(r["benchmark"]).partition("@")[2]
            if cfg_s and ns != f"seed{params.get('seed')}":
                problems.append(f"{run}: case namespace {ns!r} does not match params.seed {params.get('seed')}")
                break
        ev |= {(str(r["task"]), int(r["episode_idx"])) for r in recs}
    if dev or ev:
        report.update(check_pairs(dev, ev, problems) if (dev and ev) else
                      {"dev_pairs": len(dev), "eval_pairs": len(ev), "note": "pair check needs both sides"})
    report["problems"] = problems
    report["ok"] = not problems
    return report


def self_test() -> bool:
    bad: list = []
    check_pairs({("libero_10/x#0", 3)}, {("libero_10/x#0", 3)}, bad)
    good: list = []
    check_pairs({("libero_10/x#0", 13)}, {("libero_10/x#0", 3)}, good)
    return bool(bad) and not good


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared")
    ap.add_argument("--dev-run", action="append")
    ap.add_argument("--eval", action="append", help="RUN_DIR=CONFIG.yaml")
    ap.add_argument("--config", action="append")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        ok = self_test()
        print(json.dumps({"self_test_ok": ok}))
        return 0 if ok else 1
    report = run_checks(a)
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
