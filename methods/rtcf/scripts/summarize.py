#!/usr/bin/env python3
"""Paired summary of rtcf runs against the (withdrawn) paper table.

    .venv/bin/python methods/rtcf/scripts/summarize.py --arm baseline=RUN[,RUN...] --arm candidate=RUN[,RUN...] \
        [--arm null=RUN,...] [--suite libero_10] [--out S.json]

RUN directories are `hs eval` outputs (<run>/<arm>/traces/*.json); several runs per arm (task shards, seeds) are
merged by case key (benchmark incl. @namespace, task, episode_idx). Outcomes come from the record's `success`
field only (P-18). Reports per-task and total successes with denominators, Wilson 95% CIs, exact two-sided
McNemar on the paired cases (wins/losses), per seed row and pooled, CIs clustered by (task, init state) because the
five seeds are correlated replicates (critic B9), invalid cases per arm, and RTCF activation/clipping statistics.
All paper numbers are WITHDRAWN v2 claims; ours use a disjoint-memory protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

PAPER = {  # WITHDRAWN arXiv:2608.04527v2 Table 1 (500 episodes per cell) and Fig. 6 (Long, 50 per task)
    "libero_10": {"baseline": 61.6, "rtcf": 68.6, "per_task_baseline": [82, 100, 30, 34, 96, 46, 100, 90, 6, 32],
                  "per_task_rtcf": [98, 100, 38, 48, 92, 68, 94, 100, 0, 48]},
    "libero_spatial": {"baseline": 96.4, "rtcf": 97.4}, "libero_object": {"baseline": 97.4, "rtcf": 97.8},
    "libero_goal": {"baseline": 90.0, "rtcf": 89.8},
}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (mid - half), 1), round(100 * (mid + half), 1)]


def clustered(values_by_cluster: dict) -> dict:
    """Mean and 95% CI (normal approx.) treating each (task, init state) as one cluster: the five seeds of a
    cluster are correlated replicates (critic B9), so their mean is the unit of analysis."""
    means = [sum(v) / len(v) for v in values_by_cluster.values() if v]
    k = len(means)
    if k < 2:
        return {"clusters": k, "mean": (means[0] if means else None), "ci95": [None, None]}
    m = sum(means) / k
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (k - 1))
    half = 1.959964 * sd / math.sqrt(k)
    return {"clusters": k, "mean": round(100 * m, 2), "ci95": [round(100 * (m - half), 2), round(100 * (m + half), 2)]}


def load(arm: str, runs: list) -> dict:
    out = {}
    for run in runs:
        for p in sorted((Path(run) / arm).glob("traces/*.json")):
            r = json.loads(p.read_text())
            out[(r["benchmark"], r["task"], int(r["episode_idx"]))] = r
    return out


def task_id(task: str) -> str:
    return task.rsplit("#", 1)[-1] if "#" in task else task


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=RUN[,RUN...]")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--out")
    a = ap.parse_args()
    arms = {}
    for spec in a.arm:
        name, _, runs = spec.partition("=")
        arms[name] = load(name, runs.split(","))
    report = {"suite": a.suite, "paper_withdrawn_v2": PAPER.get(a.suite), "arms": {}, "pairs": {}}
    for name, recs in arms.items():
        valid = {k: r for k, r in recs.items() if r["valid"]}
        per_task = defaultdict(lambda: [0, 0])
        for k, r in valid.items():
            per_task[task_id(k[1])][0] += int(bool(r["success"]))
            per_task[task_id(k[1])][1] += 1
        succ = sum(int(bool(r["success"])) for r in valid.values())
        act = [e["diagnostics"]["action_op"] for r in valid.values() for e in r["events"]
               if e.get("t") == "decision" and "action_op" in (e.get("diagnostics") or {})]
        by_cluster = defaultdict(list)
        for k, r in valid.items():
            by_cluster[(k[1], k[2])].append(int(bool(r["success"])))
        report["arms"][name] = {
            "rate_clustered_by_task_init": clustered(by_cluster),
            "successes": succ, "episodes": len(valid), "rate": round(100 * succ / len(valid), 2) if valid else None,
            "wilson95": wilson(succ, len(valid)), "invalid": len(recs) - len(valid),
            "invalid_reasons": sorted({str(r["invalid_reason"])[:120] for r in recs.values() if not r["valid"]})[:5],
            "per_task": {t: f"{s}/{n}" for t, (s, n) in sorted(per_task.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)},
            "decisions_logged": len(act), "applied": sum(bool(d.get("applied")) for d in act),
            "skips": {s: sum(d.get("skip") == s for d in act)
                      for s in ("empty_bank", "past_memory_end", "memory_decode_failed", "memory_chunk_unusable")},
            "native_decode_failed": sum(bool(d.get("native_decode_failed")) for d in act),
            "clipped_frac_mean": (sum(d.get("clipped_frac", 0.0) for d in act if d.get("applied"))
                                  / max(1, sum(bool(d.get("applied")) for d in act))),
            "limit_clipped_frac_mean": (sum(d.get("limit_clipped_frac", 0.0) for d in act if d.get("applied"))
                                        / max(1, sum(bool(d.get("applied")) for d in act)))}
    if "baseline" in arms:
        base = {k: r for k, r in arms["baseline"].items() if r["valid"]}
        for name, recs in arms.items():
            if name == "baseline":
                continue
            keys = sorted(set(base) & {k for k, r in recs.items() if r["valid"]})
            wins = sum(1 for k in keys if recs[k]["success"] and not base[k]["success"])
            losses = sum(1 for k in keys if base[k]["success"] and not recs[k]["success"])
            same_actions = sum(1 for k in keys if recs[k]["action_sha256"] == base[k]["action_sha256"])
            per_seed = defaultdict(lambda: [0, 0, 0])
            diffs = defaultdict(list)
            for k in keys:
                seed = k[0].partition("@")[2] or "-"
                w = int(bool(recs[k]["success"]) and not base[k]["success"])
                l = int(bool(base[k]["success"]) and not recs[k]["success"])
                per_seed[seed][0] += 1
                per_seed[seed][1] += w
                per_seed[seed][2] += l
                diffs[(k[1], k[2])].append(int(bool(recs[k]["success"])) - int(bool(base[k]["success"])))
            report["pairs"][f"{name}_vs_baseline"] = {
                "paired": len(keys), "wins": wins, "losses": losses, "mcnemar_p_exact": mcnemar(wins, losses),
                "identical_action_sequences": same_actions,
                "per_seed": {sd: {"paired": n, "wins": w, "losses": l, "mcnemar_p_exact": mcnemar(w, l)}
                             for sd, (n, w, l) in sorted(per_seed.items())},
                "difference_pp_clustered_by_task_init": clustered(diffs)}
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
