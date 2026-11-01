#!/usr/bin/env python3
"""Paired summary of rtcf runs against the (withdrawn) paper table.

    .venv/bin/python methods/rtcf/scripts/summarize.py --suite libero_10 \
        --arm baseline=RUN[,RUN...] --arm candidate=RUN[,RUN...] [--arm null=RUN,...] \
        [--expected-per-seed 100] [--memory-meta PREP/artifacts/memory.json] [--out S.json]

RUN directories are `hs eval` outputs (<run>/<arm>/traces/*.json); task shards and seeds are merged by case key
(benchmark incl. @namespace, task, episode_idx). The seed is the case namespace (seed101, ...). Outcomes come from
the record's `success` field only (P-18).

Stopping rule (runs/rtcf-20261031/protocol/run-order.md; preflight review R2): a seed is COMPLETE when baseline and
candidate both have exactly --expected-per-seed valid cases (10 tasks x 10 init states) on the same case keys. The
headline (arm rates, pooled McNemar, clustered paired difference, our - paper) uses complete seeds only and the
matched case set; partial seeds are listed separately and never pooled.
Statistics: exact two-sided McNemar per seed and pooled (the pooled test treats correlated seeds as independent;
lead with the clustered difference); 95% CIs clustered by (task, init state), whose seeds are correlated replicates.
Mechanism statistics (R3): activation and skips by reason, clipping, encoder/PMA/op latency median and p95, encoder
calls per logged decision, the post-hoc same-task retrieval fraction (i* -> memory.json task; never visible at
runtime), and the b* advance distribution. Null-arm activation is SHADOW (computed, not executed).
All paper numbers are WITHDRAWN v2 claims; ours use a disjoint-memory protocol. Paper Fig. 6 per-task values were
read from a bar chart (approximate).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

PAPER = {  # WITHDRAWN arXiv:2608.04527v2 Table 1 (500 episodes per cell) and Fig. 6 (Long, 50 per task, approximate)
    "libero_10": {"baseline": 61.6, "rtcf": 68.6, "n": 500,
                  "per_task_baseline_approx": [82, 100, 30, 34, 96, 46, 100, 90, 6, 32],
                  "per_task_rtcf_approx": [98, 100, 38, 48, 92, 68, 94, 100, 0, 48]},
    "libero_spatial": {"baseline": 96.4, "rtcf": 97.4, "n": 500},
    "libero_object": {"baseline": 97.4, "rtcf": 97.8, "n": 500},
    "libero_goal": {"baseline": 90.0, "rtcf": 89.8, "n": 500},
}
Z = 1.959964


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def clustered(values_by_cluster: dict) -> dict:
    """Mean (in percent) and 95% CI treating each (task, init state) as one cluster (mean over its seeds)."""
    means = [sum(v) / len(v) for v in values_by_cluster.values() if v]
    k = len(means)
    if k == 0:
        return {"clusters": 0, "mean": None, "se": None, "ci95": [None, None]}
    m = sum(means) / k
    if k < 2:
        return {"clusters": k, "mean": round(100 * m, 2), "se": None, "ci95": [None, None]}
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (k - 1))
    se = sd / math.sqrt(k)
    return {"clusters": k, "mean": round(100 * m, 2), "se": round(100 * se, 3),
            "ci95": [round(100 * (m - Z * se), 2), round(100 * (m + Z * se), 2)]}


def quantiles(xs: list) -> dict:
    if not xs:
        return {"n": 0, "median": None, "p95": None}
    s = sorted(xs)
    return {"n": len(s), "median": round(s[len(s) // 2], 3), "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3)}


def load(arm: str, runs: list) -> dict:
    out = {}
    for run in runs:
        for p in sorted((Path(run) / arm).glob("traces/*.json")):
            r = json.loads(p.read_text())
            out[(r["benchmark"], r["task"], int(r["episode_idx"]))] = r
    return out


def seed_of(key) -> str:
    return key[0].partition("@")[2] or "-"


def task_id(task: str) -> str:
    return task.rsplit("#", 1)[-1] if "#" in task else task


def arm_stats(recs: dict, keys: list, memory_tasks: list | None, shadow: bool) -> dict:
    valid = [k for k in keys if k in recs and recs[k]["valid"]]
    succ = sum(int(bool(recs[k]["success"])) for k in valid)
    per_task = defaultdict(lambda: [0, 0])
    by_cluster = defaultdict(list)
    for k in valid:
        s = int(bool(recs[k]["success"]))
        per_task[task_id(k[1])][0] += s
        per_task[task_id(k[1])][1] += 1
        by_cluster[(k[1], k[2])].append(s)
    act, same_task, advances = [], [], Counter()
    for k in valid:
        prev = None
        for e in recs[k]["events"]:
            d = (e.get("diagnostics") or {}).get("action_op") if e.get("t") == "decision" else None
            if not d:
                continue
            act.append(d)
            if "i" in d and memory_tasks is not None and 0 <= d["i"] < len(memory_tasks):
                same_task.append(memory_tasks[d["i"]] == k[1])     # both "suite/instruction#task_id"
            if "i" in d:
                if prev is not None and prev[0] == d["i"]:
                    advances[str(d["b"] - prev[1])] += 1
                elif prev is not None:
                    advances["switch_memory"] += 1
                prev = (d["i"], d["b"])
    applied = [d for d in act if d.get("applied")]
    reasons = Counter(d.get("skip") for d in act if d.get("skip"))
    return {
        "successes": succ, "episodes": len(valid), "rate": round(100 * succ / len(valid), 2) if valid else None,
        "rate_clustered_by_task_init": clustered(by_cluster),
        "invalid": sum(1 for k in keys if k in recs and not recs[k]["valid"]),
        "missing": sum(1 for k in keys if k not in recs),
        "per_task": {t: f"{s}/{n}" for t, (s, n) in sorted(per_task.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)},
        "activation_is_shadow": shadow,
        "decisions_logged": len(act), "applied": len(applied),
        "skips": {s: reasons.get(s, 0) for s in ("empty_bank", "past_memory_end", "memory_decode_failed",
                                                 "memory_chunk_unusable")},
        "native_decode_failed": sum(bool(d.get("native_decode_failed")) for d in act),
        "clipped_frac_mean": (sum(d.get("clipped_frac", 0.0) for d in applied) / len(applied)) if applied else 0.0,
        "limit_clipped_frac_mean": (sum(d.get("limit_clipped_frac", 0.0) for d in applied) / len(applied)) if applied else 0.0,
        "latency_ms": {"encoder": quantiles([d["enc_ms"] for d in act if "enc_ms" in d]),
                       "pma": quantiles([d["pma_ms"] for d in act if "pma_ms" in d]),
                       "op": quantiles([d["op_ms"] for d in act if "op_ms" in d])},
        "encoder_calls_per_logged_decision": (sum(1 for d in act if "enc_ms" in d) / len(act)) if act else None,
        "same_task_retrieval_fraction_posthoc": (sum(same_task) / len(same_task)) if same_task else None,
        "b_star_advance_distribution": dict(sorted(advances.items())),
    }


def paired(base: dict, cand: dict, keys: list) -> dict:
    keys = [k for k in keys if k in base and k in cand and base[k]["valid"] and cand[k]["valid"]]
    wins = sum(1 for k in keys if cand[k]["success"] and not base[k]["success"])
    losses = sum(1 for k in keys if base[k]["success"] and not cand[k]["success"])
    diffs = defaultdict(list)
    for k in keys:
        diffs[(k[1], k[2])].append(int(bool(cand[k]["success"])) - int(bool(base[k]["success"])))
    return {"paired": len(keys), "wins": wins, "losses": losses, "mcnemar_p_exact": mcnemar(wins, losses),
            "identical_action_sequences": sum(1 for k in keys if cand[k]["action_sha256"] == base[k]["action_sha256"]),
            "difference_pp_clustered_by_task_init": clustered(diffs)}


def vs_paper(ours: dict, paper_rate: float, n_paper: int) -> dict:
    c = ours.get("rate_clustered_by_task_init") or {}
    if c.get("mean") is None:
        return {}
    se_p = 100 * math.sqrt((paper_rate / 100) * (1 - paper_rate / 100) / n_paper)
    se_o = c.get("se") or 0.0
    d = c["mean"] - paper_rate
    half = Z * math.sqrt(se_o ** 2 + se_p ** 2)
    return {"ours": c["mean"], "ours_n": ours["episodes"], "paper": paper_rate, "paper_n": n_paper,
            "difference_pp": round(d, 2), "ci95": [round(d - half, 2), round(d + half, 2)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=RUN[,RUN...]")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--expected-per-seed", type=int, default=100)
    ap.add_argument("--memory-meta", help="memory.json of the prepared memory (post-hoc retrieval task check)")
    ap.add_argument("--out")
    a = ap.parse_args()
    arms = {}
    for spec in a.arm:
        name, _, runs = spec.partition("=")
        arms[name] = load(name, runs.split(","))
    if "baseline" not in arms or "candidate" not in arms:
        raise SystemExit("need --arm baseline=... and --arm candidate=...")
    memory_tasks = None
    if a.memory_meta:
        memory_tasks = [m["task"] for m in json.loads(Path(a.memory_meta).read_text())["memories"]]
    base, cand = arms["baseline"], arms["candidate"]
    seeds = sorted({seed_of(k) for k in set(base) | set(cand)})
    seed_rows, complete_keys = {}, []
    for sd in seeds:
        bk = {k for k, r in base.items() if seed_of(k) == sd and r["valid"]}
        ck = {k for k, r in cand.items() if seed_of(k) == sd and r["valid"]}
        complete = len(bk) == a.expected_per_seed and bk == ck
        keys = sorted(bk & ck)
        if complete:
            complete_keys += keys
        seed_rows[sd] = {"status": "complete" if complete else "partial",
                         "baseline_valid": len(bk), "candidate_valid": len(ck), "expected": a.expected_per_seed,
                         "baseline": {k2: v for k2, v in arm_stats(base, keys, None, False).items()
                                      if k2 in ("successes", "episodes", "rate", "rate_clustered_by_task_init")},
                         "candidate": {k2: v for k2, v in arm_stats(cand, keys, None, False).items()
                                       if k2 in ("successes", "episodes", "rate", "rate_clustered_by_task_init")},
                         "pair": paired(base, cand, keys)}
    paper = PAPER.get(a.suite, {})
    report = {"suite": a.suite, "paper_withdrawn_v2": paper,
              "headline_seeds": [sd for sd, r in seed_rows.items() if r["status"] == "complete"],
              "partial_seeds": [sd for sd, r in seed_rows.items() if r["status"] == "partial"],
              "headline_note": "complete seeds only, matched case set (run-order.md stopping rule)",
              "seeds": seed_rows, "arms": {}, "pairs": {}, "vs_paper": {}}
    for name, recs in arms.items():
        report["arms"][name] = arm_stats(recs, complete_keys, memory_tasks, shadow=(name == "null"))
    for name in arms:
        if name != "baseline":
            report["pairs"][f"{name}_vs_baseline"] = paired(base, arms[name], complete_keys)
    pc = report["pairs"].get("candidate_vs_baseline")
    if pc:
        pc["per_seed"] = {sd: r["pair"] for sd, r in seed_rows.items() if r["status"] == "complete"}
    if paper:
        report["vs_paper"] = {"baseline": vs_paper(report["arms"]["baseline"], paper["baseline"], paper["n"]),
                              "rtcf": vs_paper(report["arms"]["candidate"], paper["rtcf"], paper["n"])}
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
