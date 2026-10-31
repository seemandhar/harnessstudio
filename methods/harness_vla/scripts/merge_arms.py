"""Pair arms that ran as separate `hs eval --arms <arm>` processes (same config, cases and seeds).

Usage: python merge_arms.py OUT baseline=DIR null=DIR candidate=DIR [baseline=DIR2 ...]
Each DIR is an `hs eval --out` directory holding <arm>/records.jsonl. Records of the same arm from several
directories are concatenated (e.g. Goal-T + Goal-S). Writes OUT/summary.json (readable by `hs report OUT`)
and prints per-arm accounting (LLM calls/tokens, SAM3 calls, wall time) and per-case outcomes.
Pairing and M/R/X attribution use harnessstudio.eval.pairing.summarize unchanged.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from harnessstudio.eval.pairing import summarize


def disambiguate(recs: list) -> list:
    """LIBERO-Pro libero_goal_task has two tasks with the same instruction ("put the wine bottle in the bowl"), so
    HarnessStudio's case key (contract, suite/instruction, episode_idx) collides (see the results doc, Core
    requests). Records are appended in episode order and every arm runs the tasks in the same order, so the k-th
    record with a given key is the k-th task carrying that instruction: tag repeats with " [dup k]"."""
    seen: dict = {}
    for r in recs:
        key = (r["benchmark"], r["task"], r["episode_idx"])
        k = seen.get(key, 0)
        seen[key] = k + 1
        if k:
            r["task"] = f"{r['task']} [dup {k}]"
    return recs


def main(argv: list) -> int:
    out = Path(argv[0])
    arms: dict = defaultdict(list)
    for spec in argv[1:]:
        arm, _, d = spec.partition("=")
        path = Path(d) / arm / "records.jsonl"
        arms[arm] += disambiguate([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    summary = summarize(arms)
    extra = {}
    for arm, recs in arms.items():
        acc = [r.get("accounting") or {} for r in recs if r.get("valid")]
        extra[arm] = {"llm_calls": sum(a.get("llm_calls", 0) for a in acc),
                      "llm_tokens": sum(a.get("llm_tokens", 0) for a in acc),
                      "tool_calls": sum(a.get("tool_calls", 0) for a in acc),
                      "method_seconds": round(sum(a.get("method_seconds", 0.0) for a in acc), 1),
                      "wall_seconds": round(sum(r.get("wall_seconds", 0.0) for r in recs if r.get("valid")), 1),
                      "terminations": sorted({a.get("termination", "") for a in acc})}
    summary["accounting_extra"] = extra
    base = {(r["benchmark"], r["task"], r["episode_idx"]): r for r in arms.get("baseline", []) if r["valid"]}
    if "null" in arms and "candidate" in arms:
        null = {(r["benchmark"], r["task"], r["episode_idx"]): r for r in arms["null"] if r["valid"]}
        cand = {(r["benchmark"], r["task"], r["episode_idx"]): r for r in arms["candidate"] if r["valid"]}
        keys = sorted(set(null) & set(cand) & (set(base) if base else set(cand)))
        wins = sum(1 for k in keys if cand[k]["success"] and not null[k]["success"])
        losses = sum(1 for k in keys if null[k]["success"] and not cand[k]["success"])
        summary["candidate_vs_null"] = {"cases": len(keys), "wins": wins, "losses": losses}
    rows = []
    keys = sorted({(r["benchmark"], r["task"], r["episode_idx"]) for recs in arms.values() for r in recs})
    for k in keys:
        row = {"case": f"{k[1]} #{k[2]}"}
        for arm, recs in arms.items():
            r = next((x for x in recs if (x["benchmark"], x["task"], x["episode_idx"]) == k), None)
            row[arm] = None if r is None else ("invalid" if not r["valid"] else ("S" if r["success"] else "F"))
        rows.append(row)
    summary["per_case"] = rows
    # Per cell (suite prefix of the task key) and episode-0-only breakdowns, on cases valid in every arm.
    def _cell(task: str) -> str:
        return task.split("/", 1)[0] if "/" in task else "all"
    valid_keys = set.intersection(*[{(r["benchmark"], r["task"], r["episode_idx"]) for r in recs if r["valid"]}
                                    for recs in arms.values()]) if arms else set()
    breakdown = {}
    for label, pred in [("cell:" + c, (lambda k, c=c: _cell(k[1]) == c)) for c in sorted({_cell(k[1]) for k in valid_keys})] + \
                       [("episode_idx 0 only", lambda k: k[2] == 0)]:
        ks = [k for k in valid_keys if pred(k)]
        row = {"cases": len(ks)}
        for arm, recs in arms.items():
            idx = {(r["benchmark"], r["task"], r["episode_idx"]): r for r in recs}
            row[arm] = sum(bool(idx[k]["success"]) for k in ks)
        if "baseline" in arms and "candidate" in arms:
            idx_b = {(r["benchmark"], r["task"], r["episode_idx"]): r for r in arms["baseline"]}
            idx_c = {(r["benchmark"], r["task"], r["episode_idx"]): r for r in arms["candidate"]}
            row["cand_vs_base_wins"] = sum(1 for k in ks if idx_c[k]["success"] and not idx_b[k]["success"])
            row["cand_vs_base_losses"] = sum(1 for k in ks if idx_b[k]["success"] and not idx_c[k]["success"])
        breakdown[label] = row
    summary["breakdown"] = breakdown
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_case",)}, indent=2)[:4000])
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
