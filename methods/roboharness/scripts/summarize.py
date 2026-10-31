"""Paired RoboHarness results per LIBERO-Pro cell, plus the paper's Table III rows.

    .venv/bin/python methods/roboharness/scripts/summarize.py EVAL_DIR [--json OUT]

EVAL_DIR holds one `hs eval --out` directory per cell (libero_pro_<suite>_<pert>/{baseline,candidate}/traces).
Cases pair by (benchmark, task, episode_idx); p-values are exact two-sided McNemar (binomial on discordant pairs).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from math import comb
from pathlib import Path

SPATIAL = ["btw(p,r)", "next_to(r)", "center", "on(ck)", "drw_top(cab_w)", "on(r)", "next_to(ck)", "on(stove)",
           "next_to(p)", "on(cab_w)"]          # standard libero_spatial task order (task_id)
OBJECT = ["soup", "cream_cheese", "dressing", "bbq", "ketchup", "tomato", "butter", "milk", "pudding", "juice"]
# Paper Table III (pi0.5, 199 rollouts/task): rows reported per perturbation
TABLE3 = {"swap": {"spatial": ["btw(p,r)", "center", "drw_top(cab_w)", "next_to(p)", "next_to(r)"], "object": ["soup"]},
          "task": {"spatial": ["btw(p,r)", "center", "next_to(ck)", "next_to(p)", "next_to(r)", "on(ck)", "on(r)",
                               "on(stove)", "on(cab_w)"],
                   "object": ["soup", "bbq", "juice", "dressing", "tomato"]}}
PAPER = {"swap": (2.33, 57.2), "task": (0.86, 55.2)}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(arm_dir: Path) -> dict:
    out = {}
    for p in sorted((arm_dir / "traces").glob("*.json")):
        r = json.loads(p.read_text())
        out[(r["benchmark"], r["task"], r["episode_idx"])] = r
    return out


def task_id(task: str) -> int | None:
    m = re.search(r"#(\d+)$", task)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = Path(args.eval_dir)
    report: dict = {"cells": {}, "table3": {}}
    t3 = {pert: Counter() for pert in TABLE3}
    for cell in sorted(root.glob("libero_pro_*")):
        if not cell.is_dir():
            continue
        _, _, suite, pert = cell.name.split("_", 3)
        base, cand = load(cell / "baseline"), load(cell / "candidate")
        keys = sorted(set(base) & set(cand))
        valid = [k for k in keys if base[k].get("valid") and cand[k].get("valid")]
        bs = sum(bool(base[k]["success"]) for k in valid)
        cs = sum(bool(cand[k]["success"]) for k in valid)
        wins = sum(bool(cand[k]["success"]) and not base[k]["success"] for k in valid)
        losses = sum(bool(base[k]["success"]) and not cand[k]["success"] for k in valid)
        llm = [cand[k]["accounting"].get("llm_calls", 0) for k in valid]
        report["cells"][cell.name] = {"paired": len(valid), "baseline": bs, "candidate": cs, "wins": wins,
                                      "losses": losses, "p": round(mcnemar(wins, losses), 4),
                                      "missing": {"baseline_only": len(set(base) - set(cand)),
                                                  "candidate_only": len(set(cand) - set(base))},
                                      "invalid": [k[1] for k in keys if k not in valid],
                                      "llm_calls_mean": round(sum(llm) / max(1, len(llm)), 1)}
        names = SPATIAL if suite == "spatial" else OBJECT
        rows = set(TABLE3.get(pert, {}).get(suite, []))
        for k in valid:
            tid = task_id(k[1])
            if tid is not None and tid < len(names) and names[tid] in rows:
                t3[pert]["n"] += 1
                t3[pert]["baseline"] += bool(base[k]["success"])
                t3[pert]["candidate"] += bool(cand[k]["success"])
                t3[pert]["wins"] += bool(cand[k]["success"]) and not base[k]["success"]
                t3[pert]["losses"] += bool(base[k]["success"]) and not cand[k]["success"]
    for pert, c in t3.items():
        if c["n"]:
            report["table3"][pert] = {**dict(c), "p": round(mcnemar(c["wins"], c["losses"]), 4),
                                      "paper_base_pct": PAPER[pert][0], "paper_roboharness_pct": PAPER[pert][1]}
    print(json.dumps(report, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
