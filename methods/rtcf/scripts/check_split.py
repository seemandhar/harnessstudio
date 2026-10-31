#!/usr/bin/env python3
"""Assert that the rtcf development memory never touches an evaluation initial state.

    python3 methods/rtcf/scripts/check_split.py PREPARED_DIR_OR_JSON EVAL_RUN_DIR [EVAL_RUN_DIR ...] [--out R.json]

Case keys differ by namespace (dev: 'libero', eval: 'libero@seed10X'), so HarnessStudio's automatic refusal of
development cases cannot catch a collision of the same LIBERO init state under another seed. This check compares
(task, init-state index) pairs: dev pairs from prepared.json dev_cases, evaluation pairs from every trace record
(<run>/<arm>/traces/*.json) of the given evaluation runs. Exit 1 on any overlap. A sabotage check is
built in: `--self-test` must report an overlap on a synthetic colliding input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def task_part(task: str) -> str:
    return str(task)          # includes suite/name#task_id as HarnessStudio keys it


def dev_pairs(prepared: Path) -> set:
    p = prepared / "prepared.json" if prepared.is_dir() else prepared
    data = json.loads(p.read_text())
    return {(task_part(t), int(e)) for (_bench, t, e) in data["dev_cases"]}


def eval_pairs(run: Path) -> set:
    out = set()
    for f in run.glob("*/traces/*.json"):
        r = json.loads(f.read_text())
        out.add((task_part(r["task"]), int(r["episode_idx"])))
    return out


def check(dev: set, ev: set) -> dict:
    overlap = sorted(dev & ev)
    return {"ok": not overlap and bool(dev) and bool(ev), "dev_pairs": len(dev), "eval_pairs": len(ev),
            "overlap": [list(x) for x in overlap[:50]], "overlap_count": len(overlap),
            "dev_init_states": sorted({e for _, e in dev}), "eval_init_states": sorted({e for _, e in ev})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared", nargs="?")
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        bad = check({("libero_10/x#0", 3)}, {("libero_10/x#0", 3)})
        good = check({("libero_10/x#0", 13)}, {("libero_10/x#0", 3)})
        ok = (not bad["ok"]) and good["ok"]
        print(json.dumps({"self_test_ok": ok}))
        return 0 if ok else 1
    report = check(dev_pairs(Path(a.prepared)), set().union(*(eval_pairs(Path(r)) for r in a.runs)))
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
