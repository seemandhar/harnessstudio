"""Case-by-case table: RPent adapted reference run vs the HarnessStudio port on the episode_idx 0 cases.

Usage: python compare_rpent.py RPENT_EPISODES_DIR baseline=DIR candidate=DIR [baseline=DIR candidate=DIR ...]
RPENT_EPISODES_DIR holds <suite>-t<task>-s0/result.json (rpent_ref/launch_rpent_ref.sh). RPent (suite, task t,
seed 0) = vla-eval task index t of that suite, episode_idx 0 (same init state, same task order). The port's task
index is the order in which task names first appear in its records.jsonl (episodes run task by task).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from merge_arms import disambiguate


def main(argv: list) -> None:
    ref_dir = Path(argv[0])
    by_arm: dict = {}
    for spec in argv[1:]:
        arm, _, d = spec.partition("=")
        recs = disambiguate([json.loads(x) for x in (Path(d) / arm / "records.jsonl").read_text().splitlines() if x.strip()])
        for r in recs:
            suite = r["task"].split("/", 1)[0]
            entry = by_arm.setdefault(arm, {}).setdefault(suite, {"order": [], "recs": {}})
            if r["task"] not in entry["order"]:
                entry["order"].append(r["task"])
            entry["recs"][(r["task"], r["episode_idx"])] = r
    rows = []
    for res in sorted(ref_dir.glob("*/result.json")):
        ref = json.loads(res.read_text())
        suite, t = ref["suite"], int(ref["task"])
        port, name = {}, ""
        for arm, suites in by_arm.items():
            entry = suites.get(suite)
            if not entry or t >= len(entry["order"]):
                port[arm] = "?"
                continue
            name = entry["order"][t]
            r = entry["recs"].get((name, 0))
            port[arm] = "?" if r is None or not r["valid"] else ("S" if r["success"] else "F")
        rows.append((suite, t, "S" if ref.get("success") else "F", port, name, ref.get("task_language", "")))
    for suite, t, ok, port, name, lang in rows:
        print(f"{suite:17s} t{t}  rpent_ref={ok}  " + "  ".join(f"{a}={v}" for a, v in port.items())
              + f"  | port task: {name.split('/', 1)[-1]} | rpent language: {lang}")
    n = len(rows)
    print(f"rpent_ref: {sum(r[2] == 'S' for r in rows)}/{n} successes")
    for arm in by_arm:
        print(f"port {arm}: {sum(r[3].get(arm) == 'S' for r in rows)}/{n} on the same cases "
              f"(unmatched {sum(r[3].get(arm) == '?' for r in rows)})")


if __name__ == "__main__":
    main(sys.argv[1:])
