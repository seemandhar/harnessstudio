"""Per-arm statistics from harness-vla traces: primitive usage, stalls, rejected repeats, stop reasons, LLM cost.

Usage: python trace_stats.py DIR [DIR ...]   (each DIR = an `hs eval --out` dir with <arm>/records.jsonl)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(dirs: list) -> None:
    for d in dirs:
        for arm_dir in sorted(p for p in Path(d).iterdir() if p.is_dir() and (p / "traces").is_dir()):
            rec_file = arm_dir / "records.jsonl"
            if rec_file.exists():   # written when the arm finishes; traces/ holds finished episodes before that
                recs = [json.loads(x) for x in rec_file.read_text().splitlines() if x.strip()]
            else:
                recs = [json.loads(f.read_text()) for f in sorted((arm_dir / "traces").glob("*.json"))]
            arm = arm_dir.name
            prims, stops, cells = Counter(), Counter(), defaultdict(lambda: [0, 0])
            stalls = repeats = turns = vla_lift = vla_calls = 0
            llm = tok = tools = 0
            wall = 0.0
            first_success_steps = []
            for r in recs:
                if not r.get("valid"):
                    stops["INVALID: " + str(r.get("invalid_reason"))[:60]] += 1
                    continue
                cell = r["task"].split("/", 1)[0]
                cells[cell][0] += 1
                cells[cell][1] += bool(r["success"])
                a = r["accounting"]
                llm += a.get("llm_calls", 0)
                tok += a.get("llm_tokens", 0)
                tools += a.get("tool_calls", 0)
                wall += r.get("wall_seconds", 0.0)
                if r["success"]:
                    first_success_steps.append(a.get("steps", 0))
                for e in r.get("events", []):
                    if e.get("t") != "event":
                        continue
                    if e.get("name") == "turn":
                        turns += 1
                        for x in e.get("results", []):
                            act = x["command"].get("action")
                            prims[act] += 1
                            res = x.get("result") or {}
                            stalls += "stalled" in res
                            if act == "vla_act":
                                vla_calls += 1
                                vla_lift += bool(res.get("lift_detected"))
                    elif e.get("name") == "repeat_rejected":
                        repeats += 1
                    elif e.get("name") == "planner_stop":
                        stops[str(e.get("reason", ""))[:40]] += 1
            n = sum(c[0] for c in cells.values())
            print(f"== {d} [{arm}] valid episodes {n}, wall {wall:.0f} s")
            for cell, (m, s) in sorted(cells.items()):
                print(f"   {cell}: {s}/{m} success")
            if n and arm != "baseline":
                print(f"   per episode: LLM calls {llm / n:.1f}, tokens {tok / n:.0f}, perception calls {tools / n:.1f}, "
                      f"turns {turns / n:.1f}")
                print(f"   commands: {dict(prims.most_common())}")
                print(f"   vla_act lift detected {vla_lift}/{vla_calls}; stalled moves {stalls}; repeats rejected {repeats}")
                print(f"   planner stops: {dict(stops.most_common(8))}")
            if first_success_steps:
                print(f"   steps at success: median {sorted(first_success_steps)[len(first_success_steps) // 2]}")


if __name__ == "__main__":
    main(sys.argv[1:])
