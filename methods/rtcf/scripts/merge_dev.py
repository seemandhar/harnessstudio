#!/usr/bin/env python3
"""Build one rtcf memory from several `hs prepare` development shards (task-sharded runs).

    .venv/bin/python methods/rtcf/scripts/merge_dev.py OUT DEV_DIR [DEV_DIR ...] [--method rtcf|rtcf-framenn|rtcf-timedomain]

Each DEV_DIR is an `hs prepare --out` directory (records in <dir>/candidate/traces/*.json, manifest development
true). The merged prepared.json pins memory.npz + memory.json and forbids every development case of every shard,
as a single `hs prepare` over all of them would. The ablation methods need their own prepared.json (it names the
method) built from the same development records: run this again with --method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harnessstudio.core.registry import load as get_method
from harnessstudio.eval.prepare import build_prepared


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--method", default="rtcf")
    a = ap.parse_args()
    records = []
    for d in map(Path, a.dirs):
        manifest = json.loads((d / "candidate" / "manifest.json").read_text())
        if not manifest.get("development"):
            raise SystemExit(f"{d} is not a development run")
        recs = [json.loads(p.read_text()) for p in sorted((d / "candidate" / "traces").glob("*.json"))]
        for r in recs:   # PNGs are kept in the shard traces only; the memory needs descriptors and chunks
            for ev in r.get("events") or []:
                ((ev.get("diagnostics") or {}).get("action_op") or {}).pop("png_b64", None)
        records += recs
        print(f"{d}: {len(recs)} records, {sum(bool(r.get('success')) for r in recs)} successes")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prepared = build_prepared(get_method(a.method), records, out)
    print(json.dumps({k: prepared[k] for k in ("method", "artifacts", "dev_episodes", "dev_valid", "dev_successes")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
