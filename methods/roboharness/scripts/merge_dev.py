"""Build one RoboHarness memory artifact from several `hs prepare` development shards.

    .venv/bin/python methods/roboharness/scripts/merge_dev.py OUT DEV_DIR [DEV_DIR ...]

Each DEV_DIR is an `hs prepare --out` directory (records in <dir>/candidate/traces/*.json). The merged
prepared.json pins memory.json + build_log.json and forbids every development case of every shard, exactly as a
single `hs prepare` run over all of them would.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from harnessstudio.core.registry import load as get_method
from harnessstudio.eval.prepare import build_prepared


def main() -> int:
    out, dirs = Path(sys.argv[1]), [Path(d) for d in sys.argv[2:]]
    records = []
    for d in dirs:
        traces = sorted((d / "candidate" / "traces").glob("*.json"))
        recs = [json.loads(p.read_text()) for p in traces]
        manifest = json.loads((d / "candidate" / "manifest.json").read_text())
        if not manifest.get("development"):
            raise SystemExit(f"{d} is not a development run")
        records += recs
        print(f"{d}: {len(recs)} records, {sum(bool(r.get('success')) for r in recs)} successes")
    out.mkdir(parents=True, exist_ok=True)
    prepared = build_prepared(get_method("roboharness"), records, out)
    print(json.dumps({k: prepared[k] for k in ("artifacts", "dev_episodes", "dev_valid", "dev_successes")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
