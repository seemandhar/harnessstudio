"""Offline trigger replay (review A2) with the package's own frozen trigger code (the same TriggerMonitor.check the
controller runs online). Input: vla-eval episode jsonl files (rows with 'action' and post-step 'state').
Output: JSON {file: fires}. Outcome-free by itself: it never reads success flags; pairing fires with outcomes is the
analyst's step.

  python scripts/trigger_replay.py --out fires.json [--horizon 520] [--params '{"drop": true}'] FILE_OR_GLOB...
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hs_witnessed_rescue.triggers import replay_jsonl  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=520)
    ap.add_argument("--params", default="{}")
    a = ap.parse_args(argv)
    params = json.loads(a.params)
    files = sorted({f for p in a.paths for f in (glob.glob(p) or [p])})
    out = {f: replay_jsonl(f, a.horizon, params) for f in files}
    Path(a.out).write_text(json.dumps({"params": params, "horizon": a.horizon, "fires": out}, indent=1))
    n = sum(1 for v in out.values() if any(x["eligible"] for x in v))
    print(f"{len(files)} episodes, {n} with an eligible fire -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
