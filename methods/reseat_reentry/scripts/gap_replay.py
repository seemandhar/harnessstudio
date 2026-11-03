"""Offline post-release gap replay (review K1 + pilot manifest) with the package's FROZEN trigger code (the same
GapMonitor.check the controller runs online). Input: vla-eval episode jsonl files (rows with post-step 'state').
Output JSON {file: fires}. Reads no success flag (file-name outcomes are left to the analyst).

  python scripts/gap_replay.py --out fires.json [--horizon 520] FILE_OR_GLOB...
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hs_reseat_reentry.triggers_gap import DEFAULTS, replay_jsonl  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=520)
    a = ap.parse_args(argv)
    files = sorted({f for p in a.paths for f in (glob.glob(p) or [p])})
    out = {f: replay_jsonl(f, a.horizon) for f in files}
    src = Path(__file__).resolve().parents[1] / "hs_reseat_reentry" / "triggers_gap.py"
    doc = {"trigger_params": DEFAULTS, "horizon": a.horizon,
           "triggers_gap_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "jsonl_sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}, "fires": out}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"{len(files)} episodes, {sum(1 for v in out.values() if v)} fired -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
