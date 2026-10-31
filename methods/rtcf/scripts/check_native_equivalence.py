#!/usr/bin/env python3
"""rtcf in collection mode (no memory) is exactly the native pi0-FAST policy.

    .venv/bin/python methods/rtcf/scripts/check_native_equivalence.py RUN_DIR [--out R.json]

RUN_DIR: `hs eval rtcf --arms baseline,candidate` WITHOUT --prepared (or with a prepared memory and the null arm:
pass --arms baseline,null). Every common case must have identical per-step action hashes and outcomes.
Exit 1 on any difference or when no case was compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("traces/*.json")):
        r = json.loads(p.read_text())
        out[(r["benchmark"], r["task"], r["episode_idx"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--arms", default="baseline,candidate")
    ap.add_argument("--out")
    a = ap.parse_args()
    ref_arm, other_arm = a.arms.split(",")
    ref, other = load(Path(a.run) / ref_arm), load(Path(a.run) / other_arm)
    keys = sorted(set(ref) & set(other))
    valid = [k for k in keys if ref[k]["valid"] and other[k]["valid"]]
    same = [k for k in valid if ref[k]["action_sha256"] == other[k]["action_sha256"]
            and ref[k]["success"] == other[k]["success"]]
    diff = [k for k in valid if k not in same]
    modes = sorted({(e.get("diagnostics") or {}).get("action_op", {}).get("mode", "?")
                    for k in valid for e in other[k]["events"] if e.get("t") == "decision"})
    report = {"arms": [ref_arm, other_arm], "cases": len(keys), "valid_both": len(valid),
              "identical_actions_and_outcome": len(same), "different": len(diff),
              "steps_compared": sum(len(ref[k]["action_sha256"]) for k in valid),
              "action_op_modes": modes, "first_different": [list(k) for k in diff[:5]],
              "ok": bool(valid) and not diff and len(valid) == len(keys)}
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
