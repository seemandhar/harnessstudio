"""The controller's step emulation is exactly the native policy when nothing intervenes.

    .venv/bin/python methods/roboharness/scripts/check_native_equivalence.py DEV_DIR BASELINE_DIR

DEV_DIR: an `hs prepare` shard (candidate arm in memory-pass mode: perception off, LeRobot-style action queue,
seeded samples); BASELINE_DIR: `hs eval --arms baseline` on the same development config. Every case must have the
same per-step action hashes and the same outcome.
"""
from __future__ import annotations

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
    dev, base = load(Path(sys.argv[1]) / "candidate"), load(Path(sys.argv[2]) / "baseline")
    keys = sorted(set(dev) & set(base))
    same = [k for k in keys if dev[k]["action_sha256"] == base[k]["action_sha256"]
            and dev[k]["success"] == base[k]["success"]]
    diff = [k for k in keys if k not in same]
    print(json.dumps({"cases": len(keys), "identical_actions_and_outcome": len(same), "different": len(diff),
                      "steps_compared": sum(len(dev[k]["action_sha256"]) for k in keys),
                      "first_different": [list(k) for k in diff[:5]]}, indent=1))
    return 0 if keys and not diff else 1


if __name__ == "__main__":
    sys.exit(main())
