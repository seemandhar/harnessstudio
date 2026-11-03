"""Bitwise checks for the evaluator (review A3, C1, C4, screen arm 3, post-hoc CRN prefix diagnostic).

  python scripts/bitwise_compare.py --ref REF_ARM_DIR --cand CAND_ARM_DIR [--ref-jsonl REF_VLA_EVAL_EPISODES_DIR]
         --out compare.json

REF/CAND_ARM_DIR = an `hs eval` arm folder (OUT/candidate) holding traces/*.json. Per case (task, episode_idx):
full action/obs hash equality, first mismatching step, the candidate's first ELIGIBLE online fire step (wr_trigger
events), and prefix identity (no mismatch before that fire). With --ref-jsonl (the reference run's vla-eval episode
jsonl folder), the offline replay's first eligible fire on the reference trajectory is compared with the online
first fire (A2: must be equal on every scene; a mismatch is infrastructure). Reads no success flag.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hs_witnessed_rescue.triggers import replay_jsonl  # noqa: E402


def traces(d):
    out = {}
    for f in glob.glob(str(Path(d) / "traces" / "*.json")):
        r = json.loads(Path(f).read_text())
        out[(r["task"], int(r["episode_idx"]))] = r
    return out


def first_fire(rec):
    for e in rec.get("events", []):
        if e.get("t") == "event" and e.get("name") == "wr_trigger" and e.get("fire", {}).get("eligible"):
            return int(e["fire"]["step"])
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref-jsonl")
    ap.add_argument("--horizon", type=int, default=520)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R, C = traces(a.ref), traces(a.cand)
    rows, summary = [], {"cases": 0, "missing": [], "unfired": 0, "unfired_bitwise": 0, "fired": 0,
                         "fired_prefix_identical": 0, "online_offline_checked": 0, "online_offline_equal": 0}
    for key in sorted(set(R) | set(C)):
        if key not in R or key not in C:
            summary["missing"].append(list(key))
            continue
        r, c = R[key], C[key]
        ra, ca, ro, co = r["action_sha256"], c["action_sha256"], r["obs_sha256"], c["obs_sha256"]
        mism = next((i for i in range(max(len(ra), len(ca)))
                     if i >= len(ra) or i >= len(ca) or ra[i] != ca[i] or ro[i] != co[i]), None)
        ff = first_fire(c)
        row = {"task": key[0], "episode_idx": key[1], "valid": [r["valid"], c["valid"]], "steps": [len(ra), len(ca)],
               "bitwise": mism is None, "first_mismatch": mism, "first_fire": ff,
               "prefix_identical": mism is None or (ff is not None and mism >= ff)}
        if a.ref_jsonl:
            tid = re.search(r"#(\d+)$", key[0])
            # vla-eval names files by the orchestrator's task index within the run's (possibly filtered) task
            # list: for a one-task shard that is task0000, else the suite order.
            single = len({k[0] for k in R}) == 1
            idx = 0 if single else (int(tid.group(1)) if tid else 0)
            pat = f"task{idx:04d}_ep{key[1]:04d}_*.jsonl"
            files = glob.glob(str(Path(a.ref_jsonl) / "**" / pat), recursive=True)
            if len(files) == 1:
                off = next((x["step"] for x in replay_jsonl(files[0], a.horizon) if x["eligible"]), None)
                row["offline_first_fire"] = off
                summary["online_offline_checked"] += 1
                summary["online_offline_equal"] += int(off == ff)
            else:
                row["offline_first_fire"] = f"jsonl not found uniquely ({len(files)})"
        summary["cases"] += 1
        if ff is None:
            summary["unfired"] += 1
            summary["unfired_bitwise"] += int(mism is None)
        else:
            summary["fired"] += 1
            summary["fired_prefix_identical"] += int(row["prefix_identical"])
        rows.append(row)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
