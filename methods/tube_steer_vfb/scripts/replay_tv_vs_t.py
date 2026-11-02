"""CPU open-loop replay: tube-steer-vfb (TV) vs the frozen tube-steer (T), and the two ortho controls, on every NON-t8
dev80 identity episode (runs/rtcfpp-20261101/eval/dev-baseline/shards/identity-*/baseline, init 30-37 development
scenes; vla_eval jsonl states/actions, decision-time state = state[d-1] at step d (state[0] at d = 0), executed rows =
the native 5 rows), streamed one episode at a time (host memory pressure).

Per decision both ops get the identical chunk array and observation; TV additionally gets an agent-view image, and its
encoder is replaced by a tripwire that raises if called (an instruction with a tube must never reach the visual branch).
Checks per decision: output arrays bitwise equal (np.array_equal on float64 and identical object identity class:
both passthrough or both steered), D / armed / steer diagnostics equal; per episode: stats equal.

Usage: replay_tv_vs_t.py --run RUN_DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from harnessstudio.core import registry

PAIRS = (("tube-steer", "tube-steer-vfb"), ("tube-steer-ortho", "tube-steer-vfb-ortho"))


def _task_of_trace(path: str) -> tuple[int, str]:
    with open(path) as f:
        d = json.load(f)
    out = (int(d["episode_idx"]), d["task"])
    del d
    return out


def dev_episodes(run: str, include_t8: bool = False):
    for sh in sorted(glob.glob(os.path.join(run, "eval/dev-baseline/shards/identity-t*-i*"))):
        t, a, _ = map(int, re.search(r"identity-t(\d+)-i(\d+)-(\d+)$", sh).groups())
        if t == 8 and not include_t8:
            continue
        tasks = dict(_task_of_trace(p) for p in sorted(glob.glob(os.path.join(sh, "baseline/traces/*.json"))))
        for f in sorted(glob.glob(os.path.join(sh, "baseline/vla_eval/episodes/*/task*_ep*.jsonl"))):
            init = a + int(re.search(r"_ep(\d+)_", f).group(1))
            instr = tasks[init].split("/", 1)[1].rsplit("#", 1)[0]
            st, ac = [], []
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    st.append(r["state"])
                    ac.append(r["action"])
            st, ac = np.asarray(st, dtype=np.float64), np.asarray(ac, dtype=np.float64)
            decs = [(st[d - 1] if d > 0 else st[0], ac[d:d + 5]) for d in range(0, len(st), 5)]
            yield t, init, instr, decs


class Tripwire(Exception):
    pass


def _trip(_img):
    raise Tripwire("visual branch reached for an instruction with a tube")


def compare_episode(entry_t: str, entry_v: str, instr: str, decs, image: np.ndarray, opT=None, opV=None) -> dict:
    opT = opT or registry.load(entry_t).slots["action_op"].build()
    opV = opV or registry.load(entry_v).slots["action_op"].build()
    opV.descriptor = _trip
    mism, steered = [], 0
    for q, (s, rows) in enumerate(decs):
        cT, cV = np.array(rows, copy=True), np.array(rows, copy=True)
        oT = opT.transform(cT, {"states": s, "task_description": instr}, {})
        oV = opV.transform(cV, {"states": s, "task_description": instr, "images": {"agentview": image}}, {})
        dT, dV = opT.diagnostics(), opV.diagnostics()
        same = (np.array_equal(np.asarray(oT), np.asarray(oV)) and (oT is cT) == (oV is cV)
                and all(dT.get(k) == dV.get(k) for k in ("D", "armed", "steer", "c", "u", "skip", "ortho_sign")))
        steered += int(bool(dT.get("steer")))
        if not same:
            mism.append(q)
    return {"decisions": len(decs), "mismatch_decisions": mism, "stats_equal": opT.stats == opV.stats,
            "steered_decisions_T": steered, "first_arm_T": opT.stats["first_arm"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    image = (np.arange(256 * 256 * 3) % 251).astype(np.uint8).reshape(256, 256, 3)
    per, tot = [], {p[1]: {"episodes": 0, "decisions": 0, "mismatch_episodes": 0, "steered_decisions_T": 0,
                           "armed_episodes_T": 0} for p in PAIRS}
    for t, init, instr, decs in dev_episodes(a.run):
        row = {"task": t, "init": init}
        for et, ev in PAIRS:
            try:
                r = compare_episode(et, ev, instr, decs, image)
            except Tripwire as exc:
                r = {"tripwire": str(exc), "mismatch_decisions": ["tripwire"], "stats_equal": False, "decisions": len(decs),
                     "steered_decisions_T": 0, "first_arm_T": None}
            row[ev] = r
            s = tot[ev]
            s["episodes"] += 1
            s["decisions"] += r["decisions"]
            s["mismatch_episodes"] += int(bool(r["mismatch_decisions"]) or not r["stats_equal"])
            s["steered_decisions_T"] += r["steered_decisions_T"]
            s["armed_episodes_T"] += int(r["first_arm_T"] is not None)
        per.append(row)
        del decs
    ok = all(v["mismatch_episodes"] == 0 and v["episodes"] == 72 for v in tot.values())
    out = {"check": "TV == T (and TV-ortho == T-ortho) bitwise on all non-t8 dev80 identity episodes (open loop)",
           "run": os.path.abspath(a.run), "expected_episodes": 72, "pass": ok, "totals": tot, "episodes": per}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"pass": ok, "totals": tot}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
