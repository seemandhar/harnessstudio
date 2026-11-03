#!/usr/bin/env python3
"""Online gpp trigger replay on recorded greedy episodes (observe-only: no intervention), H1-RC1 / H2-RC1 parity.

For every recorded greedy episode (development map, libero_10 tasks 0-9 x init 0-9 ONLY) the frozen online triggers of
rules.py run on the recorded rows exactly as the controller runs them: at decision k the H1 check sees the recorded
action A[k] as the policy's proposal (before A[k] executes); after A[k] the tracker gets (A[k], S[k]) and the H2 check
runs. After a fire the replay keeps observing (no primitive), so later fires are listed too but only the FIRST fire is
exact for a live candidate (its trajectory changes there). Outcomes are not read.

    harnessstudio/.venv/bin/python harnessstudio/methods/gpp/scripts/fire_set.py --entry gpp-core --out FIRESET.json

Output: per scene {first: {module, trigger_at, acts_at}, h1: [...], h2: [...]} and the sorted fire set F_gpp; plus the
sha256 of every greedy jsonl read. `acts_at` = the step whose action the primitive replaces (H1: the vetoed open j;
H2: detect step + 1).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SHARDS = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/scalepp-20261102/eval/devbase/shards")


def load_card_params(entry: str) -> dict:
    import yaml
    path = Path(__file__).resolve().parents[1] / "hs_gpp" / f"paper_{entry.replace('-', '_')}.yaml"
    return yaml.safe_load(path.read_text())["params"]


def replay(actions, states, params: dict, xyz0=None) -> dict:
    from hs_gpp.rules import H1Trigger, H2Trigger, Tracker, proprio
    h2p, h1p, modes = params["h2"], params["h1"], params["modules"]
    h2 = H2Trigger(**{k: h2p[k] for k in ("min_close", "within", "max_fires", "refire_rise")})
    h1 = H1Trigger(**{k: h1p[k] for k in ("min_run", "clamp_w", "settle", "max_rise", "max_descent", "max_rot",
                                          "chain_gap", "chain_lift", "max_fires")})
    tr = Tracker(xyz0)
    out = {"h1": [], "h2": []}
    for k, (a, s) in enumerate(zip(actions, states)):
        if modes["h1"] != "off":
            ev = h1.check(tr, a)
            if ev is not None:
                h1.fired(ev)
                out["h1"].append({"trigger_at": ev["veto"], "acts_at": ev["veto"], "close": ev["close"],
                                  "w": round(ev["w"], 4), "rot": round(ev["rot"], 3)})
        tr.update(a, proprio(s))
        if modes["h2"] != "off":
            ev = h2.check(tr)
            if ev is not None:
                h2.fired(ev)
                out["h2"].append({"trigger_at": ev["detect"], "acts_at": ev["detect"] + 1, "close": ev["close"],
                                  "lat": ev["lat"]})
    cands = [("h1", f) for f in out["h1"]] + [("h2", f) for f in out["h2"]]
    # an H2 detected after action k acts at k+1; H1 at decision j acts at j; H2 has precedence at equal acts_at
    cands.sort(key=lambda mf: (mf[1]["acts_at"], 0 if mf[0] == "h2" else 1))
    out["first"] = ({"module": cands[0][0], **cands[0][1]} if cands else None)
    if out["first"] is not None and out["first"]["acts_at"] >= len(actions):
        out["first_observable"] = False      # would act after the last recorded step
    return out


def greedy_rows(t: int, i: int):
    d = SHARDS / f"baseline-t{t}/baseline/vla_eval/episodes/LIBEROBenchmark_libero_10"
    fs = glob.glob(str(d / f"task0000_ep{i:04d}_*.jsonl"))
    assert len(fs) == 1, fs
    raw = open(fs[0], "rb").read()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return ([r["action"] for r in rows], [r["state"] for r in rows], fs[0], hashlib.sha256(raw).hexdigest())


def fire_table(entry: str = "gpp-core") -> dict:
    params = load_card_params(entry)
    scenes, hashes = {}, {}
    for t in range(10):
        for i in range(10):                     # development map only; never init >= 10
            A, S, path, h = greedy_rows(t, i)
            hashes[path] = h
            scenes[f"t{t}i{i}"] = replay(A, S, params)
    fset = sorted((k for k, v in scenes.items() if v["first"] is not None and v.get("first_observable", True)),
                  key=lambda s: (int(s[1]), int(s[3:])))
    return {"entry": entry, "rule": "online H1-RC1/H2-RC1 (hs_gpp.rules), observe-only replay of the map greedy records",
            "fire_set": fset, "scenes": scenes, "greedy_jsonl_sha256": hashes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="gpp-core")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    table = fire_table(args.entry)
    text = json.dumps(table, indent=1, sort_keys=True)
    Path(args.out).write_text(text)
    print(len(table["fire_set"]), table["fire_set"], hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    sys.exit(main())
