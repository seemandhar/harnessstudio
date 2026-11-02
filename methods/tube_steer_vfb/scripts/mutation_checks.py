"""Mutation checks: each mutant of tube-steer-vfb must be caught by the replay comparator (TV vs frozen T on non-t8 dev80
identity episodes, streamed; stops at the first detection) or refused at load. A mutant that survives = FAIL.

M1 fallback on a tube instruction (TV memory emptied -> visual branch for every instruction): replay tripwire/mismatch.
M2 gain cap 1.0 -> 0.9 in TV only: replay mismatch on an armed episode. (A gain-floor mutant is EQUIVALENT: steering
   happens only while D > 7.5 = release, so D/15 > 0.5 and the 0.5 floor never binds; recorded as m2_equivalent_note.)
M3 decision window 4 -> 5 in TV only: replay mismatch.
M4 visual memory sha pin changed: construction refuses.
M5 arming eps 15 -> 14 in TV only: replay mismatch.

Usage: mutation_checks.py --run RUN_DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from harnessstudio.core import registry
from hs_tube_steer_vfb import action_op as A

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from replay_tv_vs_t import Tripwire, compare_episode, dev_episodes  # noqa: E402

IMG = (np.arange(256 * 256 * 3) % 251).astype(np.uint8).reshape(256, 256, 3)


def detect(run, make_v) -> dict:
    n = 0
    for t, init, instr, decs in dev_episodes(run):
        n += 1
        try:
            r = compare_episode("tube-steer", "tube-steer-vfb", instr, decs, IMG, opV=make_v())
        except Tripwire as exc:
            return {"caught": True, "how": f"tripwire: {exc}", "episode": f"t{t}i{init}", "episodes_scanned": n}
        if r["mismatch_decisions"] or not r["stats_equal"]:
            return {"caught": True, "how": f"mismatch at decisions {r['mismatch_decisions'][:5]}", "episode": f"t{t}i{init}",
                    "episodes_scanned": n}
    return {"caught": False, "episodes_scanned": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = dict(registry.load("tube-steer-vfb").card["params"])

    def m1():
        op = A.TubeSteerActionOp(**p)
        op.memory = {}
        return op

    res = {"M1_fallback_on_tube_instruction": detect(a.run, m1),
           "M2_gain_cap_0.9": detect(a.run, lambda: A.TubeSteerActionOp(**{**p, "gain_cap": 0.9})),
           "M3_window_5": detect(a.run, lambda: A.TubeSteerActionOp(**{**p, "window_decisions": 5})),
           "M5_eps_14": detect(a.run, lambda: A.TubeSteerActionOp(**{**p, "eps_deg": 14.0}))}
    A.VisualMemory._cache.clear()
    try:
        A.TubeSteerActionOp(**{**p, "visual_memory_npz_sha256": "0" * 64})
        res["M4_visual_sha_pin"] = {"caught": False}
    except RuntimeError as exc:
        res["M4_visual_sha_pin"] = {"caught": True, "how": str(exc)[:200]}
    ok = all(v["caught"] for v in res.values())
    out = {"check": "mutants of tube-steer-vfb are caught", "pass": ok, "mutants": res,
           "m2_equivalent_note": "gain_floor 0.5 -> 0.4 survived all 72 non-t8 episodes (first run): an equivalent mutant, because steering requires D > release 7.5 deg so clip(D/15, 0.5, 1) never reaches the floor (same in frozen T)."}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
