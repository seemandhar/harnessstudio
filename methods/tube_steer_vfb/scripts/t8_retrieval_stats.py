"""Offline t8 retrieval statistics through the frozen tube-steer-vfb op (A5.2 qualification preview, outcome-free).

Source: the 20 S2 t8 episodes (runs/rtcf-20261031/eval/s2, init 10-29, seed 7; reserved development memory scenes, all
failures, none in the visual memory since t8 has no success). Per decision the op gets the recorded decision-time state
and native chunk (ideation/scripts-r5-memory/s2_desc.npz, extracted from the S2 traces by load_s2.py) and, instead of a
live SigLIP call, the SigLIP descriptor RTCF collect mode recorded for that decision (same encoder pin; injected through
op.descriptor). Streamed one episode at a time. dev80 t8 identity traces store no frames, so a dev t8 statistic is not
computable offline (the tier-2 shadow null measures it live).

Reported: fraction of decisions whose k = 8 retrieved rows contain >= 6 from one task (criterion >= 90%), per-episode
fractions, retrieved-task histogram, arming decisions (open loop on the native trajectory, informational only).

Usage: t8_retrieval_stats.py --s2-desc S2_DESC.npz --s2-meta S2_DESC.json --out OUT.json [--k K]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

from harnessstudio.core import registry

T8 = "put both moka pots on the stove"


def run(s2_desc: str, s2_meta: str, entry: str = "tube-steer-vfb", k: int | None = None) -> dict:
    meta = [m for m in json.load(open(s2_meta)) if int(m["task"]) == 8]
    z = np.load(s2_desc)
    image = np.zeros((256, 256, 3), np.uint8)
    hist, per, n_dec, n_ok = Counter(), [], 0, 0
    for m in meta:
        assert 10 <= int(m["init"]) <= 29
        key = m["key"]
        D, S, C = z[key + "_desc"].astype(np.float64), z[key + "_state"].astype(np.float64), z[key + "_chunk"].astype(np.float64)
        ref = registry.load(entry).slots["action_op"]
        params = dict(ref.params)
        if k is not None:
            params["visual_k"] = int(k)
        from hs_tube_steer_vfb.action_op import TubeSteerActionOp
        op = TubeSteerActionOp(**params)
        it = iter(D / np.linalg.norm(D, axis=1, keepdims=True))
        op.descriptor = lambda _img, _it=it: next(_it)
        ok = dec = 0
        for j in range(len(S)):
            op.transform(np.array(C[j]), {"states": S[j], "task_description": T8, "images": {"agentview": image}}, {})
            v = op.diagnostics()["visual"]
            hist.update(v["top_tasks"])
            dec += 1
            ok += int(v["max_same_task"] >= 6)
        per.append({"key": key, "init": int(m["init"]), "decisions": dec, "frac_ge6_same_task": ok / max(dec, 1),
                    "first_arm": op.stats["first_arm"], "steered_decisions": op.stats["steered_decisions"]})
        n_dec += dec
        n_ok += ok
        del D, S, C
    frac = n_ok / max(n_dec, 1)
    return {"entry": entry, "k": k or 8, "episodes": len(per), "decisions": n_dec, "frac_decisions_ge6_same_task": frac,
            "criterion_ge_0.90": frac >= 0.90, "episodes_with_frac_ge_0.90": sum(p["frac_ge6_same_task"] >= 0.9 for p in per),
            "retrieved_task_histogram": {str(t): int(c) for t, c in sorted(hist.items())},
            "armed_episodes_open_loop": sum(p["first_arm"] is not None for p in per), "per_episode": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-desc", required=True)
    ap.add_argument("--s2-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args(argv)
    out = run(a.s2_desc, a.s2_meta, k=a.k)
    out["source"] = {"s2_desc": a.s2_desc, "s2_meta": a.s2_meta}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_episode"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
