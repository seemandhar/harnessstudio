"""CPU replay of the frozen ReDecode controllers over recorded DEVELOPMENT identity rollouts (dev init 30-37 jsonl and S2
init 10-29 compact arrays; never init 0-9 or 38-49), open loop with a scripted context:

 R1  every retry fails as openpi serves a failure (mean + s_sent): 'redecode' == tilt-guard-v12-z and 'redecode-b' ==
     tilt-guard-v12-bz, bitwise, on every decision of every episode (C-R9);
 R2  episodes without a native zero chunk: 'redecode' == identity (native rows) and 'redecode-b' == tilt-guard-v12-b;
 R3  statistics at the real zero-chunk states: state tokens changed by the +1/-1 bin shift (public openpi digitize
     replica, C-R2), split by rx <= 3.40 vs > 3.40 (C-R6), and the worst-case policy-call ratio per episode.

Alignment as tilt_guard scripts (dev state[5d-1], 5 executed rows; zero-chunk gripper restored to the actions mean).
Usage: replay_redecode.py --run RUN_DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re

import numpy as np

from harnessstudio.core import registry

HARD = {2, 3, 5, 6, 8, 9}


def episodes(run, amean):
    lab = {}
    for line in open(os.path.join(run, "eval/dev-baseline/episodes_identity.tsv")).read().strip().split("\n")[1:]:
        f = line.split("\t")
        lab[(int(f[0]), int(f[1]))] = int(f[2])
    for sh in sorted(glob.glob(os.path.join(run, "eval/dev-baseline/shards/identity-t*-i*"))):
        t, a, _ = map(int, re.search(r"identity-t(\d+)-i(\d+)-(\d+)$", sh).groups())
        for f in sorted(glob.glob(os.path.join(sh, "baseline/vla_eval/episodes/*/task*_ep*.jsonl"))):
            init = a + int(re.search(r"_ep(\d+)_", f).group(1))
            recs = [json.loads(x) for x in open(f)]
            st = np.array([r["state"] for r in recs], dtype=np.float64)
            ac = np.array([r["action"] for r in recs], dtype=np.float64)
            S, N = [], []
            for d in range(0, len(recs), 5):
                s = st[d - 1] if d > 0 else st[0]
                rows = ac[d:d + 5].copy()
                if np.all(np.abs(rows[:, :6] - (amean[:6] + s[:6])) <= 1e-4) and np.all(rows[:, 6] == -1.0):
                    rows[:, 6] = amean[6]
                S.append(s)
                N.append(rows)
            yield "dev", t, init, bool(lab[(t, init)]), S, N
    z = np.load(os.path.join(run, "analysis/s2_compact.npz"))
    for e in json.load(open(os.path.join(run, "analysis/s2_index.json"))):
        k = e["key"]
        S, ch = z[k + "_state"].astype(np.float64), z[k + "_chunk"].astype(np.float64)
        yield "s2", int(e["task"]), int(e["init"]), bool(e["success"]), list(S), list(ch)


class _P:
    def __init__(self, a):
        self.actions = np.asarray(a)
        self.extras = {}


class Ctx:
    def __init__(self, states, natives, retry_fn):
        self.states, self.natives, self.retry_fn = states, natives, retry_fn
        self.j, self.executed, self.records, self.memory = 0, [], [], {}
        self.native_calls = self.retry_calls = 0
        outer = self

        class P:
            async def sample(self, k=1, obs=None, instruction=None, sampler=None, readouts=()):
                if obs is None:
                    outer.native_calls += 1
                    return [_P(outer.natives[outer.j])]
                outer.retry_calls += 1
                return [_P(outer.retry_fn(np.asarray(obs["states"]), outer.natives[outer.j].shape[0]))]
        self.policy = P()

    @property
    def ended(self):
        return self.j >= len(self.natives)

    def observe(self):
        return {"states": np.array(self.states[self.j])}

    async def execute(self, actions):
        self.executed.append(np.array(actions))
        self.j += 1
        return len(actions)

    def record(self, event, **f):
        self.records.append(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ref = {e: registry.load(e) for e in ("redecode", "redecode-b", "tilt-guard-v12-z", "tilt-guard-v12-bz", "tilt-guard-v12-b")}
    c0 = ref["redecode"].controller.build()
    amean = c0.action_mean

    def fail(s_sent, rows):
        zc = np.tile(amean.copy(), (rows, 1))
        zc[:, :6] += s_sent[:6]
        return zc
    R1 = {"redecode_vs_v12_z": {"decisions": 0, "mismatch": 0, "episodes_mismatch": []},
          "redecode-b_vs_v12_bz": {"decisions": 0, "mismatch": 0, "episodes_mismatch": []}}
    R2 = {"episodes_without_zero": 0, "redecode_vs_identity_mismatch": 0, "redecode-b_vs_v12_b_mismatch": 0, "decisions": 0}
    tok = {"plus": [], "minus": [], "rx_le_3.40": {"plus": [], "minus": []}, "rx_gt_3.40": {"plus": [], "minus": []}}
    rows = []
    for src, t, init, succ, S, N in episodes(args.run, amean):
        rec = {"src": src, "task": t, "init": init, "success": succ, "decisions": len(N)}
        for entry, v12 in (("redecode", "tilt-guard-v12-z"), ("redecode-b", "tilt-guard-v12-bz")):
            ctl = ref[entry].controller.build()
            ctx = Ctx(S, N, fail)
            asyncio.run(ctl.run(ctx))
            op = ref[v12].slots["action_op"].build()
            exp = [np.asarray(op.transform(np.array(c), {"states": np.array(s)}, {}))[:5] for s, c in zip(S, N)]
            key = f"{entry}_vs_v12_{v12.split('-')[-1]}"
            bad = sum(not np.array_equal(a, b) for a, b in zip(ctx.executed, exp))
            R1[key]["decisions"] += len(N)
            R1[key]["mismatch"] += bad
            if bad:
                R1[key]["episodes_mismatch"].append(f"{src}:t{t}i{init}")
            if entry == "redecode":
                rec["native_zero"] = ctl.stats["native_zero"]
                rec["retry_calls_worst"] = ctx.retry_calls
                rec["worst_ratio"] = (ctx.native_calls + ctx.retry_calls) / max(ctx.native_calls, 1)
                for r in ctx.records:
                    if r["native_zero"]:
                        s = np.asarray(S[r["decision"]])
                        band = "rx_le_3.40" if s[3] <= 3.40 else "rx_gt_3.40"
                        for rr in r["retries"]:
                            side = "plus" if rr["shift_bins"] > 0 else "minus"
                            tok[side].append(rr["tokens_changed"])
                            tok[band][side].append(rr["tokens_changed"])
                if ctl.stats["native_zero"] == 0:
                    R2["episodes_without_zero"] += 1
                    R2["decisions"] += len(N)
                    R2["redecode_vs_identity_mismatch"] += sum(not np.array_equal(a, n[:5]) for a, n in zip(ctx.executed, N))
            elif rec["native_zero"] == 0:
                opb = ref["tilt-guard-v12-b"].slots["action_op"].build()
                expb = [np.asarray(opb.transform(np.array(c), {"states": np.array(s)}, {}))[:5] for s, c in zip(S, N)]
                R2["redecode-b_vs_v12_b_mismatch"] += sum(not np.array_equal(a, b) for a, b in zip(ctx.executed, expb))
        rows.append(rec)

    def summ(v):
        v = np.asarray(v)
        return {"n": int(v.size), "mean": float(v.mean()) if v.size else None,
                "hist": {int(k): int((v == k).sum()) for k in np.unique(v)} if v.size else {}}
    dev = [r for r in rows if r["src"] == "dev"]
    R3 = {"tokens_changed": {k: (summ(v) if isinstance(v, list) else {kk: summ(vv) for kk, vv in v.items()}) for k, v in tok.items()},
          "zero_events": {"dev80": sum(r["native_zero"] for r in dev), "s2": sum(r["native_zero"] for r in rows if r["src"] == "s2")},
          "dev_hard48_eps_with_zero": {"F": sum(r["native_zero"] > 0 and not r["success"] for r in dev if r["task"] in HARD),
                                       "S": sorted(f"t{r['task']}i{r['init']}" for r in dev if r["task"] in HARD and r["success"] and r["native_zero"] > 0)},
          "dev80_successes_with_zero": sorted(f"t{r['task']}i{r['init']}" for r in dev if r["success"] and r["native_zero"] > 0),
          "worst_case_ratio_dev80": {"mean": float(np.mean([r["worst_ratio"] for r in dev])),
                                     "max": float(max(r["worst_ratio"] for r in dev)),
                                     "argmax": max(dev, key=lambda r: r["worst_ratio"])["task"].__str__() + "i" +
                                     str(max(dev, key=lambda r: r["worst_ratio"])["init"])},
          "note": "worst case = both retries fail at every zero chunk (2 extra calls each); the real ratio is lower when retry 1 succeeds"}
    ok = all(v["mismatch"] == 0 for v in R1.values()) and R2["redecode_vs_identity_mismatch"] == 0 and R2["redecode-b_vs_v12_b_mismatch"] == 0
    out = {"visibility": "development_only (dev init 30-37 identity, S2 init 10-29)", "pass": bool(ok),
           "R1_all_retries_fail_equals_v12": R1, "R2_no_zero_equals_identity_and_v12_b": R2, "R3_statistics": R3, "episodes": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print("PASS" if ok else "FAIL")
    print(json.dumps({"R1": R1, "R2": R2}, indent=None))
    print(json.dumps(R3, indent=None)[:2500])


if __name__ == "__main__":
    main()
