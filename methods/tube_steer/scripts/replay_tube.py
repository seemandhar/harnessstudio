"""CPU replay of the frozen tube-steer ops over recorded DEVELOPMENT identity rollouts (dev init 30-37 traces/jsonl and
S2 init 10-29 compact arrays; never init 0-9 or 38-49), open loop on the native trajectory (A5.2 / C3 test (d)).

- Instruction: the benchmark-written task name of each episode (dev: the hs trace 'task' field
  'libero_10/<instruction>#<id>'; S2: s2_index task_name) -> exact memory lookup (C2: 9 hits, t8 abstains).
- dev decision-time state = state[5d-1] (state[0] for d = 0) from the vla_eval jsonl; rows = executed rows (native).
- S2: leave-one-out -- the op's memory for an S2 episode excludes that episode's own key (test-only injection on the op
  instance; the frozen package memory is unchanged).
Expected (review C3): dev hard excl t8 0/16 vs 19/24, easy 0/29 vs 2/3; S2 LOO hard 0/52 vs 41/48, easy 0/75 vs 1/5;
per task (dev, 200-step window) t2 6/7, t3 6/6, t5 4/5, t9 3/4, t6 0/2.
Also: arming-decision agreement with ideation/scripts-r5-memory/dev80_tube.json (f15_40), T vs ortho arming identity,
ortho steered decisions <= 6, steer-mode authority proxy (open-loop commanded component), bitwise native when not steering.

Usage: replay_tube.py --run RUN_DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re

import numpy as np

from harnessstudio.core import registry

HARD_X8 = {2, 3, 5, 6, 9}
EASY = {0, 1, 4, 7}


def dev_episodes(run):
    lab = {}
    for line in open(os.path.join(run, "eval/dev-baseline/episodes_identity.tsv")).read().strip().split("\n")[1:]:
        f = line.split("\t")
        lab[(int(f[0]), int(f[1]))] = int(f[2])
    for sh in sorted(glob.glob(os.path.join(run, "eval/dev-baseline/shards/identity-t*-i*"))):
        t, a, _ = map(int, re.search(r"identity-t(\d+)-i(\d+)-(\d+)$", sh).groups())
        traces = {json.load(open(p))["episode_idx"]: json.load(open(p))["task"]
                  for p in glob.glob(os.path.join(sh, "baseline/traces/*.json"))}
        for f in sorted(glob.glob(os.path.join(sh, "baseline/vla_eval/episodes/*/task*_ep*.jsonl"))):
            init = a + int(re.search(r"_ep(\d+)_", f).group(1))
            instr = traces[init].split("/", 1)[1].rsplit("#", 1)[0]
            recs = [json.loads(x) for x in open(f)]
            st = np.array([r["state"] for r in recs], dtype=np.float64)
            ac = np.array([r["action"] for r in recs], dtype=np.float64)
            decs = [(st[d - 1] if d > 0 else st[0], ac[d:d + 5]) for d in range(0, len(recs), 5)]
            yield "dev", t, init, bool(lab[(t, init)]), instr, None, decs


def s2_episodes(run):
    z = np.load(os.path.join(run, "analysis/s2_compact.npz"))
    for e in json.load(open(os.path.join(run, "analysis/s2_index.json"))):
        k = e["key"]
        instr = e["task_name"].split("/", 1)[1].rsplit("#", 1)[0]
        S, ch = z[k + "_state"].astype(np.float64), z[k + "_chunk"].astype(np.float64)
        yield "s2", int(e["task"]), int(e["init"]), bool(e["success"]), instr, k, [(S[j], ch[j]) for j in range(len(S))]


def keyed_memory(pkg_assets):
    meta = json.load(open(os.path.join(pkg_assets, "tube_memory.json")))
    z = np.load(io.BytesIO(open(os.path.join(pkg_assets, "tube_memory.npz"), "rb").read()))
    out = {}
    for e in meta["episodes"]:
        out.setdefault(e["instruction"], []).append((e["key"], np.asarray(z[e["key"] + "_z"], dtype=np.float64)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import hs_tube_steer
    assets = os.path.join(os.path.dirname(hs_tube_steer.__file__), "assets")
    KM = keyed_memory(assets)
    refs = {m: registry.load(m).slots["action_op"] for m in ("tube-steer", "tube-steer-ortho")}
    sci = {(r["src"], r["task"], r["init"]): r for r in
           json.load(open(os.path.join(args.run, "ideation/scripts-r5-memory/dev80_tube.json")))["rows"]}
    rows = []
    instr_seen = {}
    agree = {"compared": 0, "mismatch": []}
    passthrough_bad = 0
    ortho_over_cap = 0
    auth = []
    for src, t, init, succ, instr, key, decs in list(dev_episodes(args.run)) + list(s2_episodes(args.run)):
        instr_seen.setdefault(t, set()).add(instr)
        ops = {}
        for m, ref in refs.items():
            op = ref.build()
            if src == "s2" and instr in KM:                    # leave-one-out
                op.memory = {instr: [zm for k, zm in KM[instr] if k != key]}
            ops[m] = op
        steered_comp = []
        for s, r in decs:
            for m, op in ops.items():
                inp = np.array(r)
                out = op.transform(inp, {"states": s, "task_description": instr}, {})
                d = op.diagnostics()
                if not d.get("steer") and out is not inp:
                    passthrough_bad += 1
                if m == "tube-steer" and d.get("steer"):
                    steered_comp.append((d["native_component"], d["executed_component"]))
        T, O = ops["tube-steer"], ops["tube-steer-ortho"]
        ortho_over_cap += O.stats["steered_decisions"] > 6
        rec = {"src": src, "task": t, "init": init, "success": succ, "tube_found": T.stats["tube_found"],
               "first_arm": T.stats["first_arm"], "first_arm_ortho": O.stats["first_arm"],
               "arm_events": T.stats["arm_events"], "steered": T.stats["steered_decisions"],
               "ortho_steered": O.stats["steered_decisions"]}
        if steered_comp:
            auth.append(float(np.median([e for _, e in steered_comp])) - float(np.median([n for n, _ in steered_comp])))
        sk = sci.get((src, t, init))
        if sk is not None and "f15_40" in sk:
            agree["compared"] += 1
            if sk["f15_40"] != rec["first_arm"]:
                agree["mismatch"].append({"src": src, "task": t, "init": init, "script": sk["f15_40"], "op": rec["first_arm"]})
        rows.append(rec)

    def frac(R, s):
        sel = [r for r in R if r["success"] == s]
        return f"{sum(r['first_arm'] is not None for r in sel)}/{len(sel)}"
    tab = {}
    for src in ("dev", "s2"):
        for grp, tasks in (("hard_excl_t8", HARD_X8), ("easy", EASY)):
            R = [r for r in rows if r["src"] == src and r["task"] in tasks and r["tube_found"]]
            tab[f"{src}_{grp}"] = {"success_fired": frac(R, True), "failure_fired": frac(R, False)}
        tab[f"{src}_t8"] = {"tube_found": sorted({r["tube_found"] for r in rows if r["src"] == src and r["task"] == 8}),
                            "fired": sum(r["first_arm"] is not None for r in rows if r["src"] == src and r["task"] == 8)}
    per_task = {f"t{t}": f"{sum(r['first_arm'] is not None for r in rows if r['src'] == 'dev' and r['task'] == t and not r['success'])}/"
                         f"{sum(1 for r in rows if r['src'] == 'dev' and r['task'] == t and not r['success'])}" for t in (2, 3, 5, 6, 9)}
    expected = {"dev_hard_excl_t8": {"success_fired": "0/16", "failure_fired": "19/24"},
                "dev_easy": {"success_fired": "0/29", "failure_fired": "2/3"},
                "s2_hard_excl_t8": {"success_fired": "0/52", "failure_fired": "41/48"},
                "s2_easy": {"success_fired": "0/75", "failure_fired": "1/5"},
                "per_task_dev_failures": {"t2": "6/7", "t3": "6/6", "t5": "4/5", "t9": "3/4", "t6": "0/2"}}
    lookup = {f"t{t}": {"instructions": sorted(v), "hit": all(i in KM for i in v)} for t, v in sorted(instr_seen.items())}
    ok = (all(tab[k] == v for k, v in expected.items() if k in tab) and per_task == expected["per_task_dev_failures"]
          and not agree["mismatch"] and passthrough_bad == 0 and ortho_over_cap == 0
          and all(r["first_arm"] == r["first_arm_ortho"] for r in rows)
          and sum(v["hit"] for v in lookup.values()) == 9 and not lookup["t8"]["hit"])
    out = {"visibility": "development_only (dev init 30-37 identity, S2 init 10-29 LOO)", "pass": bool(ok),
           "exposure": tab, "per_task_dev_failures": per_task, "expected": expected,
           "arming_agreement_with_dev80_tube_json_f15_40": agree,
           "instruction_lookup_from_benchmark_task_names": lookup,
           "T_and_ortho_first_arm_identical": all(r["first_arm"] == r["first_arm_ortho"] for r in rows),
           "non_steered_decisions_not_bitwise": passthrough_bad, "ortho_episodes_over_6_steered": ortho_over_cap,
           "steer_authority_open_loop": {"episodes": len(auth), "median_executed_minus_native_component": float(np.median(auth)) if auth else None,
                                         "note": "open-loop proxy only (states do not respond); F1 is closed-loop"},
           "episodes": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print("PASS" if ok else "FAIL")
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
