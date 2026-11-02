"""CPU replay of band-guard / band-guard-signctrl over recorded DEVELOPMENT identity rollouts (open loop).

Feeds the frozen action_op (built from the installed entry point's config, i.e. the card params) with the recorded
decision-time state and the recorded native chunk rows, and compares the activation sets with the scientist's offline
counts (ideation/scripts-general/band_trigger.json, counts.json).

Data (development scenes only):
- dev init 30-37: eval/dev-baseline/shards/identity-t*-i*/baseline/vla_eval/episodes/*/task*_ep*.jsonl. Row k holds
  action k and the state AFTER it, so the state the policy saw at decision d (step d, cadence 5) is state[d-1]
  (state[0] approximates decision 0). The chunk is the 5 executed rows a[d:d+5] (the jsonl keeps no unexecuted rows).
- s2 init 10-29: analysis/s2_compact.npz + s2_index.json (decision state + full native chunk per decision).

Usage: replay_dev.py --run RUN_DIR --out OUT.json [--method band-guard]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HARD = {2, 3, 5, 6, 8, 9}


def load_op_factory(method: str):
    from harnessstudio.core import registry
    cfg = registry.load(method)
    ref = cfg.slots["action_op"]
    return cfg, ref.build


def dev_episodes(run: str):
    lab = {}
    lines = open(os.path.join(run, "eval/dev-baseline/episodes_identity.tsv")).read().strip().split("\n")[1:]
    for line in lines:
        f = line.split("\t")
        lab[(int(f[0]), int(f[1]))] = int(f[2])
    for sh in sorted(glob.glob(os.path.join(run, "eval/dev-baseline/shards/identity-t*-i*"))):
        t, a, _ = map(int, re.search(r"identity-t(\d+)-i(\d+)-(\d+)$", sh).groups())
        for f in sorted(glob.glob(os.path.join(sh, "baseline/vla_eval/episodes/*/task*_ep*.jsonl"))):
            init = a + int(re.search(r"_ep(\d+)_", f).group(1))
            recs = [json.loads(x) for x in open(f)]
            st = np.array([r["state"] for r in recs], dtype=np.float64)
            ac = np.array([r["action"] for r in recs], dtype=np.float64)
            dec = [(d, st[d - 1] if d > 0 else st[0], ac[d:d + 5]) for d in range(0, len(recs), 5)]
            yield "dev", t, init, lab[(t, init)], dec, os.path.relpath(f, run)


def s2_episodes(run: str):
    z = np.load(os.path.join(run, "analysis/s2_compact.npz"))
    for e in json.load(open(os.path.join(run, "analysis/s2_index.json"))):
        k = e["key"]
        st, ch, ds = z[k + "_state"], z[k + "_chunk"], z[k + "_dstep"]
        yield "s2", e["task"], e["init"], int(e["success"]), [(int(ds[j]), st[j], ch[j]) for j in range(len(st))], k


def dev_zero_alignment(run: str, action_mean, tol=1e-4):
    """The jsonl gripper column is the executed (sign-snapped) value, so the live op's full zero-chunk signature cannot
    match there; check the state alignment with the dims 0-5 signature instead: rows identical and == mean + state."""
    n_prev = n_same = 0
    for sh in sorted(glob.glob(os.path.join(run, "eval/dev-baseline/shards/identity-t*-i*"))):
        for f in sorted(glob.glob(os.path.join(sh, "baseline/vla_eval/episodes/*/task*_ep*.jsonl"))):
            recs = [json.loads(x) for x in open(f)]
            st = np.array([r["state"] for r in recs], dtype=np.float64)
            ac = np.array([r["action"] for r in recs], dtype=np.float64)
            for d in range(0, len(recs), 5):
                rows = ac[d:d + 5, :6]
                if not np.all(np.abs(rows - rows[0:1]) <= tol):
                    continue
                for s, key in ((st[d - 1] if d > 0 else st[0], "prev"), (st[d], "same")):
                    if np.all(np.abs(rows[0] - (action_mean[:6] + s[:6])) <= tol):
                        if key == "prev":
                            n_prev += 1
                        else:
                            n_same += 1
    return {"zero_chunks_dims0_5_with_state_d_minus_1": n_prev, "zero_chunks_dims0_5_with_state_d": n_same,
            "expected_from_proposal": {"state_d_minus_1": 150, "state_d": 1}}


def replay(build, episodes):
    out = []
    totals = {"decisions": 0, "active": 0, "inactive_identity_object": 0, "inactive_not_identity": 0,
              "skip_zero_chunk": 0, "active_rows_changed_only_rx": 0, "active_rows_changed_other": 0}
    for src, t, init, succ, dec, ref in episodes:
        o = build()
        first, n, shift = None, 0, 0.0
        for d, s, rows in dec:
            rows = np.asarray(rows)
            if len(rows) == 0:
                continue
            res = o.transform(rows, {"states": np.asarray(s)}, {})
            diag = o.diagnostics()
            totals["decisions"] += 1
            if diag.get("active"):
                n += 1
                shift += abs(diag["shift"]) * 5   # band_trigger.py's bookkeeping (x5 even for a cut last chunk)
                first = d if first is None else first
                totals["active"] += 1
                other = [i for i in range(rows.shape[1]) if i != 3]
                ok = np.array_equal(np.asarray(res)[:, other], rows[:, other])
                totals["active_rows_changed_only_rx" if ok else "active_rows_changed_other"] += 1
            else:
                totals["inactive_identity_object" if res is rows else "inactive_not_identity"] += 1
        totals["skip_zero_chunk"] += o.stats["skip_zero_chunk"]
        out.append({"src": src, "task": t, "init": init, "success": succ, "first": first, "n": n,
                    "shift": shift, "stats": dict(o.stats), "ref": ref})
    return out, totals


def scope_counts(eps):
    res = {}
    for scope, sel in (("hard48", lambda e: e["task"] in HARD), ("all80", lambda e: True),
                       ("campaign30", lambda e: e["init"] <= 32)):
        E = [e for e in eps if sel(e)]
        F = [e for e in E if not e["success"]]
        S = [e for e in E if e["success"]]
        res[scope] = {"n": len(E), "succ": len(S), "fail": len(F),
                      "rx_g_fail": sum(1 for e in F if e["first"] is not None),
                      "rx_g_succ": [f"t{e['task']}i{e['init']}" for e in S if e["first"] is not None],
                      "untouched_fail": [f"t{e['task']}i{e['init']}" for e in F if e["first"] is None]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="band-guard")
    ap.add_argument("--control", default="band-guard-signctrl")
    args = ap.parse_args(argv)
    run = os.path.abspath(args.run)
    cfg, build = load_op_factory(args.method)
    ccfg, cbuild = load_op_factory(args.control)
    import hs_band_guard.action_op as A
    report = {"visibility": "development_only", "method": args.method, "method_fingerprint": cfg.fingerprint(),
              "control": args.control, "control_fingerprint": ccfg.fingerprint(),
              "action_op_file": A.__file__, "alignment": "dev: decision d state = state[d-1] (state[0] for d=0); "
              "chunk = executed rows a[d:d+5]; s2: recorded decision state + full native chunk"}
    dev, dev_tot = replay(build, dev_episodes(run))
    s2, s2_tot = replay(build, s2_episodes(run))
    cdev, cdev_tot = replay(cbuild, dev_episodes(run))
    report["dev_totals"], report["s2_totals"], report["control_dev_totals"] = dev_tot, s2_tot, cdev_tot
    report["counts"] = scope_counts(dev)
    report["dev_alignment_check"] = dev_zero_alignment(run, build().action_mean)

    ref_counts = json.load(open(os.path.join(run, "ideation/scripts-general/counts.json")))["counts"]
    cmp = {}
    for scope, mine in report["counts"].items():
        ref = ref_counts[scope]
        cmp[scope] = {"rx_g_fail": [mine["rx_g_fail"], ref["rx_g_fail"]],
                      "rx_g_succ": [mine["rx_g_succ"], ref["rx_g_succ"]],
                      "match": mine["rx_g_fail"] == ref["rx_g_fail"] and mine["rx_g_succ"] == ref["rx_g_succ"]
                      and (mine["n"], mine["succ"], mine["fail"]) == (ref["n"], ref["succ"], ref["fail"])}
    report["vs_counts_json"] = cmp

    bt = json.load(open(os.path.join(run, "ideation/scripts-general/band_trigger.json")))["episodes"]
    btk = {(e["src"], e["task"], e["init"], e["success"]): e for e in bt}
    mism = []
    n_cmp = 0
    for e in dev + s2:
        b = btk.get((e["src"], e["task"], e["init"], e["success"]))
        if b is None:
            mism.append({"key": [e["src"], e["task"], e["init"]], "why": "missing in band_trigger.json"})
            continue
        n_cmp += 1
        if (e["first"], e["n"]) != (b.get("first_rx_g"), b.get("rx_g_n", 0)) or \
                abs(e["shift"] - b.get("rx_g_shift", 0.0)) > 1e-6:
            mism.append({"key": [e["src"], e["task"], e["init"]], "mine": [e["first"], e["n"], e["shift"]],
                         "band_trigger": [b.get("first_rx_g"), b.get("rx_g_n", 0), b.get("rx_g_shift", 0.0)]})
    report["vs_band_trigger_per_episode"] = {"episodes_compared": n_cmp, "mismatches": mism}
    for src, eps in (("s2", s2),):
        F = [e for e in eps if not e["success"]]
        S = [e for e in eps if e["success"]]
        report[f"{src}_counts"] = {"n": len(eps), "fail": len(F), "succ": len(S),
                                   "rx_g_fail": sum(1 for e in F if e["first"] is not None),
                                   "rx_g_succ": sum(1 for e in S if e["first"] is not None)}
    # control: same gate in open loop -> same activation decisions as the candidate
    same = all((a["first"], a["n"]) == (b["first"], b["n"]) for a, b in zip(dev, cdev))
    report["control_same_activation_open_loop"] = same
    report["control_coins"] = {"plus": sum(e["stats"].get("coin_plus", 0) for e in cdev),
                               "minus": sum(e["stats"].get("coin_minus", 0) for e in cdev)}
    report["zero_activation_episodes_bitwise_native"] = {
        "dev_episodes_zero_activation": sum(1 for e in dev if e["n"] == 0),
        "all_inactive_decisions_returned_input_object": dev_tot["inactive_not_identity"] == 0
        and s2_tot["inactive_not_identity"] == 0 and cdev_tot["inactive_not_identity"] == 0}
    report["episodes"] = {"dev": dev, "s2": s2}
    report["all_match"] = all(c["match"] for c in cmp.values()) and not mism and same and \
        report["zero_activation_episodes_bitwise_native"]["all_inactive_decisions_returned_input_object"] and \
        dev_tot["active_rows_changed_other"] == 0 and s2_tot["active_rows_changed_other"] == 0
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: report[k] for k in ("method_fingerprint", "dev_alignment_check", "control_fingerprint", "counts", "vs_counts_json",
                                             "s2_counts", "dev_totals", "s2_totals", "control_dev_totals",
                                             "control_same_activation_open_loop", "control_coins",
                                             "zero_activation_episodes_bitwise_native", "all_match")}, indent=1))
    print("band_trigger per-episode mismatches:", len(mism), mism[:5])
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
