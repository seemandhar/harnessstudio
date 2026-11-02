"""CPU replay for the r2 saturation-arming tests S1, S2, S3 (replay part) and S7 over recorded DEVELOPMENT identity
rollouts (dev init 30-37 jsonl + S2 init 10-29 compact arrays only).

Alignment (as scripts/replay_dev.py): dev decision state = state[5d-1], rows = executed rows 5d..5d+4 (= native rows;
the gripper column of a decision whose rows match the zero signature on dims 0-5 within 1e-4 is restored to the actions
mean, since vla-eval records it snapped to -1); S2 decision state = state[j], chunk = chunk[j] (10 rows), env step label
= _dstep[j] exactly as ideation/scripts-residual/common.py (S2 _dstep is the step after the decision's rows; the label
is only compared, never used by the op).

Usage: replay_sat.py --run RUN_DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from harnessstudio.core import registry
from hs_tilt_guard_v11 import action_op as A

HARD = {2, 3, 5, 6, 8, 9}
V10 = ("gb", "gyaw", "gbz", "gz", "gbzsign", "gzsign", "z", "zsign")


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
            decs = []
            for d in range(0, len(recs), 5):
                s = st[d - 1] if d > 0 else st[0]
                rows = ac[d:d + 5].copy()
                if np.all(np.abs(rows[:, :6] - (amean[:6] + s[:6])) <= 1e-4) and np.all(rows[:, 6] == -1.0):
                    rows[:, 6] = amean[6]
                decs.append((d, s, rows))
            yield "dev", t, init, bool(lab[(t, init)]), decs
    z = np.load(os.path.join(run, "analysis/s2_compact.npz"))
    for e in json.load(open(os.path.join(run, "analysis/s2_index.json"))):
        k = e["key"]
        S, ch, ds = z[k + "_state"].astype(np.float64), z[k + "_chunk"].astype(np.float64), z[k + "_dstep"].astype(int)
        yield "s2", int(e["task"]), int(e["init"]), bool(e["success"]), [(int(ds[j]), S[j], ch[j]) for j in range(len(S))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = args.run
    base = dict(registry.load("tilt-guard-v11-gbsat").card["params"])
    amean = np.asarray(A.load_norm_stats("assets/pi0_fast_libero_norm_stats.json", base["norm_stats_sha256"])["actions"]["mean"])
    s1_params = {**base, "guard": False, "envelope": "bte", "zero": "off", "saturation_arming": "on"}
    ct_path = os.path.join(run, "ideation/scripts-residual/commanded_tilt.json")
    fc_path = os.path.join(run, "ideation/scripts-residual/final_classes.json")
    ct = {(r["src"], r["task"], r["init"]): r for r in json.load(open(ct_path))["rows"]}
    fc = {(r["src"], r["task"], r["init"]): r for r in json.load(open(fc_path))["rows"]}
    refs10 = {v: registry.load(f"tilt-guard-{v}").slots["action_op"] for v in V10}
    refs11 = {v: registry.load(f"tilt-guard-v11-{v}").slots["action_op"] for v in V10}
    ref_gbsat = registry.load("tilt-guard-v11-gbsat").slots["action_op"]

    rows_out = []
    s1_mism, s1_fc_mism = [], []
    s3 = {v: {"decisions": 0, "mismatch": 0, "episodes_mismatch": []} for v in V10}
    s7 = {"episodes": 0, "v10_active": 0, "bsat_active": 0, "missing_from_bsat": 0, "shared_rows_differ": 0,
          "extra": 0, "extra_not_sat_only": 0}
    for src, t, init, succ, decs in episodes(run, amean):
        op = A.TiltGuardActionOp(**s1_params)
        sat_only_decs = 0
        for j, (dstep, s, rows) in enumerate(decs):
            op.transform(np.array(rows), {"states": s}, {})
            sat_only_decs += bool(op.diagnostics().get("sat_only_would_act"))
        sat_first_step = int(decs[op.sat_first][0]) if op.sat_first is not None else None
        key = (src, t, init)
        exp = ct[key]["first_c05_t25_sat"]
        if exp != sat_first_step:
            s1_mism.append({"src": src, "task": t, "init": init, "op": sat_first_step, "commanded_tilt_json": exp})
        if key in fc and fc[key].get("tBsat") != sat_first_step:
            s1_fc_mism.append({"src": src, "task": t, "init": init, "op": sat_first_step, "final_classes_tBsat": fc[key].get("tBsat")})
        # S3: v1.0 vs v1.1 (flag off) on the same open-loop inputs, all 8 variants
        for v in V10:
            o10, o11 = refs10[v].build(), refs11[v].build()
            bad = 0
            for _, s, rows in decs:
                i10, i11 = np.array(rows), np.array(rows)
                a, b = o10.transform(i10, {"states": s}, {}), o11.transform(i11, {"states": s}, {})
                same = np.array_equal(np.asarray(a), np.asarray(b)) and ((a is i10) == (b is i11))
                bad += not same
                s3[v]["decisions"] += 1
            s3[v]["mismatch"] += bad
            if bad:
                s3[v]["episodes_mismatch"].append(f"{src}:t{t}i{init}")
        # S7: v1.1 gbsat (on) vs v1.0 gb (open loop)
        o10, o11 = refs10["gb"].build(), ref_gbsat.build()
        s7["episodes"] += 1
        for _, s, rows in decs:
            i10, i11 = np.array(rows), np.array(rows)
            a, b = o10.transform(i10, {"states": s}, {}), o11.transform(i11, {"states": s}, {})
            act10 = bool(o10.diagnostics().get("envelope", {}).get("active"))
            d11 = o11.diagnostics()
            act11 = bool(d11.get("envelope", {}).get("active"))
            s7["v10_active"] += act10
            s7["bsat_active"] += act11
            if act10 and not act11:
                s7["missing_from_bsat"] += 1
            if act10 and act11 and not np.array_equal(np.asarray(a), np.asarray(b)):
                s7["shared_rows_differ"] += 1
            if act11 and not act10:
                s7["extra"] += 1
                s7["extra_not_sat_only"] += not d11.get("sat_only_would_act")
        rows_out.append({"src": src, "task": t, "init": init, "success": succ, "sat_first_step": sat_first_step,
                         "sat_first_decision": op.sat_first, "boundary_decision": op.detector.boundary,
                         "sat_only_would_act_decisions": sat_only_decs,
                         "b_active_by_route": dict(op.stats["b_active_by_route"])})

    def scope(src, hard=None, succ=None):
        return [r for r in rows_out if r["src"] == src and (hard is None or (r["task"] in HARD) == hard)
                and (succ is None or r["success"] == succ)]
    armed = lambda R: sum(r["sat_first_step"] is not None for r in R)  # noqa: E731
    div = lambda R: sorted(f"t{r['task']}i{r['init']}" for r in R if r["sat_only_would_act_decisions"] > 0)  # noqa: E731
    critic_list = ["t2i30", "t2i31", "t2i32", "t2i33", "t2i34", "t2i35", "t2i37", "t3i30", "t3i32", "t3i34", "t3i35",
                   "t3i36", "t3i37", "t5i32", "t5i33", "t5i34", "t5i35", "t5i36", "t8i30", "t8i31", "t8i32", "t8i33",
                   "t8i34", "t8i35", "t8i36", "t8i37", "t9i30", "t9i31", "t9i34"]
    S1 = {"commanded_tilt_json": {"path": ct_path, "sha256": __import__("hashlib").sha256(open(ct_path, "rb").read()).hexdigest()},
          "final_classes_json": {"path": fc_path, "sha256": __import__("hashlib").sha256(open(fc_path, "rb").read()).hexdigest()},
          "episodes": len(rows_out), "mismatches_vs_commanded_tilt": s1_mism, "mismatches_vs_final_classes_tBsat": s1_fc_mism,
          "final_classes_rows_compared": sum(1 for r in rows_out if (r["src"], r["task"], r["init"]) in fc),
          "dev_hard_failures_armed": f"{armed(scope('dev', True, False))}/{len(scope('dev', True, False))}",
          "dev_successes_armed": [f"t{r['task']}i{r['init']}" for r in scope("dev", None, True) if r["sat_first_step"] is not None],
          "s2_hard_failures_armed": f"{armed(scope('s2', True, False))}/{len(scope('s2', True, False))}",
          "s2_successes_armed": [f"t{r['task']}i{r['init']}" for r in scope("s2", None, True) if r["sat_first_step"] is not None],
          "gates": "dev hard failures 30/32, dev successes [t9i37]; S2 64/68, [t9i28]"}
    S1["pass"] = bool(not s1_mism and not s1_fc_mism and S1["dev_hard_failures_armed"] == "30/32"
                      and S1["dev_successes_armed"] == ["t9i37"] and S1["s2_hard_failures_armed"] == "64/68"
                      and S1["s2_successes_armed"] == ["t9i28"])
    dev_hard_div = div(scope("dev", True, False))
    S2 = {"dev_hard_failures_sat_only": dev_hard_div, "n_dev_hard": f"{len(dev_hard_div)}/{len(scope('dev', True, False))}",
          "matches_critic_list": dev_hard_div == sorted(critic_list),
          "dev_easy_failures_sat_only": div(scope("dev", False, False)), "dev_successes_sat_only": div(scope("dev", None, True)),
          "s2_hard_failures_sat_only": f"{len(div(scope('s2', True, False)))}/{len(scope('s2', True, False))}",
          "s2_easy_failures_sat_only": div(scope("s2", False, False)), "s2_successes_sat_only": div(scope("s2", None, True)),
          "expected": "dev hard 29/32 (critic list), dev easy {t4i35,t4i37,t7i35}, dev successes 0; S2 59/68, 3 easy, 0 successes"}
    S2["pass"] = bool(S2["matches_critic_list"] and S2["dev_easy_failures_sat_only"] == ["t4i35", "t4i37", "t7i35"]
                      and not S2["dev_successes_sat_only"] and S2["s2_hard_failures_sat_only"] == "59/68"
                      and len(S2["s2_easy_failures_sat_only"]) == 3 and not S2["s2_successes_sat_only"])
    S3 = {"per_variant": s3, "pass": all(v["mismatch"] == 0 for v in s3.values())}
    S7 = {**s7, "pass": s7["missing_from_bsat"] == 0 and s7["shared_rows_differ"] == 0 and s7["extra_not_sat_only"] == 0}
    out = {"visibility": "development_only (dev init 30-37 identity, S2 init 10-29)",
           "tests": {"S1": S1, "S2": S2, "S3_replay": S3, "S7": S7}, "episodes": rows_out}
    json.dump(out, open(args.out, "w"), indent=1)
    for k, v in out["tests"].items():
        print(k, "PASS" if v["pass"] else "FAIL", json.dumps({kk: vv for kk, vv in v.items() if kk not in ("per_variant",)}, default=str)[:900])


if __name__ == "__main__":
    main()
