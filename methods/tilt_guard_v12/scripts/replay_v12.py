"""CPU replay for tilt-guard v1.2 over recorded DEVELOPMENT identity rollouts (dev init 30-37 jsonl + S2 init 10-29
compact arrays; never init 0-9 or 38-49), open loop on the native trajectory:

 E1  every v1.2 entry == its v1.1 counterpart's op with G disabled (v1.1 card params, guard False), bitwise, on every
     decision of every episode;
 E2  every v1.2 entry == its real v1.1 counterpart entry (G on) bitwise on every episode in which the v1.1 G never
     activates (episodes where G is active are listed and excluded);
 A   activation counts per entry (B active decisions/episodes by outcome, Z detections/tails/holds/passthroughs, Bsat
     arming and saturation-only divergence), plus the S1/S2 gates of the r2 review recomputed on the v1.2 ops and the
     S7 superset (bsat contains b, shared rows bitwise equal).

Alignment as scripts/replay_dev.py (dev state[5d-1], executed rows; zero-chunk gripper restored to the actions mean).
Usage: replay_v12.py --run RUN_DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from harnessstudio.core import registry
from hs_tilt_guard_v12 import ENTRIES
from hs_tilt_guard_v12 import action_op as A

HARD = {2, 3, 5, 6, 8, 9}
COUNTERPART = {"b": "gb", "b-yaw": "gyaw", "bz": "gbz", "bsat": "gbsat", "bsatz": "gbsatz", "bsat-yaw": "gbsat-yaw", "z": "z"}
CRITIC_DEV_HARD = ["t2i30", "t2i31", "t2i32", "t2i33", "t2i34", "t2i35", "t2i37", "t3i30", "t3i32", "t3i34", "t3i35",
                   "t3i36", "t3i37", "t5i32", "t5i33", "t5i34", "t5i35", "t5i36", "t8i30", "t8i31", "t8i32", "t8i33",
                   "t8i34", "t8i35", "t8i36", "t8i37", "t9i30", "t9i31", "t9i34"]


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
    from hs_tilt_guard_v11.action_op import TiltGuardActionOp as V11
    refs12 = {v: registry.load(f"tilt-guard-v12-{v}").slots["action_op"] for v in ENTRIES}
    refs11 = {v: registry.load(f"tilt-guard-v11-{COUNTERPART[v]}").slots["action_op"] for v in ENTRIES}
    p11off = {}
    for v in ENTRIES:
        p = dict(refs11[v].params)
        p.pop("guard_factory")
        p.pop("guard_params")
        p["guard"] = False
        p11off[v] = p
    sha = refs12["b"].params["norm_stats_sha256"]
    amean = np.asarray(A.load_norm_stats("assets/pi0_fast_libero_norm_stats.json", sha)["actions"]["mean"])
    E1 = {v: {"decisions": 0, "mismatch": 0} for v in ENTRIES}
    E2 = {v: {"episodes_compared": 0, "episodes_g_active_excluded": 0, "decisions": 0, "mismatch": 0} for v in ENTRIES}
    S7 = {"b_active": 0, "bsat_active": 0, "missing": 0, "shared_rows_differ": 0, "extra": 0, "extra_not_sat_only": 0}
    rows = []
    for src, t, init, succ, decs in episodes(args.run, amean):
        rec = {"src": src, "task": t, "init": init, "success": succ}
        for v in ENTRIES:
            o12, o11off, o11 = refs12[v].build(), V11(**p11off[v]), refs11[v].build()
            outs11 = []
            g_active = 0
            for _, s, r in decs:
                i12, iof, i11 = np.array(r), np.array(r), np.array(r)
                a = o12.transform(i12, {"states": s}, {})
                b = o11off.transform(iof, {"states": s}, {})
                c = o11.transform(i11, {"states": s}, {})
                E1[v]["decisions"] += 1
                E1[v]["mismatch"] += not (np.array_equal(np.asarray(a), np.asarray(b)) and ((a is i12) == (b is iof)))
                outs11.append((np.asarray(a), np.asarray(c), (a is i12) == (c is i11)))
            g_active = o11.stats["g_active"]
            if g_active:
                E2[v]["episodes_g_active_excluded"] += 1
            else:
                E2[v]["episodes_compared"] += 1
                for a, c, same_obj in outs11:
                    E2[v]["decisions"] += 1
                    E2[v]["mismatch"] += not (np.array_equal(a, c) and same_obj)
            st = o12.stats
            rec[v] = {"b_active": st["b_active"], "zero_detected": st["zero_detected"], "z_tail": st["z_tail"],
                      "z_hold": st["z_hold"], "z_passthrough": st["z_passthrough"], "sat_first": st["sat_first_decision"],
                      "n_sat_only_would_act": st["n_sat_only_would_act"], "n_sat_only_active": st["n_sat_only_active"],
                      "b_active_by_route": dict(st["b_active_by_route"]), "boundary": st["boundary_decision"],
                      "v11_g_active": g_active}
        # S7: bsat vs b on the same inputs
        ob, os_ = refs12["b"].build(), refs12["bsat"].build()
        for _, s, r in decs:
            ib, isat = np.array(r), np.array(r)
            a, b = ob.transform(ib, {"states": s}, {}), os_.transform(isat, {"states": s}, {})
            ab = bool(ob.diagnostics().get("envelope", {}).get("active"))
            d = os_.diagnostics()
            asat = bool(d.get("envelope", {}).get("active"))
            S7["b_active"] += ab
            S7["bsat_active"] += asat
            S7["missing"] += ab and not asat
            S7["shared_rows_differ"] += ab and asat and not np.array_equal(np.asarray(a), np.asarray(b))
            if asat and not ab:
                S7["extra"] += 1
                S7["extra_not_sat_only"] += not d.get("sat_only_would_act")
        rows.append(rec)

    def scope(src, hard=None, succ=None):
        return [r for r in rows if r["src"] == src and (hard is None or (r["task"] in HARD) == hard)
                and (succ is None or r["success"] == succ)]
    lab = lambda r: f"t{r['task']}i{r['init']}"  # noqa: E731
    act = {}
    for name, R in (("dev_hard48", scope("dev", True)), ("dev_easy32", scope("dev", False)), ("dev_all80", scope("dev")),
                    ("dev_campaign30", [r for r in scope("dev") if r["init"] <= 32]), ("s2_all200", scope("s2"))):
        act[name] = {"episodes": len(R), "S": sum(r["success"] for r in R)}
        for v in ENTRIES:
            x = [r[v] for r in R]
            act[name][v] = {"b_active_decisions": sum(e["b_active"] for e in x),
                            "b_eps_F": sum(r[v]["b_active"] > 0 and not r["success"] for r in R),
                            "b_eps_S": sorted(lab(r) for r in R if r[v]["b_active"] > 0 and r["success"]),
                            "zero_detections": sum(e["zero_detected"] for e in x), "z_tail": sum(e["z_tail"] for e in x),
                            "z_hold": sum(e["z_hold"] for e in x), "z_passthrough": sum(e["z_passthrough"] for e in x),
                            "sat_armed_eps_F": sum(r[v]["sat_first"] is not None and not r["success"] for r in R),
                            "sat_armed_eps_S": sorted(lab(r) for r in R if r[v]["sat_first"] is not None and r["success"]),
                            "sat_only_eps_F": sum(r[v]["n_sat_only_would_act"] > 0 and not r["success"] for r in R),
                            "sat_only_eps_S": sorted(lab(r) for r in R if r[v]["n_sat_only_would_act"] > 0 and r["success"])}
    dev_hard_F = [r for r in scope("dev", True, False)]
    gates = {
        "B_dev80_activated_episodes": f"{act['dev_all80']['b']['b_eps_F']} F + {act['dev_all80']['b']['b_eps_S']} (expected 27 F + [t9i37])",
        "B_hard48": f"{act['dev_hard48']['b']['b_eps_F']} F + {act['dev_hard48']['b']['b_eps_S']} (expected 24 F + [t9i37])",
        "B_dev80_active_decisions": f"{act['dev_all80']['b']['b_active_decisions']} (expected 498)",
        "Z_detections": f"dev {act['dev_all80']['z']['zero_detections']} (expected 150), S2 {act['s2_all200']['z']['zero_detections']} (expected 276), S2 tails {act['s2_all200']['z']['z_tail']} (expected 249)",
        "S1_arming": f"dev hard F {sum(r['bsat']['sat_first'] is not None for r in dev_hard_F)}/{len(dev_hard_F)} (expected 30/32), dev S {act['dev_all80']['bsat']['sat_armed_eps_S']} (expected [t9i37]); S2 hard F {sum(r['bsat']['sat_first'] is not None for r in scope('s2', True, False))}/{len(scope('s2', True, False))} (expected 64/68), S2 S {act['s2_all200']['bsat']['sat_armed_eps_S']} (expected [t9i28])",
        "S2_divergence_dev_hard_matches_critic": sorted(lab(r) for r in dev_hard_F if r["bsat"]["n_sat_only_would_act"] > 0) == sorted(CRITIC_DEV_HARD),
    }
    ok = (all(x["mismatch"] == 0 for x in E1.values()) and all(x["mismatch"] == 0 for x in E2.values())
          and S7["missing"] == 0 and S7["shared_rows_differ"] == 0 and S7["extra_not_sat_only"] == 0
          and act["dev_all80"]["b"]["b_eps_F"] == 27 and act["dev_all80"]["b"]["b_eps_S"] == ["t9i37"]
          and act["dev_hard48"]["b"]["b_eps_F"] == 24 and act["dev_all80"]["b"]["b_active_decisions"] == 498
          and act["dev_all80"]["z"]["zero_detections"] == 150 and act["s2_all200"]["z"]["zero_detections"] == 276
          and act["s2_all200"]["z"]["z_tail"] == 249 and gates["S2_divergence_dev_hard_matches_critic"])
    out = {"visibility": "development_only (dev init 30-37 identity, S2 init 10-29)", "pass": bool(ok),
           "E1_v12_equals_v11_guard_off": E1, "E2_v12_equals_v11_where_G_inactive": E2, "S7_bsat_superset_of_b": S7,
           "gates": gates, "activation": act, "episodes": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print("PASS" if ok else "FAIL")
    print("E1", json.dumps(E1))
    print("E2", json.dumps(E2))
    print("S7", json.dumps(S7))
    for k, v in gates.items():
        print(k, v)
    for name, a in act.items():
        print("==", name, a["episodes"], "eps", a["S"], "S")
        for v in ENTRIES:
            x = a[v]
            print(f"  {v:9s} B dec {x['b_active_decisions']:4d} epsF {x['b_eps_F']:2d} S {x['b_eps_S']} | Z det {x['zero_detections']:3d} "
                  f"tail {x['z_tail']:3d} hold {x['z_hold']:3d} pass {x['z_passthrough']:2d} | sat armed F {x['sat_armed_eps_F']} "
                  f"S {x['sat_armed_eps_S']} | sat-only F {x['sat_only_eps_F']} S {x['sat_only_eps_S']}")


if __name__ == "__main__":
    main()
