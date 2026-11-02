"""CPU geometry tests T1, T2, T4, T8 and open-loop activation replay of the frozen tilt-guard ops over recorded
DEVELOPMENT identity rollouts (dev init 30-37 and S2 init 10-29 only; never init 0-9 or 38-49).

Data alignment:
- dev jsonl (eval/dev-baseline/shards/identity-t*-i*/baseline/vla_eval/episodes/*/task*_ep*.jsonl): row k holds action
  k and the state AFTER it, so the decision-time state of decision d (step 5d) is state[5d-1] (state[0] approximates
  decision 0). Only the 5 executed rows of each chunk are recorded, so a Z tail can never be formed there (prev chunk
  shorter than 10 rows -> hold): the dev replay counts detections, not tails. vla-eval records the gripper snapped to
  +/-1, so the exact openpi zero signature (gripper == actions mean -0.0456) never matches a recorded row; the replay
  restores the gripper column of a decision to the actions mean iff all its recorded rows have dims 0-5 equal to
  actions_mean[:6] + state[5d-1][:6] within 1e-4 and gripper -1 (the snapped mean). Live, the op tests the exact
  signature on the native chunk the policy server returns (all 7 dims, tol 1e-4).
- S2 (analysis/s2_compact.npz): decision-time state and the full native 10-row chunk per decision.

Usage: replay_dev.py --run RUN_DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from harnessstudio.core import registry
from hs_tilt_guard_v12 import action_op as A
from hs_tilt_guard_v12 import geometry as geo

HARD = {2, 3, 5, 6, 8, 9}
GAIN = 0.136   # critic-measured median realized/commanded rotation per env step


def dev_episodes(run):
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
            yield "dev", t, init, bool(lab[(t, init)]), st, ac


def s2_episodes(run):
    z = np.load(os.path.join(run, "analysis/s2_compact.npz"))
    for e in json.load(open(os.path.join(run, "analysis/s2_index.json"))):
        k = e["key"]
        yield "s2", int(e["task"]), int(e["init"]), bool(e["success"]), z[k + "_state"].astype(np.float64), \
            z[k + "_chunk"].astype(np.float64)


def dev_decisions(st, ac, amean):
    out = []
    for d in range(0, len(st), 5):
        s = st[d - 1] if d > 0 else st[0]
        rows = ac[d:d + 5].copy()
        unsnapped = False
        if np.all(np.abs(rows[:, :6] - (amean[:6] + s[:6])) <= 1e-4) and np.all(rows[:, 6] == -1.0):
            rows[:, 6] = amean[6]
            unsnapped = True
        out.append((d, s, rows, unsnapped))
    return out


def build(entry):
    cfg = registry.load(entry)
    return cfg, cfg.slots["action_op"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = args.run
    rep: dict = {"visibility": "development_only (dev init 30-37 identity, S2 init 10-29)", "tests": {}}

    dev = list(dev_episodes(run))
    s2 = list(s2_episodes(run))
    cfg_gb, ref_gb = build("tilt-guard-v12-b")
    params = dict(cfg_gb.card["params"])
    amean = A.load_norm_stats("assets/pi0_fast_libero_norm_stats.json", params["norm_stats_sha256"])["actions"]["mean"]
    amean = np.asarray(amean)

    # ---- T1 frame/sign on every dev env step with |command| > 0.05 rad
    cw, cb, ratio = [], [], []
    for *_, st, ac in dev:
        for k in range(1, len(st)):
            c = 0.5 * np.clip(ac[k, 3:6], -1, 1)
            if np.linalg.norm(c) < 0.05:
                continue
            Rp, Rn = geo.rotmat(st[k - 1, 3:6]), geo.rotmat(st[k, 3:6])
            ww, wb = geo.log_so3(Rn @ Rp.T), geo.log_so3(Rp.T @ Rn)
            if np.linalg.norm(ww) < 1e-6:
                continue
            cw.append(ww @ c / np.linalg.norm(ww) / np.linalg.norm(c))
            cb.append(wb @ c / np.linalg.norm(wb) / np.linalg.norm(c))
            ratio.append(np.linalg.norm(ww) / np.linalg.norm(c))
    cw, cb, ratio = map(np.asarray, (cw, cb, ratio))
    rep["tests"]["T1"] = {"n": int(len(cw)), "median_cos_world_left": float(np.median(cw)),
                          "median_cos_body_right": float(np.median(cb)), "frac_world_pos": float(np.mean(cw > 0)),
                          "realized_over_commanded_median": float(np.median(ratio)),
                          "realized_over_commanded_p90": float(np.quantile(ratio, 0.9)),
                          "pass": bool(np.median(cw) >= 0.9 and abs(np.median(cb)) < 0.1),
                          "critic": "0.941 vs -0.012, n=15763"}

    # ---- T2 obs0 tilt
    t0 = [float(np.degrees(geo.tilt_rad(st[0, 3:6]))) for *_, st, _ in dev]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        th = rng.uniform(np.pi + 1e-3, 2 * np.pi - 1e-3)
        worst = max(worst, float(np.abs(geo.rotmat(ax * th) - geo.rotmat(ax * th * (1 - 2 * np.pi / th))).max()))
    rep["tests"]["T2"] = {"n": len(t0), "obs0_tilt_min": min(t0), "obs0_tilt_max": max(t0),
                          "antipodal_max_abs_diff": worst,
                          "pass": bool(len(t0) == 80 and min(t0) >= 2 and max(t0) <= 5 and worst <= 1e-9)}

    # ---- open-loop replay of every entry point + a B-only op (T4/T8)
    entries = ["tilt-guard-v12-" + v for v in ("b", "b-yaw", "bz", "bsat", "bsatz", "bsat-yaw", "z")]
    refs = {e: build(e)[1] for e in entries}
    bonly_params = {**{k: v for k, v in params.items()}, "guard": False, "envelope": "bte", "zero": "off"}
    dev_act = {(r["task"], r["init"]): r for r in json.load(open(os.path.join(run, "ideation/scripts-domain/dev_activation.json")))["rows"]}
    episodes = []
    t4 = defaultdict(int)
    t4_rows_rescaled = t4_rows = 0
    unsnapped_total = 0
    for src, t, init, succ, st, data in dev + s2:
        if src == "dev":
            decs = dev_decisions(st, data, amean)
            unsnapped_total += sum(u for *_, u in decs)
        else:
            dsteps = None
            decs = [(j, st[j], data[j], False) for j in range(len(st))]
        ops = {e: refs[e].build() for e in entries}
        bop = A.TiltGuardActionOp(**bonly_params)
        tilt_cross_window = False   # detector + tilt>theta in window, ignoring zero chunks (dev_activation.py rule)
        b_dec = []
        for j, (d, s, rows, _) in enumerate(decs):
            for e, op in ops.items():
                op.transform(rows, {"states": s}, {})
            out = bop.transform(rows, {"states": s}, {})
            b = bop.detector.boundary
            tilt = geo.tilt_rad(s[3:6])
            if b is not None and b <= j < b + 30 and tilt > np.radians(55):
                tilt_cross_window = True
            dg = bop.diagnostics()
            if dg.get("envelope", {}).get("active"):
                b_dec.append(j)
                n = min(5, len(rows))
                R = geo.rotmat(s[3:6])
                for g, tag in ((GAIN, "g0136"), (1.0, "g1")):
                    tn = geo.commanded_tilts(R, rows[:n, 3:6], g)
                    tb = geo.commanded_tilts(R, out[:n, 3:6], g)
                    t4[f"{src}_{tag}_decisions"] += 1
                    t4[f"{src}_{tag}_bte_end_lt_start"] += tb[-1] < tilt
                    t4[f"{src}_{tag}_bte_rowmax_gt_start_1deg"] += max(tb) > tilt + np.radians(1)
                    t4[f"{src}_{tag}_native_rise_above_start"] += max(tn) > tilt + 1e-9
                    t4[f"{src}_{tag}_bte_rise_above_start"] += max(tb) > tilt + 1e-9
                if src == "dev":
                    t4_rows += n
                    t4_rows_rescaled += int(dg["envelope"].get("rescaled", 0))
        rec = {"src": src, "task": t, "init": init, "success": succ, "decisions": len(decs),
               "boundary_decision": bop.detector.boundary, "b_active_decisions": len(b_dec),
               "b_first_decision": b_dec[0] if b_dec else None, "tilt_window_rule": tilt_cross_window}
        for e, op in ops.items():
            stt = op.stats
            rec[e] = {k: stt[k] for k in ("zero_detected", "z_tail", "z_hold", "z_passthrough", "z_max_run", "g_active",
                                          "b_active", "b_rows_rescaled", "k_fallback")}
        if src == "dev":
            ref = dev_act[(t, init)]
            rec["predeclared_boundary_step"] = ref["boundary_step"]
            rec["predeclared_p1_active"] = ref["p1_active"]
        episodes.append(rec)

    # ---- T4
    t4d = dict(t4)
    n1 = t4d.get("dev_g0136_decisions", 0)
    rep["tests"]["T4"] = {**t4d, "dev_rows_rescaled": t4_rows_rescaled, "dev_rows": t4_rows,
                          "frac_end_lt_start_g0136": t4d.get("dev_g0136_bte_end_lt_start", 0) / max(n1, 1),
                          "frac_rowmax_gt_start1_g0136": t4d.get("dev_g0136_bte_rowmax_gt_start_1deg", 0) / max(n1, 1),
                          "pass": bool(n1 > 0 and t4d.get("dev_g0136_bte_end_lt_start", 0) >= 0.95 * n1
                                       and t4d.get("dev_g0136_bte_rowmax_gt_start_1deg", 0) <= 0.05 * n1),
                          "critic": "498 decisions; g=0.136: BTE end<start 485/498, rises 13/498; g=1.0 rises 70/498 "
                                    "(critic formula without the C7 re-clip)"}

    # ---- T8 detector replay vs the predeclared activation table (dev80) and tilt_window W150 th55 (dev80+S2)
    devr = [e for e in episodes if e["src"] == "dev"]
    mism_b = [(e["task"], e["init"], e["boundary_decision"], e["predeclared_boundary_step"]) for e in devr
              if (e["boundary_decision"] * 5 if e["boundary_decision"] is not None else None) != e["predeclared_boundary_step"]]
    mism_p1 = [(e["task"], e["init"]) for e in devr if e["tilt_window_rule"] != e["predeclared_p1_active"]]
    mism_op = [(e["task"], e["init"]) for e in devr if (e["b_active_decisions"] > 0) != e["predeclared_p1_active"]]
    table = {}
    for t in range(10):
        R = [e for e in devr if e["task"] == t]
        table[f"t{t}"] = {"S": sum(e["success"] for e in R),
                          "act_S": sum(e["success"] and e["b_active_decisions"] > 0 for e in R),
                          "act_F": sum((not e["success"]) and e["b_active_decisions"] > 0 for e in R),
                          "pre_act_S": sum(e["success"] and e["predeclared_p1_active"] for e in R),
                          "pre_act_F": sum((not e["success"]) and e["predeclared_p1_active"] for e in R)}
    tw = json.load(open(os.path.join(run, "ideation/scripts-domain/tilt_window.json")))["W150_th55"]
    allr = episodes
    rep["tests"]["T8"] = {
        "dev80_boundary_step_mismatches": mism_b,
        "dev80_window_rule_vs_predeclared_p1_mismatches": mism_p1,
        "dev80_op_active_vs_predeclared_p1_mismatches": mism_op,
        "dev80_activated_episodes_op": sum(e["b_active_decisions"] > 0 for e in devr),
        "dev80_activated_episodes_predeclared": sum(e["predeclared_p1_active"] for e in devr),
        "per_task_table": table,
        "hard48_activated_F": sum(e["b_active_decisions"] > 0 and not e["success"] for e in devr if e["task"] in HARD),
        "hard48_activated_S": sum(e["b_active_decisions"] > 0 and e["success"] for e in devr if e["task"] in HARD),
        "dev80_plus_s2_boundary_eps": {"S": sum(e["boundary_decision"] is not None and e["success"] for e in allr),
                                       "F": sum(e["boundary_decision"] is not None and not e["success"] for e in allr)},
        "dev80_plus_s2_activated_window_rule": {"S": sum(e["tilt_window_rule"] and e["success"] for e in allr),
                                                "F": sum(e["tilt_window_rule"] and not e["success"] for e in allr)},
        "dev80_plus_s2_activated_op": {"S": sum(e["b_active_decisions"] > 0 and e["success"] for e in allr),
                                       "F": sum(e["b_active_decisions"] > 0 and not e["success"] for e in allr)},
        "s2_activated_successes_op": [(e["task"], e["init"]) for e in allr if e["src"] == "s2" and e["success"] and e["b_active_decisions"] > 0],
        "tilt_window_json_W150_th55": {k: {"boundary_eps": v["boundary_eps"], "activated": v["activated"]} for k, v in tw.items()},
        "zero_chunk_never_release": "unit test test_detector_semantics + replay uses the native-zero flag",
        "dev_zero_rows_unsnapped": unsnapped_total,
    }
    # S2 alignment audit: analysis/s2_compact.npz '_dstep'[j] is the env step AFTER decision j's executed rows
    # (chunk[j][:5] == act[dstep[j]-5:dstep[j]]), so the domain scripts' g = sign(act[dstep[j], 6]) is the NEXT
    # decision's gripper. Re-run the scientists' detector loop with both alignments and compare with the op.
    zz = np.load(os.path.join(run, "analysis/s2_compact.npz"))
    s2_idx = json.load(open(os.path.join(run, "analysis/s2_index.json")))
    mine = {(e["task"], e["init"]): e for e in episodes if e["src"] == "s2"}

    def sci_boundary(S, g, zc):
        w = S[:, 6] - S[:, 7]
        r = 0
        for j in range(len(S)):
            if g[j] > 0 and w[j] > 0.004:
                r += 1
            elif g[j] <= 0 and not zc[j]:
                if r >= 4:
                    return j
                r = 0
        return None
    audit = {"rows_match_act_dstep_minus5": 0, "rows_match_act_dstep": 0, "decisions": 0}
    sci = {"predeclared_alignment": {"mismatch_vs_op": 0, "S": [0, 0], "F": [0, 0]},
           "corrected_alignment": {"mismatch_vs_op": 0, "S": [0, 0], "F": [0, 0]}}
    for e in s2_idx:
        k = e["key"]
        S = zz[k + "_state"].astype(np.float64)
        act = zz[k + "_act"]
        ds = zz[k + "_dstep"]
        ch = zz[k + "_chunk"]
        zc = zz[k + "_dfail"].astype(bool)
        for j in range(len(ds)):
            audit["decisions"] += 1
            audit["rows_match_act_dstep_minus5"] += bool(np.allclose(act[ds[j] - 5:ds[j], :6], ch[j][:5, :6], atol=1e-5))
            seg = act[ds[j]:ds[j] + 5, :6]
            audit["rows_match_act_dstep"] += bool(len(seg) and np.allclose(seg, ch[j][:len(seg), :6], atol=1e-5))
        for name, g in (("predeclared_alignment", np.sign(act[np.minimum(ds, len(act) - 1), 6])),
                        ("corrected_alignment", np.sign(act[ds - 5, 6]))):
            b = sci_boundary(S, g, zc)
            sci[name]["mismatch_vs_op"] += b != mine[(int(e["task"]), int(e["init"]))]["boundary_decision"]
            key = "S" if e["success"] else "F"
            if b is not None:
                sci[name][key][0] += 1
                if any(geo.tilt_rad(S[j, 3:6]) > np.radians(55) for j in range(b, min(len(S), b + 30))):
                    sci[name][key][1] += 1
    rep["tests"]["T8"]["s2_alignment_audit"] = audit
    rep["tests"]["T8"]["s2_scientist_rule"] = sci
    rep["tests"]["T8"]["s2_note"] = ("S2 '_dstep' is the step after the decision's executed rows; the domain scripts "
                                     "(tilt_window.py, boundary_and_custody.py) and the critic's S2 checks read the next "
                                     "decision's gripper, shifting S2 boundaries by one decision (and counting next-decision "
                                     "zero-chunk grippers as releases: S2 t5 i10/i11). The op uses the decision's own native "
                                     "row-0 gripper; with corrected alignment the scientists' loop equals the op on all 200 "
                                     "S2 episodes. dev80 jsonl alignment in those scripts is correct.")
    rep["tests"]["T8"]["pass"] = bool(not mism_b and not mism_p1 and not mism_op
                                      and sci["corrected_alignment"]["mismatch_vs_op"] == 0)

    # ---- activation counts per entry point (open loop on the native trajectory)
    def summarize(src_sel, task_sel):
        E = [e for e in episodes if src_sel(e) and task_sel(e)]
        out = {"episodes": len(E), "S": sum(e["success"] for e in E)}
        for ent in entries:
            out[ent] = {
                "zero_detections": sum(e[ent]["zero_detected"] for e in E),
                "z_tail": sum(e[ent]["z_tail"] for e in E), "z_hold": sum(e[ent]["z_hold"] for e in E),
                "z_passthrough": sum(e[ent]["z_passthrough"] for e in E),
                "z_eps_F": sum(e[ent]["zero_detected"] > 0 and not e["success"] for e in E),
                "z_eps_S": sum(e[ent]["zero_detected"] > 0 and e["success"] for e in E),
                "g_active_decisions": sum(e[ent]["g_active"] for e in E),
                "g_eps_F": sum(e[ent]["g_active"] > 0 and not e["success"] for e in E),
                "g_eps_S": sorted(f"t{e['task']}i{e['init']}" for e in E if e[ent]["g_active"] > 0 and e["success"]),
                "b_active_decisions": sum(e[ent]["b_active"] for e in E),
                "b_eps_F": sum(e[ent]["b_active"] > 0 and not e["success"] for e in E),
                "b_eps_S": sorted(f"t{e['task']}i{e['init']}" for e in E if e[ent]["b_active"] > 0 and e["success"]),
                "max_zero_run": max([e[ent]["z_max_run"] for e in E] or [0]),
            }
        return out
    rep["activation"] = {
        "dev_hard48": summarize(lambda e: e["src"] == "dev", lambda e: e["task"] in HARD),
        "dev_all80": summarize(lambda e: e["src"] == "dev", lambda e: True),
        "dev_campaign30": summarize(lambda e: e["src"] == "dev", lambda e: e["init"] <= 32),
        "s2_all200": summarize(lambda e: e["src"] == "s2", lambda e: True),
    }
    rep["expected"] = {
        "G (counts.json)": "hard48 rx_g fail 27, succ t6i32,t9i32,t9i37; s2 64 F / 4 S",
        "Z detections": "dev 150 (scientist tol 1e-3 on dims 0-5, state[d-1]); s2 276 = n_decode_failed; 249 with a usable tail",
        "B (r1-domain / domain review)": "dev80 28 activated (27 F + 1 S t9i37); hard48 24 F + 1 S",
    }
    rep["episodes"] = episodes
    json.dump(rep, open(args.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_task_table",)} for k, v in rep["tests"].items()}, indent=1, default=str))
    for scope, v in rep["activation"].items():
        print("==", scope, v["episodes"], "eps", v["S"], "S")
        for ent in entries:
            x = v[ent]
            print(f"  {ent:20s} Z det {x['zero_detections']:4d} tail {x['z_tail']:3d} hold {x['z_hold']:3d} pass {x['z_passthrough']:3d} "
                  f"epsF/S {x['z_eps_F']}/{x['z_eps_S']} | G dec {x['g_active_decisions']:4d} epsF {x['g_eps_F']} S {x['g_eps_S']} | "
                  f"B dec {x['b_active_decisions']:4d} epsF {x['b_eps_F']} S {x['b_eps_S']}")


if __name__ == "__main__":
    main()
