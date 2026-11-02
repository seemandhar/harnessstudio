"""CPU open-loop replay of the frozen U5+Z composite (and the U5+Zsign control) over the recorded round-5 T-arm
(tube-steer) DEVELOPMENT rollouts (runs/rtcfpp-20261101/eval/round5/shards/tube-steer-t{T}-i{I}/candidate, dev80 =
libero_10 init 30-37, seed 7). Streams one scene at a time (host memory). Reads only public per-step states/actions,
the public instruction and T's logged arming diagnostic; never an outcome field (no success, no file-name label).

Inputs per decision q (sz.py convention, A6.Z.3): decision-time state = jsonl state[5q-1] (state[0] for q = 0); chunk =
the 5 executed rows of decision q, with the gripper column restored to the actions mean iff all 5 grippers are -1 (the
benchmark's sign snap) and the restored rows are the openpi zero chunk. Before T's first arming the executed rows ARE
the native rows (T passes the native chunk through); after it they are T's outputs, so post-arming inputs are not
native -- the replay then only checks that the composite equals a standalone T fed identical inputs.

Checks (pass = all):
  (a) composite U5 first arming == standalone T first arming == T's logged first arming (80/80);
  (b) the decisions where the composite output differs bitwise from standalone T are exactly its Z substitution
      decisions, all strictly before U5's first arming; outside S_Z there are none (composite == U5 bitwise);
  (c) zero-chunk detections before arming == sz.json z_fires_before_T_arming and S_Z(replay) == sz.json S_Z (25);
  (d) pre-arming composite output == standalone Z output at every pre-arming decision;
  (e) U5+Zsign: same Z detections/substitution decisions as U5+Z; equal to U5+Z bitwise before its first substitution.
Note: chunks have 5 rows here (vla-eval stores executed rows only), so Z's run-1 tail is unavailable and run 1 = hold;
the tail branch is covered by the synthetic tests. Usage:
    harnessstudio/.venv/bin/python replay_round5.py --out OUT.json [--mutant NoZ|ZPriority|ArmingDecisionToZ]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

from harnessstudio.core import registry
from hs_tilt_guard_v12 import action_op as Zmod

RUN = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtcfpp-20261101"
SZ = RUN + "/analysis/scripts-r6/data-u5z/sz.json"
SZ_SHA = "e0df428f8a43ca568322f2b5e0a2b0955980ad8dbb41788f65abb444881e4b99"
TESTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")


def first_logged_arm(trace_path):
    tr = json.load(open(trace_path))
    for e in tr["events"]:
        if e.get("t") == "decision":
            ao = e.get("diagnostics", {}).get("action_op", {})
            if ao.get("armed"):
                return ao["q"], tr["task"]
    return None, tr["task"]


def scene_inputs(t, i, mean):
    d = os.path.join(RUN, f"eval/round5/shards/tube-steer-t{t}-i{i}/candidate")
    traces = glob.glob(os.path.join(d, "traces", f"*__{i}.json"))
    js = glob.glob(os.path.join(d, "vla_eval/episodes/*/task*_ep0000_*.jsonl"))
    assert len(traces) == 1 and len(js) == 1, (t, i, traces, js)
    qa, task = first_logged_arm(traces[0])
    instr = task.split("/", 1)[1].rsplit("#", 1)[0]
    S, A = [], []
    with open(js[0]) as f:
        for line in f:
            r = json.loads(line)
            S.append(r["state"])
            A.append(r["action"])
    S = np.asarray(S, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    seq = []
    for q in range((len(A) + 4) // 5):
        s = S[5 * q - 1] if q > 0 else S[0]
        seg = A[5 * q:5 * q + 5].copy()
        if np.all(seg[:, 6] == -1.0):
            cand = seg.copy()
            cand[:, 6] = mean[6]
            if Zmod.is_openpi_zero_chunk(cand, s, mean, 1e-4):
                seg = cand
        seq.append((s, seg))
    return seq, instr, qa, {"trace": os.path.relpath(traces[0], RUN), "jsonl_sha256": hashlib.sha256(open(js[0], "rb").read()).hexdigest()}


def build(name, mutant=None):
    op = registry.load(name).slots["action_op"].build()
    if mutant:
        sys.path.insert(0, TESTS)
        import test_mutations as M
        op.__class__ = getattr(M, mutant)
    return op


def replay_scene(seq, instr, mutant):
    comp, ctrl = build("u5z-t", mutant), build("u5zsign-t", mutant)
    T, Z = build("tube-steer"), build("tilt-guard-v12-z")
    obs = {"task_description": instr}
    diff_T, subs, zeros_pre, z_mismatch_pre, ctrl_subs, ctrl_prefix_bad = [], [], [], [], [], []
    first_sub = None
    for q, (s, c) in enumerate(seq):
        o = dict(obs, states=s)
        out = np.asarray(comp.transform(c, o, {}))
        g = comp.diagnostics()
        outc = np.asarray(ctrl.transform(c, o, {}))
        gc = ctrl.diagnostics()
        to = np.asarray(T.transform(c, o, {}))
        pre = T.first_arm is None
        if pre:
            zo = np.asarray(Z.transform(c, o, {}))
            if not np.array_equal(out, zo):
                z_mismatch_pre.append(q)
            if Z.diagnostics().get("zero"):
                zeros_pre.append(5 * q)
        if not np.array_equal(out, to):
            diff_T.append(q)
        zd = g.get("z") or {}
        if zd.get("z_mode") in ("tail", "hold"):
            subs.append(q)
            first_sub = q if first_sub is None else first_sub
        if (gc.get("z") or {}).get("z_mode") in ("tail", "hold"):
            ctrl_subs.append(q)
        if (first_sub is None or q < first_sub) and not np.array_equal(out, outc):
            ctrl_prefix_bad.append(q)
    garm = (g.get("glue") or {}).get("arm_decision")
    return {"composite_arm_q": garm, "standalone_T_arm_q": T.first_arm, "z_substitution_q": subs,
            "zero_detected_pre_arming_steps": zeros_pre, "diff_vs_T_q": diff_T, "pre_arming_composite_ne_Z_q": z_mismatch_pre,
            "zsign_substitution_q": ctrl_subs, "zsign_prefix_mismatch_q": ctrl_prefix_bad, "decisions": len(seq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutant", default=None)
    args = ap.parse_args()
    raw = open(SZ, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == SZ_SHA, "sz.json changed"
    szj = json.loads(raw)
    sz_scenes = {sc: {k: r[k] for k in ("in_S_Z", "first_z_step", "z_fires_before_T_arming", "T_first_arming_step")}
                 for sc, r in szj["scenes"].items()}           # outcome fields deliberately not copied
    S_Z = list(szj["S_Z"])
    del szj, raw
    mean = build("tilt-guard-v12-z").action_mean
    rows, fails = {}, []
    for t in range(10):
        for i in range(30, 38):
            sc = f"t{t}i{i}"
            seq, instr, qa, src = scene_inputs(t, i, mean)
            r = replay_scene(seq, instr, args.mutant)
            ref = sz_scenes[sc]
            r.update(src)
            r["T_logged_arm_q"] = qa
            r["in_S_Z_replay"] = bool(r["z_substitution_q"])
            r["in_S_Z_sz_json"] = ref["in_S_Z"]
            r["first_z_step_replay"] = 5 * r["z_substitution_q"][0] if r["z_substitution_q"] else None
            arm = r["composite_arm_q"]
            checks = {
                "a_arming": r["composite_arm_q"] == r["standalone_T_arm_q"] == qa
                            and (ref["T_first_arming_step"] == (5 * qa if qa is not None else None)),
                "b_diff_is_z_substitutions": r["diff_vs_T_q"] == r["z_substitution_q"]
                            and all(arm is None or q < arm for q in r["z_substitution_q"])
                            and (ref["in_S_Z"] or not r["diff_vs_T_q"]),
                "c_detection_matches_sz": r["zero_detected_pre_arming_steps"] == ref["z_fires_before_T_arming"]
                            and r["in_S_Z_replay"] == ref["in_S_Z"] and r["first_z_step_replay"] == ref["first_z_step"],
                "d_pre_arming_equals_Z": not r["pre_arming_composite_ne_Z_q"],
                "e_zsign": r["zsign_substitution_q"] == r["z_substitution_q"] and not r["zsign_prefix_mismatch_q"],
            }
            r["checks"] = checks
            r["bitwise_U5"] = not r["diff_vs_T_q"]
            if not all(checks.values()):
                fails.append({sc: [k for k, v in checks.items() if not v]})
            rows[sc] = r
            del seq
    s_rep = sorted([sc for sc, r in rows.items() if r["in_S_Z_replay"]], key=lambda s: (int(s[1]), int(s.split("i")[1])))
    non = [sc for sc in rows if sc not in S_Z]
    out = {
        "visibility": "development_only: recorded round-5 T arm dev80 (init 30-37) public states/actions + T arming "
                      "diagnostic; no outcome field read",
        "mutant": args.mutant, "sz_json_sha256": SZ_SHA,
        "S_Z_replay": s_rep, "S_Z_sz_json": S_Z, "S_Z_equal": s_rep == S_Z,
        "non_S_Z_scenes": len(non), "non_S_Z_bitwise_U5": sum(rows[sc]["bitwise_U5"] for sc in non),
        "S_Z_first_substitution_step": {sc: rows[sc]["first_z_step_replay"] for sc in S_Z},
        "S_Z_substitution_steps": {sc: [5 * q for q in rows[sc]["z_substitution_q"]] for sc in S_Z},
        "arming_agree": sum(r["checks"]["a_arming"] for r in rows.values()),
        "spot_check_T_logged_first_arm_q": {sc: rows[sc]["T_logged_arm_q"] for sc in ("t2i30", "t2i34")},
        "failures": fails, "pass": not fails and s_rep == S_Z,
        "scenes": rows,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("scenes", "S_Z_substitution_steps")}, indent=0)[:3000])
    print("PASS" if out["pass"] else "FAIL")


if __name__ == "__main__":
    main()
