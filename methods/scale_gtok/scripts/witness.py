#!/usr/bin/env python3
"""scale-gtok mechanism witness (review RC4) and activation metrics, from HarnessStudio v2 arm folders.

Read-only. numpy + stdlib only (run it with harnessstudio/.venv/bin/python or any python with numpy).

An ARM folder is what `hs eval/verify --out OUT` writes per arm (OUT/candidate, OUT/null, OUT/baseline): it holds
records.jsonl and vla_eval/episodes/**.jsonl (one row per executed step: action, post-step state). Each record is
matched to its episode jsonl by recomputing HarnessStudio's per-step action fingerprint (context/trace.py
`fingerprint` of the float32 action) from the jsonl rows; a record whose hashes match no jsonl is reported.

Checks (cases keyed by (benchmark, task, episode_idx); label tTiI = suite task id T, init/episode I):

- candidate vs the CRN-matched SCALE record (default: the eval/devbase map, case namespace ''):
  RC4: all executed actions before the first differing step k are bitwise equal (action hashes), the observations
  the candidate acted on up to k are equal (obs hashes; an earlier obs difference = renderer branch R), and the
  action at k differs ONLY in dim 6 (gripper). A case that never diverges must equal the record end to end,
  outcome included. Any violation -> infrastructure (no outcome interpretation), per the review.
- null / baseline arms vs the greedy map record: M (identical action hashes, same length).
- activation (review activation_check_merged): pooled executed-gripper short-run rate = interior runs of the
  executed gripper sign with length <= 2, per 100 executed steps, over the candidate episodes (pass <= 5.0,
  partial 5.0-6.6, implementation_failure >= 6.6); the same metric on the reference records of the same scenes;
  divergence share (>= 90%) and share of diverged pairs whose first divergence is before step 120 (>= 75%).
- compute: c = candidate policy_seconds per step / 0.756 (map greedy s/step); flag > 1.2, gate 4.0.
- outcome tallies vs SCALE and vs greedy, wins inside the predeclared target set T, regression-watch scenes.
  The Campaign gates and the supervisor decide; the 'r8_advisory' block only restates the review's rules.

Usage:
  witness.py --candidate ARM [ARM ...] [--null ARM ...] [--baseline ARM ...]
             [--scale-ref ARM ...] [--greedy-ref ARM ...] --out witness.json
  witness.py --self-check-map --out selfcheck.json     # metric on the map's own grid records (init 0-3)
Exit 0 = no witness violation and every null/baseline case M; 1 = violation(s); 2 = usage/matching error.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

DEVBASE = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/scalepp-20261102/eval/devbase/shards"
DEFAULT_SCALE_REF = sorted(glob.glob(f"{DEVBASE}/candidate-t*/candidate"))
DEFAULT_GREEDY_REF = sorted(glob.glob(f"{DEVBASE}/baseline-t*/baseline"))
GRIPPER_DIM = 6
NATIVE_S_PER_STEP = 0.756             # map greedy policy s/step (review overhead_null_answer)
FIRST_DIV_STEP = 120
TARGET_SET = ["t0i2", "t2i0", "t2i1", "t3i1", "t3i3", "t4i2", "t6i2", "t8i0", "t8i3"]
REGRESSION_WATCH = ["t0i0", "t7i0", "t1i0", "t4i1", "t6i0", "t6i3"]
ACTIVATION_PASS, ACTIVATION_FAIL = 5.0, 6.6


# ---- HarnessStudio's action fingerprint (harnessstudio/context/trace.py), re-implemented for float32 rows ------
def action_fingerprint(row) -> str:
    a = np.ascontiguousarray(np.asarray(row, dtype=np.float32))
    canon = {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _write(path, report) -> None:
    with open(path, "w") as fh:
        fh.write(json.dumps(report, indent=1))


def case_key(rec) -> tuple:
    return (rec["benchmark"], rec["task"], int(rec["episode_idx"]))


def case_label(key) -> str:
    m = re.search(r"#(\d+)$", key[1])
    return f"t{m.group(1) if m else '?'}i{key[2]}"


def load_arm(dirs) -> dict:
    """{case_key: {"record", "actions" [n,7] float32, "jsonl"}} over one or more arm folders (concatenated)."""
    out = {}
    for d in dirs:
        rec_path = os.path.join(d, "records.jsonl")
        if not os.path.isfile(rec_path):
            raise SystemExit(f"witness: no records.jsonl in {d}")
        with open(rec_path) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        files = sorted(glob.glob(os.path.join(d, "vla_eval", "episodes", "**", "*.jsonl"), recursive=True))
        by_first = defaultdict(list)
        loaded = {}
        for f in files:
            with open(f) as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            acts = np.asarray([r["action"] for r in rows], dtype=np.float32).reshape(len(rows), -1)
            hashes = [action_fingerprint(a) for a in acts]
            loaded[f] = (acts, hashes)
            if hashes:
                by_first[(hashes[0], len(hashes))].append(f)
        for rec in records:
            k = case_key(rec)
            if k in out:
                raise SystemExit(f"witness: duplicate case {k} across {dirs}")
            hs = rec.get("action_sha256") or []
            match = None
            for f in by_first.get((hs[0], len(hs)) if hs else ("", 0), []):
                if loaded[f][1] == hs:
                    match = f
                    break
            out[k] = {"record": rec, "actions": loaded[match][0] if match else None, "jsonl": match}
    return out


def gripper_short_runs(actions) -> tuple[int, int]:
    """(# interior runs of the executed gripper sign with length <= 2, # executed steps)."""
    g = (np.asarray(actions)[:, GRIPPER_DIM] > 0).astype(int)
    n = len(g)
    runs, s = [], 0
    for k in range(1, n + 1):
        if k == n or g[k] != g[s]:
            runs.append(k - s)
            s = k
    return sum(1 for length in runs[1:-1] if length <= 2), n


def compare(cand: dict, ref: dict) -> dict:
    """RC4 classification of one candidate episode against its CRN-matched reference episode."""
    rc, rr = cand["record"], ref["record"]
    ca, ra = rc.get("action_sha256") or [], rr.get("action_sha256") or []
    co, ro = rc.get("obs_sha256") or [], rr.get("obs_sha256") or []
    n = min(len(ca), len(ra))
    k = next((i for i in range(n) if ca[i] != ra[i]), None)
    out = {"first_divergence": k, "len_candidate": len(ca), "len_reference": len(ra),
           "success_candidate": bool(rc["success"]), "success_reference": bool(rr["success"])}
    if k is None:
        same = len(ca) == len(ra) and out["success_candidate"] == out["success_reference"]
        out.update(diverged=False, witness_ok=same,
                   reason="identical end to end" if same else "no action difference but length/outcome differs")
        return out
    obs_diff = next((i for i in range(min(len(co), len(ro), k + 1)) if co[i] != ro[i]), None)
    out["diverged"] = True
    if obs_diff is not None:
        out.update(witness_ok=False, reason=f"observation differs at step {obs_diff} <= first action difference {k} "
                                            "(renderer/env branch R): not attributable")
        return out
    if cand["actions"] is None or ref["actions"] is None:
        out.update(witness_ok=False, reason="episode jsonl not matched to the record hashes; cannot read the action")
        return out
    a, b = cand["actions"][k], ref["actions"][k]
    dims = [int(i) for i in range(len(a)) if a[i].tobytes() != b[i].tobytes()]
    out.update(dims_differing=dims, action_candidate=[float(x) for x in a], action_reference=[float(x) for x in b])
    if dims == [GRIPPER_DIM]:
        out.update(witness_ok=True, reason="first difference only in the gripper dim")
    else:
        out.update(witness_ok=False, reason=f"first difference in dims {dims} (not only dim {GRIPPER_DIM}): "
                                            "non-determinism or implementation fault -> infrastructure")
    return out


def m_check(arm: dict, ref: dict) -> dict:
    per, bad = {}, []
    for k, c in sorted(arm.items()):
        if k not in ref:
            per[case_label(k)] = "no reference record"
            bad.append(case_label(k))
            continue
        same = (c["record"].get("action_sha256") == ref[k]["record"].get("action_sha256")
                and bool(c["record"]["success"]) == bool(ref[k]["record"]["success"]))
        per[case_label(k)] = "M" if same else "not M"
        if not same:
            bad.append(case_label(k))
    return {"cases": len(arm), "not_M": bad, "per_case": per}


def flicker(arm: dict, keys) -> dict:
    short = steps = 0
    per = {}
    for k in keys:
        acts = arm[k]["actions"]
        if acts is None:
            per[case_label(k)] = None
            continue
        s, n = gripper_short_runs(acts)
        short, steps = short + s, steps + n
        per[case_label(k)] = round(100.0 * s / n, 2) if n else None
    vals = [v for v in per.values() if v is not None]
    return {"pooled_per_100": round(100.0 * short / steps, 2) if steps else None, "short_runs": short,
            "steps": steps, "episode_mean_per_100": round(float(np.mean(vals)), 2) if vals else None,
            "per_case": per}


def tally(cand: dict, ref: dict, keys) -> dict:
    wins = [case_label(k) for k in keys if cand[k]["record"]["success"] and not ref[k]["record"]["success"]]
    losses = [case_label(k) for k in keys if ref[k]["record"]["success"] and not cand[k]["record"]["success"]]
    return {"candidate": sum(bool(cand[k]["record"]["success"]) for k in keys),
            "reference": sum(bool(ref[k]["record"]["success"]) for k in keys), "episodes": len(keys),
            "wins": wins, "losses": losses, "net": len(wins) - len(losses)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--candidate", nargs="*", default=[], help="candidate arm folder(s)")
    p.add_argument("--null", nargs="*", default=[], help="null arm folder(s): must be M vs the greedy reference")
    p.add_argument("--baseline", nargs="*", default=[], help="baseline arm folder(s): must be M vs the greedy reference")
    p.add_argument("--scale-ref", nargs="*", default=DEFAULT_SCALE_REF)
    p.add_argument("--greedy-ref", nargs="*", default=DEFAULT_GREEDY_REF)
    p.add_argument("--first-div-step", type=int, default=FIRST_DIV_STEP)
    p.add_argument("--native-s-per-step", type=float, default=NATIVE_S_PER_STEP)
    p.add_argument("--self-check-map", action="store_true",
                   help="metric self-check on the map's own grid records (tasks 0-9 x init 0-3)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    scale_ref, greedy_ref = load_arm(args.scale_ref), load_arm(args.greedy_ref)
    report: dict = {"tool": os.path.abspath(__file__),
                    "tool_sha256": _file_sha256(__file__),
                    "scale_ref": args.scale_ref, "greedy_ref": args.greedy_ref}
    unmatched = [case_label(k) for arm in (scale_ref, greedy_ref) for k, v in arm.items() if v["actions"] is None]
    if args.self_check_map:
        grid = [k for k in scale_ref if k[2] <= 3 and k in greedy_ref]
        report["self_check_map"] = {"scenes": len(grid), "greedy": flicker(greedy_ref, sorted(grid)),
                                    "scale": flicker(scale_ref, sorted(grid)),
                                    "expected_pooled": {"greedy": 3.90, "scale": 7.68},
                                    "unmatched_reference_records": unmatched}
        _write(args.out, report)
        sc = report["self_check_map"]
        print(f"self-check: {len(grid)} grid scenes, greedy {sc['greedy']['pooled_per_100']} / SCALE "
              f"{sc['scale']['pooled_per_100']} short runs per 100 steps (review: 3.90 / 7.68); unmatched {unmatched}")
        return 0 if not unmatched else 2

    ok = True
    if args.candidate:
        cand = load_arm(args.candidate)
        miss = [case_label(k) for k, v in cand.items() if v["actions"] is None]
        keys = sorted(k for k in cand if k in scale_ref and cand[k]["record"]["valid"] and scale_ref[k]["record"]["valid"])
        per = {case_label(k): compare(cand[k], scale_ref[k]) for k in keys}
        violations = [c for c, v in per.items() if not v["witness_ok"]]
        diverged = [v for v in per.values() if v["diverged"]]
        early = [v for v in diverged if v["first_divergence"] < args.first_div_step]
        act = flicker(cand, keys)
        rate = act["pooled_per_100"]
        status = (None if rate is None else "pass" if rate <= ACTIVATION_PASS
                  else "implementation_failure" if rate >= ACTIVATION_FAIL else "partial")
        steps = sum(cand[k]["record"]["accounting"]["steps"] for k in keys)
        pol = sum(cand[k]["record"]["accounting"]["policy_seconds"] for k in keys)
        c = (pol / steps) / args.native_s_per_step if steps else None
        both = [k for k in keys if k in greedy_ref]
        vs_scale, vs_greedy = tally(cand, scale_ref, keys), tally(cand, greedy_ref, both)
        wins_in_t = [w for w in vs_scale["wins"] if w in TARGET_SET]
        report["candidate"] = {
            "arm_dirs": args.candidate, "records": len(cand), "paired_with_scale": len(keys),
            "not_paired": sorted(case_label(k) for k in cand if k not in keys),
            "unmatched_jsonl": miss, "unmatched_reference_records": unmatched,
            "witness_rc4": {"violations": violations, "ok": not violations and not miss, "per_case": per},
            "activation": {"candidate": act, "scale_same_scenes": flicker(scale_ref, keys),
                           "greedy_same_scenes": flicker(greedy_ref, both),
                           "status": status, "pass_le": ACTIVATION_PASS, "implementation_failure_ge": ACTIVATION_FAIL,
                           "divergence_share": round(len(diverged) / len(keys), 3) if keys else None,
                           "early_share_of_diverged": round(len(early) / len(diverged), 3) if diverged else None,
                           "first_div_step_threshold": args.first_div_step,
                           "divergence_ok": bool(keys and len(diverged) >= 0.9 * len(keys)
                                                 and diverged and len(early) >= 0.75 * len(diverged))},
            "compute": {"steps": steps, "policy_seconds": round(pol, 2),
                        "s_per_step": round(pol / steps, 4) if steps else None,
                        "c_vs_native": round(c, 3) if c is not None else None, "flag_gt": 1.2, "gate_le": 4.0},
            "outcomes": {"vs_scale": vs_scale, "vs_greedy": vs_greedy, "target_set_T": TARGET_SET,
                         "wins_vs_scale_in_T": wins_in_t,
                         "regression_watch": {w: {"candidate": per[w]["success_candidate"],
                                                  "scale": per[w]["success_reference"]}
                                              for w in REGRESSION_WATCH if w in per}},
            "r8_advisory": {"F1_drop_net_le_0": vs_scale["net"] <= 0,
                            "advance_conditions": {"net_ge_2": vs_scale["net"] >= 2,
                                                   "activation_not_implementation_failure": status != "implementation_failure",
                                                   "half_of_wins_in_T": len(wins_in_t) * 2 >= len(vs_scale["wins"]),
                                                   "tier2_null_M": "check the tier-2 witness report"},
                            "net_exactly_1": vs_scale["net"] == 1,
                            "note": "Restates the review's rules; the Campaign gates and the supervisor decide."}}
        ok = ok and not violations and not miss
        print(f"candidate: {len(keys)} paired with SCALE; RC4 violations {violations or 'none'}; unmatched {miss or 'none'}; "
              f"short-run rate {rate} ({status}); divergence {report['candidate']['activation']['divergence_share']}; "
              f"c {report['candidate']['compute']['c_vs_native']}")
    for name in ("null", "baseline"):
        dirs = getattr(args, name)
        if dirs:
            res = m_check(load_arm(dirs), greedy_ref)
            report[name] = dict(res, arm_dirs=dirs)
            ok = ok and not res["not_M"]
            print(f"{name}: {res['cases']} cases vs greedy map; not M: {res['not_M'] or 'none'}")
    report["ok"] = ok
    _write(args.out, report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
