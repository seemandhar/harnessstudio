#!/usr/bin/env python3
"""Pair greedy-escape screen outputs (plan.json from write_screen_configs.py) with the development map records
(runs/scalepp-20261102/eval/devbase/shards: greedy = baseline-tT/baseline, SCALE = candidate-tT/candidate) and apply
the predeclared rules (proposal review GE4-GE7; protocol r1 amendment A R3/R8; source review
reviews/greedy-escape-src-review.json S1-S4, F8, F13). Evaluator tool; run only after the jobs finished.

    python screen_summary.py --plan EVAL_DIR/plan.json [--phase grid|confirm|null|all] [--out SUMMARY.json]
                             [--null-classification NULLCLASS.json]   # {"tTiI": "R" | "X"} for differing null scenes

Per scene: candidate/greedy/SCALE success (trace records' `success`); windows (accounting.extra['greedy_escape']);
window decisions = |U [s, s+40) n [0, accounting.steps)| (F8: the extras' escape_decisions undercount by one when an
episode ends inside a window); first executed-action divergence from the greedy map record (vla-eval jsonl action rows,
never whole-obs hashes: F7); retention (untriggered => rows bitwise equal to the greedy record, GE5); S3 trigger parity:
the record's first fire (step, reasons) == the frozen StallMonitor replay (first_fire below, also used by
tests/test_replay_devmap.py) of the paired greedy record, for candidate AND null rows (a fire on the episode's last
executed action is unobservable, F8, and expected as none); F13: the run manifest's method fingerprint must be
f03731f1..., namespace 'hs', case namespace ''.

Decision (order): INFRASTRUCTURE if any grid gate fails; then the null (D1): not summarized -> PENDING, INFRASTRUCTURE
(X, invalid, parity, missing states) -> INFRASTRUCTURE (null), unclassified differing scenes -> PENDING, all before any
quality verdict; then the remaining rules. Grid gates: INFRASTRUCTURE if any gate fails (grid: 40 episodes; confirm, when present: 60 episodes [S1]; every
row valid, fingerprint/namespace bound, S3 parity, retention, no divergence before the first window, no missing states);
DROP F2 if the candidate inference ratio c > 4.0 (S4); DROP F1 if grid net vs SCALE <= 0; MECHANISM REJECTED if < 2 of
the 6 triggered SCALE-solvable greedy failures are rescued; then GE7: net == +1 -> advance iff pooled net >= +3 AND
init 4-9 net >= 0; net >= +2 -> advance only after GE4 (init 4-9 net vs SCALE >= 0). ADVANCE also requires the null
gate (S2): all null scenes valid with states, S3 parity, executed rows == greedy record (M) or classified R by the
evaluator (P-20261101-16: R only if the first difference follows an image-only obs difference at a decision with equal
proprio and equal earlier actions; else X), X == 0, outcome discordance vs the greedy records <= 2 and |net| <= 2.
R/X for a differing null scene follows the predeclared rule runs/scalepp-20261102/campaign-inputs/ge-null-rx-rule.md: ONE
baseline-arm replay of the scene on the same server/launch path; replay rows == map greedy record -> X (harness leak,
screen INFRASTRUCTURE); replay also differs -> R (counted against the tolerance). The evaluator records the result per
scene in null-classification.json as {"tTiI": "R" | "X"} (or {"tTiI": {"class": "R"|"X", "replay": PATH, ...}}).

Disclosure (F6): window decisions use the same CRN seeds as the SCALE map arm but a different observation and decoder
state, so window actions are NOT paired with the SCALE record; SCALE pairing is outcome-level by scene only.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

SHARDS = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/scalepp-20261102/eval/devbase/shards")
RESCUE6 = ["t0i0", "t0i1", "t2i2", "t3i0", "t7i0", "t7i2"]
FINGERPRINT = "f03731f11b84f97306ba5080b1def6b72e64468503a25813f46a700e6f9f9fe1"
GREEDY_S_PER_STEP = 0.756          # review G-F4 basis, used only when a map record lacks policy time
WINDOW = 40
EXPECTED_EPISODES = {"grid": 40, "confirm": 60}
NULL_MAX_DISCORDANT, NULL_MAX_ABS_NET = 2, 2      # protocol r1 R3
C_FLAG, C_DROP = 1.2, 4.0                          # S4 / R8 F2


# ---- the frozen trigger replay (S3; the same function tests/test_replay_devmap.py uses) ---------------------------
def first_fire(actions, states):
    """First (step, reasons) of the frozen StallMonitor on recorded rows (action k, post-step state k), or (None, None)."""
    from hs_greedy_escape.monitor import StallMonitor
    mon = StallMonitor()
    for a, s in zip(actions, states):
        s = np.asarray(s, dtype=np.float32)
        ev = mon.step(np.asarray(a, dtype=np.float32), float(s[6] - s[7]))
        if ev is not None:
            return ev["step"], list(ev["reasons"])
    return None, None


# ---- record access ------------------------------------------------------------------------------------------------
def records(arm_dir: Path) -> dict:
    out = {}
    path = arm_dir / "records.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["episode_idx"])] = r
    return out


def episode_rows(arm_dir: Path, k: int):
    """(actions, states) of the k-th recorded episode of an arm directory, or (None, None)."""
    fs = glob.glob(str(arm_dir / ("vla_eval/episodes/*/task0000_ep%04d_*.jsonl" % k)))
    if len(fs) != 1:
        return None, None
    rows = [json.loads(line) for line in open(fs[0]) if line.strip()]
    return [r["action"] for r in rows], [r["state"] for r in rows]


def first_diff(a, b):
    if a is None or b is None:
        return "missing"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def manifest_ok(arm_dir: Path) -> tuple[bool, str]:
    path = arm_dir / "manifest.json"
    if not path.is_file():
        return False, "no manifest.json"
    m = json.loads(path.read_text())
    problems = []
    if m.get("method_fingerprint") != FINGERPRINT:
        problems.append(f"fingerprint {str(m.get('method_fingerprint'))[:8]}")
    if m.get("namespace") != "hs":
        problems.append(f"namespace {m.get('namespace')!r}")
    if m.get("case_namespace") != "":
        problems.append(f"case_namespace {m.get('case_namespace')!r}")
    return not problems, "; ".join(problems)


def window_decisions(starts, steps) -> int:
    covered = set()
    for s in starts:
        covered |= set(range(int(s), min(int(s) + WINDOW, int(steps))))
    return len(covered)


# ---- rows ---------------------------------------------------------------------------------------------------------
def scene_rows(plan: dict, phase: str, shards: Path = SHARDS) -> list:
    out = []
    for job in plan["jobs"]:
        if job["phase"] == "tier2" or (phase != "all" and job["phase"] != phase):
            continue
        arm = job["arms"].split(",")[-1]
        arm_dir = Path(job["out"]) / arm
        recs = records(arm_dir)
        mok, mwhy = manifest_ok(arm_dir)
        t = job["task"]
        gdir, sdir = shards / f"baseline-t{t}/baseline", shards / f"candidate-t{t}/candidate"
        greedy, scale = records(gdir), records(sdir)
        for k, init in enumerate(job["inits"]):
            r = recs.get(init) or {}
            acc = r.get("accounting") or {}
            ge = (acc.get("extra") or {}).get("greedy_escape") or {}
            steps = int(acc.get("steps") or 0)
            starts = list(ge.get("window_starts") or [])
            g_act, g_state = episode_rows(gdir, init)
            c_act, _ = episode_rows(arm_dir, k)
            div = first_diff(c_act, g_act)
            exp_step, exp_reasons = first_fire(g_act, g_state) if g_act is not None else ("missing", None)
            if isinstance(exp_step, int) and exp_step >= steps - 1:
                exp_step, exp_reasons = None, None   # a fire on the last executed action is never recorded (F8)
            fires = list(ge.get("fires") or [])
            got = (fires[0]["step"], list(fires[0]["reasons"])) if fires else (None, None)
            g_rec = greedy.get(init) or {}
            row = {"scene": f"t{t}i{init}", "phase": job["phase"], "arm": arm, "job": job["id"],
                   "valid": bool(r.get("valid")), "manifest_ok": mok, "manifest_problem": mwhy,
                   "success": r.get("success"), "greedy": g_rec.get("success"),
                   "scale": (scale.get(init) or {}).get("success"),
                   "steps": steps, "windows": len(starts), "window_starts": starts,
                   "window_decisions": window_decisions(starts, steps),
                   "first_fire": list(got), "expected_first_fire": [exp_step, exp_reasons],
                   "trigger_parity": got == (exp_step, exp_reasons),
                   "first_divergence_vs_greedy": div, "width_missing_steps": ge.get("width_missing_steps"),
                   "policy_seconds": acc.get("policy_seconds"),
                   "greedy_policy_seconds": (g_rec.get("accounting") or {}).get("policy_seconds"),
                   "greedy_steps": (g_rec.get("accounting") or {}).get("steps")}
            if arm == "candidate":
                row["retention_ok"] = (div is None) if not starts else None
                # any divergence BEFORE the first window step is infrastructure; at the window step is the prediction
                row["divergence_before_window"] = bool(starts) and isinstance(div, int) and div < starts[0]
                row["divergence_at_window"] = (div == starts[0]) if (starts and isinstance(div, int)) else None
            if arm == "null":
                row["null_equals_greedy_record"] = div is None
            out.append(row)
    return out


# ---- phase summaries ----------------------------------------------------------------------------------------------
def net(rows_, ref):
    wins = sum(1 for r in rows_ if r["success"] and not r[ref])
    losses = sum(1 for r in rows_ if not r["success"] and r[ref])
    return [wins, losses, wins - losses]


def infra_problems(rs: list, expected: int | None) -> list:
    p = []
    if expected is not None and len(rs) != expected:
        p.append(f"{len(rs)} episodes, expected {expected}")
    p += [f"{r['scene']} invalid" for r in rs if not r["valid"]]
    p += [f"{r['scene']} manifest: {r['manifest_problem']}" for r in rs if not r["manifest_ok"]]
    p += [f"{r['scene']} trigger parity {r['first_fire']} != {r['expected_first_fire']}" for r in rs
          if not r["trigger_parity"]]
    p += [f"{r['scene']} missing states" for r in rs if r.get("width_missing_steps")]
    p += [f"{r['scene']} retention fault" for r in rs if r.get("retention_ok") is False]
    p += [f"{r['scene']} divergence before the first window" for r in rs if r.get("divergence_before_window")]
    return p


def inference_ratio(rs: list) -> dict:
    ps = sum(float(r["policy_seconds"] or 0.0) for r in rs)
    st = sum(int(r["steps"] or 0) for r in rs)
    paired = [r for r in rs if r["greedy_policy_seconds"] and r["greedy_steps"]]
    if paired and len(paired) == len(rs):
        base = sum(float(r["greedy_policy_seconds"]) for r in rs) / sum(int(r["greedy_steps"]) for r in rs)
        basis = "map greedy records of the same scenes"
    else:
        base, basis = GREEDY_S_PER_STEP, "predeclared 0.756 s/step (review G-F4)"
    c = (ps / st) / base if st else None
    return {"c": c, "s_per_step": (ps / st) if st else None, "greedy_s_per_step": base, "basis": basis,
            "flag_gt_1_2": bool(c is not None and c > C_FLAG), "drop_gt_4": bool(c is not None and c > C_DROP)}


def phase_summary(rs: list, phase: str, expected: int | None) -> dict:
    s = {"episodes": len(rs), "valid": sum(r["valid"] for r in rs),
         "candidate": sum(bool(r["success"]) for r in rs), "greedy": sum(bool(r["greedy"]) for r in rs),
         "scale": sum(bool(r["scale"]) for r in rs), "vs_greedy_w_l_net": net(rs, "greedy"),
         "vs_scale_w_l_net": net(rs, "scale"), "episodes_with_window": sum(1 for r in rs if r["windows"]),
         "window_decisions": sum(r["window_decisions"] for r in rs),
         "diverged_at_first_window": [sum(1 for r in rs if r.get("divergence_at_window") is True),
                                      sum(1 for r in rs if isinstance(r.get("divergence_at_window"), bool))],
         "infrastructure": infra_problems(rs, expected),
         "inference_ratio": inference_ratio(rs)}
    if phase == "grid":
        s["rescued_of_6"] = [r["scene"] for r in rs if r["scene"] in RESCUE6 and r["success"]]
    return s


def null_summary(rs: list, expected: int, classification: dict) -> dict:
    problems = infra_problems(rs, expected)
    differs = [r["scene"] for r in rs if not r["null_equals_greedy_record"]]
    def label(v):
        return v.get("class") if isinstance(v, dict) else v
    cls = {sc: label(classification.get(sc)) for sc in differs}
    w = sum(1 for r in rs if r["success"] and not r["greedy"])
    l = sum(1 for r in rs if not r["success"] and r["greedy"])
    s = {"episodes": len(rs), "infrastructure": problems, "equal_to_greedy_record": len(rs) - len(differs),
         "differs": [(r["scene"], r["first_divergence_vs_greedy"]) for r in rs if not r["null_equals_greedy_record"]],
         "classification": cls, "X": sum(1 for v in cls.values() if v == "X"),
         "unclassified": [sc for sc, v in cls.items() if v not in ("R", "X")],
         "discordant_vs_greedy": w + l, "net_vs_greedy": w - l, "inference_ratio": inference_ratio(rs)}
    if problems or s["X"]:
        s["status"] = "INFRASTRUCTURE"
    elif s["unclassified"]:
        s["status"] = "PENDING (evaluator R/X classification of differing scenes)"
    elif s["discordant_vs_greedy"] > NULL_MAX_DISCORDANT or abs(s["net_vs_greedy"]) > NULL_MAX_ABS_NET:
        s["status"] = "OUT_OF_TOLERANCE"
    else:
        s["status"] = "WITHIN_TOLERANCE"
    return s


# ---- decision -----------------------------------------------------------------------------------------------------
def decide(g: dict | None, c: dict | None, n: dict | None) -> str:
    if g is None:
        return "NO GRID SUMMARY"
    if g["infrastructure"]:
        return "INFRASTRUCTURE (grid): " + "; ".join(g["infrastructure"][:5])
    # D1 (reviews/ge-screen-diffcheck.json): the null is part of the instrument. A missing, leaking (X / invalid / parity /
    # missing states) or unclassified null yields no quality verdict at all (no DROP, no MECHANISM REJECTED).
    # OUT_OF_TOLERANCE (R-class noise beyond 2/2) only blocks ADVANCE below.
    if n is None:
        return "PENDING null (S2): not summarized; no outcome verdict"
    if n["status"] == "INFRASTRUCTURE":
        return "INFRASTRUCTURE (null): " + "; ".join((n.get("infrastructure") or [])[:5] + (
            [f"X = {n.get('X')}"] if n.get("X") else []))
    if n["status"].startswith("PENDING"):
        return "PENDING null R/X classification (campaign-inputs/ge-null-rx-rule.md): " + ", ".join(n.get("unclassified") or [])
    if g["inference_ratio"]["drop_gt_4"]:
        return "DROP (R8 F2: inference ratio > 4.0)"
    gnet = g["vs_scale_w_l_net"][2]
    if gnet <= 0:
        return "DROP (R8 F1: net vs SCALE <= 0)"
    if len(g["rescued_of_6"]) < 2:
        return "MECHANISM REJECTED (< 2 of the 6 triggered SCALE-solvable greedy failures rescued)"
    if c is None:
        return ("PENDING GE4 extension: advance iff pooled net (grid + init 4-9) >= +3 AND init 4-9 net >= 0"
                if gnet == 1 else "PENDING GE4 confirmation (init 4-9 net vs SCALE >= 0)")
    if c["infrastructure"]:
        return "INFRASTRUCTURE (confirm): " + "; ".join(c["infrastructure"][:5])
    if c["inference_ratio"]["drop_gt_4"]:
        return "DROP (R8 F2: inference ratio > 4.0 on the confirmation)"
    cnet = c["vs_scale_w_l_net"][2]
    outcome_ok = (cnet >= 0 and gnet + cnet >= 3) if gnet == 1 else cnet >= 0
    if not outcome_ok:
        return "DO NOT ADVANCE (GE7 extension rule)" if gnet == 1 else "DO NOT ADVANCE (GE4 failed)"
    if n["status"] != "WITHIN_TOLERANCE":
        return f"NOT ADVANCE YET: outcome rules pass, null {n['status']}"
    return "ADVANCE (" + ("GE7 extension passed" if gnet == 1 else "GE4 passed") + "; null within tolerance)"


def summarize(plan: dict, phase: str = "all", shards: Path = SHARDS, classification: dict | None = None,
              expected: dict | None = None) -> dict:
    """expected: episode counts per phase (default grid 40 / confirm 60; tests pass smaller synthetic counts)."""
    expected = dict(EXPECTED_EPISODES if expected is None else expected)
    allrows = scene_rows(plan, phase, shards)
    summary: dict = {"rows": allrows,
                     "disclosure": "window decisions use the SCALE map's CRN seeds but a different observation and "
                                   "decoder state: no action-level pairing with SCALE; SCALE pairing is outcome-level "
                                   "by scene only (F6)"}
    for ph in ("grid", "confirm"):
        rs = [r for r in allrows if r["phase"] == ph and r["arm"] == "candidate"]
        if rs:
            summary[ph] = phase_summary(rs, ph, expected.get(ph))
    nulls = [r for r in allrows if r["arm"] == "null"]
    if nulls or any(j["phase"] == "null" for j in plan["jobs"]):
        expected = sum(1 for j in plan["jobs"] if j["phase"] == "null")
        summary["null"] = null_summary(nulls, expected, classification or {})
    summary["ge7_decision"] = decide(summary.get("grid"), summary.get("confirm"), summary.get("null"))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--phase", default="all", choices=("grid", "confirm", "null", "all"))
    ap.add_argument("--map-shards", default=str(SHARDS))
    ap.add_argument("--null-classification", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    cls = json.loads(Path(args.null_classification).read_text()) if args.null_classification else {}
    summary = summarize(plan, args.phase, Path(args.map_shards), cls)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    json.dump({k: v for k, v in summary.items() if k != "rows"}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
