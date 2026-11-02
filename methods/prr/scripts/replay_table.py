"""C3 replay-equivalence tables generated FROM the package (hs_prr.planner), streaming one episode at a time.

    harnessstudio/.venv/bin/python replay_table.py OUT_DIR        # writes c3-table-dev80.json, c3-table-s2.json, summary

dev80: runs/rtcfpp-20261101/eval/dev-baseline/shards/identity-*/baseline (libero_10 init 30-37, seed 7; refuses others).
Per episode: recorded post-step states S and executed actions A from the vla-eval episode jsonl (only 'state' and
'action' are read from it); the identity success flag comes from records.jsonl (development identity outcome, used for
reach/exposure only, never by the planner). The planner is fed exactly as the live controller: at each decision step
d = 5, 10, ... (< T) decide(d, S[d-1]); then note_row(s, S[s-1], A[s]) for every step s. The scan stops at the first
valid plan (= the first restoration: the first action divergence from identity, C9 witness).

Replay compatibility (C3 'must exactly match the replay semantics'): the same planner with the C2/C4 guards disabled and
one route at a time is compared with ideation/scripts-r6-recovery/frozen_rule_replay.json (fire and checkpoint of A and B).

s2: runs/rtcfpp-20261101/analysis/s2_compact.npz (init 10-29, decision-time states only): C2 is evaluated with the
latest decision-time position <= s as p_s (per-step states are not stored; disclosed approximation).
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from hs_prr.planner import Params, Planner

RUN = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtcfpp-20261101")
SHARDS = RUN / "eval/dev-baseline/shards"
REPLAY = RUN / "ideation/scripts-r6-recovery/frozen_rule_replay.json"
S2_NPZ = RUN / "analysis/s2_compact.npz"
S2_IDX = RUN / "analysis/s2_index.json"
HARD, EASY = [2, 3, 5, 6, 8, 9], [0, 1, 4, 7]
CARD = Path(__file__).resolve().parents[1] / "hs_prr/paper.yaml"


def params_from_card() -> Params:
    import yaml
    return Params.from_mapping(yaml.safe_load(CARD.read_text())["params"])


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dev_episodes():
    """Yield (task, init, success, S, A) one episode at a time, official order (task, init)."""
    index = []
    for d in sorted(glob.glob(str(SHARDS / "identity-t*-i*-*"))):
        rec = Path(d) / "baseline/records.jsonl"
        pos = -1
        for line in rec.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pos += 1
            t, i = int(r["task"].rsplit("#", 1)[1]), int(r["episode_idx"])
            assert 30 <= i <= 37, ("non-dev scene", d, i)
            assert r["valid"], (d, i)
            eps = sorted(glob.glob(str(Path(d) / f"baseline/vla_eval/episodes/*/task*_ep{pos:04d}_*.jsonl")))
            assert len(eps) == 1, (d, pos, eps)
            index.append((t, i, bool(r["success"]), int(r["accounting"]["steps"]), eps[0]))
    for t, i, ok, steps, f in sorted(index):
        S, A = [], []
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                S.append(r["state"]); A.append(r["action"])
        S, A = np.asarray(S, np.float64), np.asarray(A, np.float64)
        assert len(S) == steps, (t, i, len(S), steps)
        yield t, i, ok, S, A


def run_planner(params: Params, S_dec: dict, A: np.ndarray, pre_pos, route: str = "AB"):
    """Drive the planner like the live controller until the first valid plan. S_dec: decision step -> state."""
    pl = Planner(params)
    if route == "A":
        pl.b_armed = False
    elif route == "B":
        pl.a_armed = False
    events, first = [], None
    T = len(A)
    for s in range(T):
        if s % params.decision_steps == 0 and s in S_dec:
            plan = pl.decide(s, S_dec[s])
            events += [{"name": n, **f} for n, f in pl.drain()]
            if plan is not None and plan.abstain is None:
                first = plan
                break
        pl.note_row(s, pre_pos(s), A[s])
    return first, events


def compat_params(p: Params) -> Params:
    d = dict(vars(p)) if hasattr(p, "__dict__") else {k: getattr(p, k) for k in p.__dataclass_fields__}
    d.update(custody_tol_m=math.inf, horizon_guard_last_step=10 ** 9)
    return Params(**d)


def first_fire(events, route):
    for e in events:
        if e["name"] == "prr_fire" and e["route"] == route:
            return e
    return None


def scene_row(params, t, i, ok, S_dec, A, pre_pos, width_at):
    first, events = run_planner(params, S_dec, A, pre_pos)
    fires = [{k: e.get(k) for k in ("route", "fire", "checkpoint", "L", "latch", "abstain", "checks")}
             for e in events if e["name"] == "prr_fire"]
    row = {"key": f"t{t}i{i}", "task": t, "init": i, "hard": t in HARD, "identity_success": ok, "steps": len(A),
           "events": [{k: v for k, v in e.items() if k not in ("checkpoint_state", "fire_state")} for e in events],
           "fires": fires, "activated": first is not None}
    if first is not None:
        f, L = first.fire, first.L
        row["first_restoration"] = {
            "route": first.route, "fire": f, "checkpoint": first.checkpoint, "L": L, "latch": first.latch,
            "latch_closed": first.latch > 0, "width_at_fire": width_at(f),
            "holding": bool(first.latch > 0 and width_at(f) is not None and width_at(f) > 0.004),
            "handback_range": [f + L + params.settle_steps, f + L + params.correction_max_decisions * params.decision_steps
                               + params.settle_steps],
            "native_steps_left_min": 520 - (f + L + params.correction_max_decisions * params.decision_steps + params.settle_steps),
            "rows_sha256_float32": hashlib.sha256(np.ascontiguousarray(first.rows.astype(np.float32)).tobytes()).hexdigest(),
            "checks": first.checks,
        }
    return row


def dev_table(params):
    rows, compat = [], []
    replay = json.loads(REPLAY.read_text())["dev"]
    cp_params = compat_params(params)
    for t, i, ok, S, A in dev_episodes():
        T = len(A)
        S_dec = {d: S[d - 1] for d in range(params.decision_steps, T, params.decision_steps)}
        pre_pos = lambda s, S=S: S[s - 1] if s > 0 else None
        width_at = lambda f, S=S: float(S[f - 1][6] - S[f - 1][7])
        rows.append(scene_row(params, t, i, ok, S_dec, A, pre_pos, width_at))
        ref = replay[f"t{t}i{i}"]
        c = {"key": f"t{t}i{i}"}
        for route in ("A", "B"):
            _, ev = run_planner(cp_params, S_dec, A, pre_pos, route)
            fe = first_fire(ev, route)
            mine = None if fe is None or fe["checkpoint"] is None else {"fire": fe["fire"], "checkpoint": fe["checkpoint"]}
            r = ref[route]
            theirs = None if r is None or r.get("checkpoint") is None else {"fire": r["fire"], "checkpoint": r["checkpoint"]}
            # the replay's route B records a stall fire with no checkpoint as {'checkpoint': None}: both mean abstain
            c[route] = {"package": mine, "replay": theirs, "equal": mine == theirs}
            if route == "B" and mine != theirs and r is not None and fe is None:
                c[route]["note"] = "replay window shorter than 60 steps at the episode tail"
        compat.append(c)
        del S, A
    return rows, compat


def s2_table(params):
    z = np.load(S2_NPZ, allow_pickle=False)
    rows = []
    for r in json.loads(S2_IDX.read_text()):
        k = r["key"]
        assert 10 <= r["init"] <= 29
        SD, A, DS = z[k + "_state"].astype(np.float64), z[k + "_act"].astype(np.float64), [int(d) for d in z[k + "_dstep"]]
        S_dec = dict(zip(DS, SD))
        dsorted = sorted(DS)

        def pre_pos(s, S_dec=S_dec, dsorted=dsorted):
            prev = [d for d in dsorted if d <= s]
            return S_dec[prev[-1]] if prev else None
        width_at = lambda f, S_dec=S_dec: float(S_dec[f][6] - S_dec[f][7]) if f in S_dec else None
        row = scene_row(params, int(r["task"]), int(r["init"]), bool(r["success"]), S_dec, A, pre_pos, width_at)
        row["key"] = k
        rows.append(row)
    return rows


def summarize(rows, sel=lambda r: True):
    act = [r for r in rows if sel(r) and r["activated"]]
    out = {}
    for name, f in (("hard_fail", lambda r: r["hard"] and not r["identity_success"]),
                    ("hard_succ", lambda r: r["hard"] and r["identity_success"]),
                    ("easy_fail", lambda r: not r["hard"] and not r["identity_success"]),
                    ("easy_succ", lambda r: not r["hard"] and r["identity_success"])):
        out[name] = {"activated": sorted(r["key"] for r in act if f(r)),
                     "of": sum(1 for r in rows if sel(r) and f(r))}
    out["route_A"] = sorted(r["key"] for r in act if r["first_restoration"]["route"] == "A")
    out["route_B"] = sorted(r["key"] for r in act if r["first_restoration"]["route"] == "B")
    return out


def official(rows, tasks):
    order = {t: k for k, t in enumerate(tasks)}
    return sorted((r for r in rows if r["task"] in order), key=lambda r: (order[r["task"]], r["init"]))


def lists(rows):
    hard = official(rows, HARD)
    easy = official(rows, EASY)
    a_scenes = [r for r in hard if r["activated"] and r["first_restoration"]["route"] == "A"]
    q, seen = [], set()
    for r in a_scenes:                         # C6: first route-A scene per task in official hard48 order ...
        if r["task"] not in seen:
            q.append(r["key"]); seen.add(r["task"])
    nxt = next((r["key"] for r in a_scenes if r["key"] not in q), None)   # ... plus the next route-A scene
    if nxt:
        q.append(nxt)
    q = [r["key"] for r in a_scenes if r["key"] in q]                    # run order = official order
    act = [r for r in hard if r["activated"]]
    hold = next((r["key"] for r in act if r["first_restoration"]["route"] == "A" and r["first_restoration"]["holding"]), None)
    rel = next((r["key"] for r in act if r["first_restoration"]["route"] == "A" and not r["first_restoration"]["latch_closed"]), None)
    rb = next((r["key"] for r in act if r["first_restoration"]["route"] == "B"), None)
    more = next((r["key"] for r in act if r["key"] not in (hold, rel, rb)), None)
    return {
        "q_recur_batch_C6": q,
        "tier2_shadow_null_C10": {"route_A_holding": hold, "route_A_released": rel, "route_B": rb, "one_more": more},
        "spot_checks_C9": {"hard48": [r["key"] for r in hard if not r["activated"]][:2],
                           "easy32": [r["key"] for r in easy if not r["activated"]][:2]},
        "prr_hard48_activated_official_order": [r["key"] for r in act],
        "prr_easy32_activated_official_order": [r["key"] for r in easy if r["activated"]],
    }


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = params_from_card()
    import hs_prr
    pkg = Path(hs_prr.__file__).resolve().parent
    prov = {"planner_sha256": sha(pkg / "planner.py"), "geometry_sha256": sha(pkg / "geometry.py"),
            "card_sha256": sha(CARD), "script_sha256": sha(__file__), "replay_reference_sha256": sha(REPLAY),
            "params": {k: getattr(params, k) for k in params.__dataclass_fields__}}
    dev, compat = dev_table(params)
    mism = [c for c in compat if not (c["A"]["equal"] and c["B"]["equal"])]
    dev_doc = {"provenance": prov, "visibility": "development_only (dev80 identity, init 30-37)", "rows": dev,
               "replay_compat": compat, "replay_compat_mismatches": mism,
               "summary": {"dev80": summarize(dev), "campaign30": summarize(dev, lambda r: r["init"] <= 32)},
               "lists": lists(dev)}
    (out / "c3-table-dev80.json").write_text(json.dumps(dev_doc, indent=1, default=float))
    s2 = s2_table(params)
    s2_doc = {"provenance": {**prov, "s2_npz_sha256": sha(S2_NPZ), "s2_index_sha256": sha(S2_IDX)},
              "visibility": "development_only (S2 init 10-29)", "approximation": "C2 uses decision-time positions",
              "rows": s2, "summary": summarize(s2)}
    (out / "c3-table-s2.json").write_text(json.dumps(s2_doc, indent=1, default=float))
    summ = {"provenance": prov,
            "c3-table-dev80.json": sha(out / "c3-table-dev80.json"), "c3-table-s2.json": sha(out / "c3-table-s2.json"),
            "dev80": dev_doc["summary"], "s2": s2_doc["summary"], "lists": dev_doc["lists"],
            "replay_compat_mismatches": mism,
            "first_restorations_dev80": {r["key"]: {k: r["first_restoration"][k] for k in
                                                    ("route", "fire", "checkpoint", "L", "latch", "holding", "width_at_fire",
                                                     "native_steps_left_min")}
                                         for r in dev if r["activated"]},
            "abstains_dev80": {r["key"]: [(f["route"], f["fire"], f["checkpoint"], f["abstain"]) for f in r["fires"]
                                          if f["abstain"]] for r in dev if any(f["abstain"] for f in r["fires"])}}
    (out / "c3-summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: summ[k] for k in ("dev80", "lists", "replay_compat_mismatches", "abstains_dev80")}, indent=1,
                     default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
