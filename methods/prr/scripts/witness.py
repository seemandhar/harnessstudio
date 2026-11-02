"""PRR evaluator hooks (review C6, C7, C9, C10). State-only and outcome-blind by construction.

Every trace is read through ``load_trace``: only HarnessStudio trace files ``<dir>/traces/*.json`` are accepted (paths
containing '_success', '_fail', 'vla_eval' or 'records.jsonl' are refused), and the JSON parser drops the keys
'success', 'metrics', 'reward' and 'done' at every nesting level while parsing (object_pairs_hook), so no outcome value
ever enters memory. Scenes are keyed 't<task id>i<init>' from the trace's own 'task' and 'episode_idx' fields.

    python witness.py q-recur  --prr DIR [DIR ...] --table c3-summary.json --out q-recur.json
        C6 + C7(W1): on the table's q_recur_batch_C6 scenes: recurrence = any decision-time tilt > 55 deg (frozen
        hs_prr.geometry.tilt_deg on the prr_d state events) at a native decision d with h < d <= h + 60 after the FIRST
        hand-back h (decisions at or after a second restoration start are excluded); >= 4/6 -> STOP (family closed).
        W1 = fraction of restorations in the batch whose hand-back pose error is <= 0.02 m and <= 10 deg (>= 0.8 passes).
        Also writes the C9 witness of the batch. Run it and record its output BEFORE any flag of the batch is read.
    python witness.py determinism --arm NAME=DIR [...] --identity DIR [DIR ...] --table c3-summary.json --out det.json
        C9: per scene, executed prefix bitwise equal to identity up to the first restoration start; first restoration
        (route, fire, checkpoint, L, latch) equal to the table; any restoration on a scene the table does not activate is
        an unpredicted fire (determinism/code fault); non-activated scenes must be whole-episode bitwise identity.
    python witness.py tier2-null --null DIR [DIR ...] --identity DIR [DIR ...] --table c3-summary.json --out t2.json
        C10: shadow-null actions bitwise equal to the recorded identity; the logged first plan (prr_restore_start in the
        shadow) equal to the table's first restoration; X = 0 means every listed scene matches.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

from hs_prr.geometry import tilt_deg

DROP = {"success", "metrics", "reward", "done"}
FORBIDDEN = ("_success", "_fail", "vla_eval", "records.jsonl")


def _hook(pairs):
    return {k: v for k, v in pairs if k not in DROP}


def load_trace(path: str) -> dict:
    p = str(path)
    if any(x in p for x in FORBIDDEN) or not p.endswith(".json") or os.sep + "traces" + os.sep not in p:
        raise ValueError(f"refusing to read {p}: only HarnessStudio traces/*.json (no outcome-named files)")
    with open(p) as fh:
        return json.load(fh, object_pairs_hook=_hook)


def scene_key(rec: dict) -> str:
    return f"t{int(rec['task'].rsplit('#', 1)[1])}i{int(rec['episode_idx'])}"


def load_dirs(dirs) -> dict:
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "traces", "*.json"))):
            rec = load_trace(f)
            out[scene_key(rec)] = {"file": f, "valid": rec.get("valid"), "invalid_reason": rec.get("invalid_reason"),
                                   "action_sha256": rec.get("action_sha256", []),
                                   "events": [e for e in rec.get("events", []) if e.get("t") == "event"]}
    return out


def ev(rec, name):
    return [e for e in rec["events"] if e.get("name") == name]


def first_div(a, b):
    n = min(len(a), len(b))
    k = next((i for i in range(n) if a[i] != b[i]), None)
    return k if k is not None else (n if len(a) != len(b) else None)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table_first(table: dict) -> dict:
    return table["first_restorations_dev80"]


def recurrence(rec: dict) -> dict:
    hb = ev(rec, "prr_handback")
    if not hb:
        return {"restored": False}
    h = int(hb[0]["step"])
    starts = [int(e["step"]) for e in ev(rec, "prr_restore_start")]
    nxt = min([s for s in starts if s > h], default=10 ** 9)
    samples = [(int(e["step"]), tilt_deg(e["state"][3:6])) for e in ev(rec, "prr_d") if h < int(e["step"]) <= h + 60
               and int(e["step"]) < nxt]
    return {"restored": True, "handback": h, "n_samples": len(samples), "max_tilt": max((t for _, t in samples), default=None),
            "recur": any(t > 55.0 for _, t in samples), "first_recur_step": next((s for s, t in samples if t > 55.0), None),
            "w1_pos_err_m": hb[0].get("pos_err_m"), "w1_rot_err_deg": hb[0].get("rot_err_deg")}


def witness_scene(key, rec, ident, table) -> dict:
    starts = ev(rec, "prr_restore_start")
    a, b = rec["action_sha256"], ident["action_sha256"] if ident else None
    out = {"scene": key, "valid": rec["valid"], "restorations": len(starts)}
    pred = table_first(table).get(key)
    if starts:
        s0 = starts[0]
        got = {k: s0[k] for k in ("route", "fire", "checkpoint", "L", "latch")}
        out["first_restoration"] = got
        out["predicted"] = None if pred is None else {k: pred[k] for k in ("route", "fire", "checkpoint", "L", "latch")}
        out["matches_table"] = out["predicted"] == got
        out["unpredicted_fire"] = pred is None
        if b is not None:
            out["prefix_bitwise_until_start"] = a[: s0["step"]] == b[: s0["step"]]
            out["first_divergence"] = first_div(a, b)
            out["divergence_at_start"] = out["first_divergence"] == s0["step"]
    else:
        out["predicted"] = pred
        out["missing_predicted_fire"] = pred is not None
        if b is not None:
            out["whole_episode_bitwise"] = a == b
    return out


def cmd_q_recur(args) -> dict:
    table = json.loads(Path(args.table).read_text())
    batch = table["lists"]["q_recur_batch_C6"]
    prr = load_dirs(args.prr)
    ident = load_dirs(args.identity) if args.identity else {}
    rows, w1 = [], []
    for k in batch:
        if k not in prr:
            rows.append({"scene": k, "missing": True})
            continue
        r = recurrence(prr[k])
        r["scene"] = k
        r["witness"] = witness_scene(k, prr[k], ident.get(k), table)
        rows.append(r)
        for hb in ev(prr[k], "prr_handback"):
            w1.append(bool(hb["pos_err_m"] <= 0.02 and hb["rot_err_deg"] <= 10.0))
    complete = all(not r.get("missing") and r.get("restored") for r in rows) and len(rows) == 6
    n_recur = sum(1 for r in rows if r.get("recur"))
    return {"kind": "C6 Q-recur + C7 W1 (state-only, outcome-blind)", "batch": batch, "complete": complete,
            "recur_count": n_recur, "of": len(rows), "decision": ("STOP_family_closed" if n_recur >= 4 else
                                                                  "continue") if complete else "incomplete",
            "w1": {"within": sum(w1), "of": len(w1), "fraction": (sum(w1) / len(w1)) if w1 else None,
                   "pass": bool(w1) and sum(w1) / len(w1) >= 0.8},
            "rows": rows}


def cmd_determinism(args) -> dict:
    table = json.loads(Path(args.table).read_text())
    ident = load_dirs(args.identity)
    arms = {}
    for spec in args.arm:
        name, d = spec.split("=", 1)
        recs = load_dirs([d])
        arms[name] = [witness_scene(k, r, ident.get(k), table) for k, r in sorted(recs.items())]
    faults = {n: [w for w in ws if w.get("unpredicted_fire") or w.get("matches_table") is False
                  or w.get("prefix_bitwise_until_start") is False or w.get("divergence_at_start") is False
                  or w.get("whole_episode_bitwise") is False or w.get("missing_predicted_fire")] for n, ws in arms.items()}
    return {"kind": "C9 determinism witness", "faults": faults, "n_faults": sum(len(v) for v in faults.values()),
            "arms": arms}


def cmd_tier2_null(args) -> dict:
    table = json.loads(Path(args.table).read_text())
    ident = load_dirs(args.identity)
    null = load_dirs(args.null)
    rows = []
    for k, r in sorted(null.items()):
        b = ident.get(k)
        s = ev(r, "prr_restore_start")
        pred = table_first(table).get(k)
        got = None if not s else {x: s[0][x] for x in ("route", "fire", "checkpoint", "L", "latch")}
        rows.append({"scene": k, "valid": r["valid"], "bitwise_identity": b is not None and r["action_sha256"] == b["action_sha256"],
                     "first_divergence": None if b is None else first_div(r["action_sha256"], b["action_sha256"]),
                     "logged_plan": got,
                     "table_plan": None if pred is None else {x: pred[x] for x in ("route", "fire", "checkpoint", "L", "latch")},
                     "plan_equal": got == (None if pred is None else {x: pred[x] for x in ("route", "fire", "checkpoint", "L", "latch")})})
    return {"kind": "C10 tier-2 shadow null", "X": sum(1 for r in rows if not r["bitwise_identity"]),
            "plan_mismatches": sum(1 for r in rows if not r["plan_equal"]), "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("q-recur"); q.add_argument("--prr", nargs="+", required=True)
    q.add_argument("--identity", nargs="*"); q.add_argument("--table", required=True); q.add_argument("--out", required=True)
    d = sub.add_parser("determinism"); d.add_argument("--arm", nargs="+", required=True)
    d.add_argument("--identity", nargs="+", required=True); d.add_argument("--table", required=True); d.add_argument("--out", required=True)
    t = sub.add_parser("tier2-null"); t.add_argument("--null", nargs="+", required=True)
    t.add_argument("--identity", nargs="+", required=True); t.add_argument("--table", required=True); t.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    fn = {"q-recur": cmd_q_recur, "determinism": cmd_determinism, "tier2-null": cmd_tier2_null}[args.cmd]
    res = fn(args)
    res["script_sha256"] = sha(__file__)
    res["table_sha256"] = sha(args.table)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "arms")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
