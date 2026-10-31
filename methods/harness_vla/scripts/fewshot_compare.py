"""Few-shot (memory) mode report: pair harness-vla-sol-fewshot with the frozen baseline and zero-shot harness-vla-sol.

Usage:
  python fewshot_compare.py --fewshot DIR [DIR ...] --baseline DIR [DIR ...] --zeroshot DIR [DIR ...]
                            [--dev DIR ...] [--episodes 1,2,3,4] [--out summary.json]
  --fewshot / --baseline / --zeroshot: `hs eval --out` dirs (records in <arm>/records.jsonl); records of the same
  arm are concatenated (Goal-T + Goal-S, split processes). --dev: `hs prepare --out` dirs (dev_records.jsonl).
Pairs by case key restricted to --episodes (duplicate instructions disambiguated by occurrence order, as in
merge_arms.py), and checks scene identity: every few-shot case's first observation hash must equal the zero-shot
record's for the same case, and must differ from the development reference scene's.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

from merge_arms import disambiguate

PRICE_PROMPT, PRICE_COMPLETION = 5e-6, 25e-6   # USD per token, working estimate for gpt-6.1-sol


def load(dirs: list, arm: str, dev: bool = False) -> list:
    out = []
    for d in dirs:
        d = Path(d)
        f = d / "dev_records.jsonl" if dev else d / arm / "records.jsonl"
        if not f.exists():
            f = next(d.rglob("records.jsonl"))
        # group by task order within a file before disambiguation (each file holds tasks in benchmark order)
        out += disambiguate([json.loads(x) for x in f.read_text().splitlines() if x.strip()])
    return out


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def usage(recs: list) -> dict:
    p = c = calls = 0
    wall = 0.0
    for r in recs:
        wall += r.get("wall_seconds", 0.0)
        for e in r.get("events") or []:
            if e.get("t") == "llm":
                u = e.get("usage") or {}
                p += int(u.get("prompt_tokens", 0))
                c += int(u.get("completion_tokens", 0))
                calls += 1
    return {"episodes": len(recs), "llm_calls": calls, "prompt_tokens": p, "completion_tokens": c,
            "usd_estimate": round(p * PRICE_PROMPT + c * PRICE_COMPLETION, 2), "wall_seconds": round(wall, 1)}


def key(r: dict) -> tuple:
    return (r["benchmark"].split("@")[0], r["task"], r["episode_idx"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fewshot", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--zeroshot", nargs="+", required=True)
    ap.add_argument("--dev", nargs="*", default=[])
    ap.add_argument("--episodes", default="1,2,3,4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps = {int(x) for x in a.episodes.split(",")}
    few = [r for r in load(a.fewshot, "candidate") if r["episode_idx"] in eps]
    base = {key(r): r for r in load(a.baseline, "baseline") if r["episode_idx"] in eps}
    zero = {key(r): r for r in load(a.zeroshot, "candidate") if r["episode_idx"] in eps}
    fewk = {key(r): r for r in few}
    if any(r["episode_idx"] == 0 for r in few):
        raise SystemExit("LEAK: a few-shot evaluation record has episode_idx 0 (the reference scene)")
    invalid = [k for k, r in fewk.items() if not r.get("valid")]
    summary: dict = {"cases_fewshot": len(fewk), "invalid_fewshot": [list(k) for k in invalid]}

    # scene identity
    dev = load(a.dev, "", dev=True) if a.dev else []
    ref_hash = {}
    for r in dev:
        if r.get("obs_sha256"):
            ref_hash.setdefault(r["task"].split(" [dup")[0], set()).add(r["obs_sha256"][0])
    same_as_zero = diff = same_as_ref = 0
    for k, r in fewk.items():
        h = (r.get("obs_sha256") or [None])[0]
        z = zero.get(k)
        if z and z.get("obs_sha256"):
            same_as_zero += h == z["obs_sha256"][0]
            diff += h != z["obs_sha256"][0]
        same_as_ref += h in ref_hash.get(k[1].split(" [dup")[0], set())
    summary["scene_check"] = {"first_obs_equal_to_zeroshot_case": same_as_zero, "differs": diff,
                              "equal_to_reference_scene": same_as_ref}

    rows = {}
    for name, other in (("baseline", base), ("zeroshot", zero)):
        cells = defaultdict(lambda: {"cases": 0, "fewshot": 0, name: 0, "wins": 0, "losses": 0})
        for k, r in fewk.items():
            o = other.get(k)
            if o is None or not r.get("valid") or not o.get("valid"):
                continue
            for c in (k[1].split("/")[0], "all"):
                row = cells[c]
                row["cases"] += 1
                row["fewshot"] += bool(r["success"])
                row[name] += bool(o["success"])
                row["wins"] += bool(r["success"]) and not o["success"]
                row["losses"] += bool(o["success"]) and not r["success"]
        for row in cells.values():
            row["sign_p"] = round(sign_p(row["wins"], row["losses"]), 4)
        rows[f"fewshot_vs_{name}"] = dict(cells)
    summary.update(rows)
    # per task outcomes
    per_task = defaultdict(lambda: [0, 0, 0, 0])
    for k, r in fewk.items():
        t = per_task[k[1]]
        t[0] += 1
        t[1] += bool(base.get(k, {}).get("success"))
        t[2] += bool(zero.get(k, {}).get("success"))
        t[3] += bool(r.get("success"))
    summary["per_task_n_base_zero_few"] = {t: v for t, v in sorted(per_task.items())}
    summary["usage"] = {"evaluation_fewshot": usage(few), "zeroshot_same_cases": usage(list(zero.values())),
                        "development": usage(dev)}
    if dev:
        rounds = defaultdict(lambda: [0, 0])
        for r in dev:
            rounds[(r["task"].split("/")[0], r["episode_idx"] + 1)][0] += 1
            rounds[(r["task"].split("/")[0], r["episode_idx"] + 1)][1] += bool(r.get("success"))
        summary["development_attempts"] = {f"{c} attempt {k}": f"{s}/{n}" for (c, k), (n, s) in sorted(rounds.items())}
    text = json.dumps(summary, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
