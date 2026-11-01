#!/usr/bin/env python3
"""Development-only leave-one-episode-out selection of PMA's (v_max, gamma) — the fixed rule of the critic's plan
review, Decision 3 (runs/rtcf-20261031/reviews/review-plan.md). Run once on the LIBERO-Long development memory at
cadence 5 with the frozen encoder, AFTER memory collection and BEFORE any evaluation episode; then freeze the
chosen values into the method params (fingerprint) and record this script's output sha256.

    .venv/bin/python methods/rtcf/scripts/select_pma_params.py PREPARED_DIR --out SELECT.json [--budget-minutes 30]

Data: the frozen memory (memory.npz + memory.json); each memory is one successful development episode (unique task,
init state), so leave-one-episode-out == leave-one-init-state-out. No robot episode, no evaluation case, no outcome
beyond development memory membership. Task labels are used for SCORING only, never in retrieval.
Procedure: for each held-out memory e, run PMA causally (the method's own pma_init/pma_step/pma_select) over e's
descriptor sequence against the bank without e (e's rows get +inf cost, which is equivalent to removal); record
i*_t, b*_t for every t.
Grid: v_max in {1, 2, 3} x gamma in {0, 0.01, 0.02, 0.05, 0.1, 0.2} (18 configurations).
Primary: mean over held-out episodes of the fraction of steps with task(i*_t) == task(e).
Secondary: mean Spearman(b*_t, t) over e's same-task steps (episodes with < 3 such steps or no variance skipped).
Choice: highest primary; every configuration within 1.0 pp of the maximum is tied; among tied prefer (2, 0.05) if
present, else highest secondary, then smaller gamma, then smaller v_max.
Also reported: cost spread (median and IQR of the local cost c over all (query step, stored position) pairs, sampled)
and the best-vs-median memory gap per step (per-memory min local cost: median minus min).
Budget: if the run exceeds --budget-minutes (default 30, measured as process CPU time) or fails, the a-priori
(2, 0.05) is kept. Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 (multithreaded BLAS spins and burns CPU budget).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from hs_rtcf.memory import MemoryBank
from hs_rtcf.pma import local_index, pma_init, pma_select, pma_step

V_MAX = (1, 2, 3)
GAMMA = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
DEFAULT = (2, 0.05)
TIE_PP = 1.0


def spearman(x: np.ndarray, y: np.ndarray):
    if len(x) < 3:
        return None
    # average ranks for ties
    def avg_rank(v):
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v), dtype=np.float64)
        sv = np.asarray(v)[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return ranks
    rx, ry = avg_rank(np.asarray(x, np.float64)), avg_rank(np.asarray(y, np.float64))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def choose(results: list) -> tuple:
    best = max(r["primary"] for r in results)
    tied = [r for r in results if r["primary"] >= best - TIE_PP / 100.0]
    for r in tied:
        if (r["v_max"], r["gamma"]) == DEFAULT:
            return DEFAULT, tied, "default (2, 0.05) within 1.0 pp of the best"
    tied_sorted = sorted(tied, key=lambda r: (-(r["secondary"] if r["secondary"] is not None else -9), r["gamma"], r["v_max"]))
    r = tied_sorted[0]
    return (r["v_max"], r["gamma"]), tied, "highest secondary among tied, then smaller gamma, then smaller v_max"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-minutes", type=float, default=30.0)
    ap.add_argument("--max-heldout", type=int, default=0, help="0 = all memories (the rule); >0 only for smoke tests")
    ap.add_argument("--allow-non-long", action="store_true", help="synthetic smoke tests only")
    a = ap.parse_args()
    t_start, cpu_start = time.monotonic(), time.process_time()
    prep = Path(a.prepared)
    pj = json.loads((prep / "prepared.json").read_text()) if (prep / "prepared.json").exists() else None
    arts = {x["name"]: Path(x["path"]) for x in pj["artifacts"]} if pj else {"memory": prep / "memory.npz",
                                                                             "memory_meta": prep / "memory.json"}
    bank = MemoryBank.load(arts["memory"], arts["memory_meta"])
    meta = bank.meta
    tasks = [m["task"] for m in meta["memories"]]
    not_long = sorted({t for t in tasks if not str(t).startswith("libero_10/")})
    if not_long and not a.allow_non_long:     # preflight R6: Decision 3 selects once, on the LIBERO-Long memory
        raise SystemExit(f"memory contains non-LIBERO-Long tasks {not_long[:3]}; the rule selects on Long only")
    n = bank.num_memories
    out: dict = {"rule": "critic review-plan Decision 3", "grid": {"v_max": V_MAX, "gamma": GAMMA},
                 "memory_sha256": hashlib.sha256(Path(arts["memory"]).read_bytes()).hexdigest(),
                 "num_memories": n, "num_positions": bank.num_positions, "encoder": meta.get("encoder"),
                 "default": list(DEFAULT)}
    status, chosen = "ok", DEFAULT
    try:
        if n < 2:
            raise ValueError("need at least 2 memories")
        offsets, loc = bank.offsets, local_index(bank.offsets)
        z = bank.descriptors.astype(np.float64)
        held = range(n) if not a.max_heldout else range(min(n, a.max_heldout))
        costs, spread_samples, gaps = {}, [], []
        rng = np.random.default_rng(0)
        for e in held:
            q = z[offsets[e]:offsets[e + 1]]
            c = 1.0 - np.clip(q @ z.T, -1.0, 1.0)                 # [T_e, sum L]
            c[:, offsets[e]:offsets[e + 1]] = np.inf             # leave e out
            costs[e] = c
            finite = np.isfinite(c)
            vals = c[finite]
            spread_samples.append(vals[rng.integers(0, vals.size, size=min(2000, vals.size))])
            per_mem = np.minimum.reduceat(np.where(finite, c, np.inf), offsets[:-1], axis=1)   # [T_e, N]
            per_mem = np.delete(per_mem, e, axis=1)
            gaps.extend((np.median(per_mem, axis=1) - per_mem.min(axis=1)).tolist())
        s = np.concatenate(spread_samples)
        out["cost_spread"] = {"median": float(np.median(s)), "q25": float(np.quantile(s, 0.25)),
                              "q75": float(np.quantile(s, 0.75)), "iqr": float(np.quantile(s, 0.75) - np.quantile(s, 0.25)),
                              "best_vs_median_gap_median": float(np.median(gaps)),
                              "best_vs_median_gap_q25_q75": [float(np.quantile(gaps, 0.25)), float(np.quantile(gaps, 0.75))]}
        results = []
        for v_max, gamma in itertools.product(V_MAX, GAMMA):
            if time.process_time() - cpu_start > 60 * a.budget_minutes:
                raise TimeoutError("budget exceeded")
            prim, sec = [], []
            for e, c in costs.items():
                J = None
                hits, bs, ts = [], [], []
                for t in range(c.shape[0]):
                    J = pma_init(c[t]) if J is None else pma_step(J, c[t], loc, v_max, gamma)
                    i, b, _ = pma_select(J, offsets, t + 1)
                    same = tasks[i] == tasks[e]
                    hits.append(same)
                    if same:
                        bs.append(b)
                        ts.append(t)
                prim.append(float(np.mean(hits)))
                rho = spearman(np.asarray(ts), np.asarray(bs))
                if rho is not None:
                    sec.append(rho)
            results.append({"v_max": v_max, "gamma": gamma, "primary": float(np.mean(prim)),
                            "secondary": float(np.mean(sec)) if sec else None, "episodes_scored_secondary": len(sec)})
        chosen, tied, why = choose(results)
        out.update({"results": results, "tied": [[r["v_max"], r["gamma"]] for r in tied], "why": why,
                    "heldout_episodes": len(costs)})
    except Exception as exc:  # noqa: BLE001
        status = f"fallback_default: {type(exc).__name__}: {exc}"
        chosen = DEFAULT
    out.update({"status": status, "chosen": {"v_max": chosen[0], "gamma": chosen[1]},
                "seconds": round(time.monotonic() - t_start, 1),
                "cpu_seconds": round(time.process_time() - cpu_start, 1),
                "blas_threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                "smoke_only": bool(a.max_heldout)})
    text = json.dumps(out, indent=1)
    Path(a.out).write_text(text)
    print(json.dumps({k: out[k] for k in ("status", "chosen", "seconds", "num_memories")}))
    print("sha256", hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
