"""First-divergence attribution of paired episodes (spec 8.1).

For a pair (reference, other) with per-step observation and action hashes:
  M  every executed action matches (observation-only differences that never changed an action are
     renderer noise and are noted, not counted against the pair);
  R  actions diverge, and an observation diverged at or before that step: the environment (usually the
     renderer) branched; the pair is unattributable for mechanism claims;
  X  actions diverge although every observation up to that step matched: the harness changed behavior
     where it must not (a leak in a null/baseline comparison). Meaningful only for deterministic or
     seeded policies (spec 16.4); stochastic T0 servers can produce X from policy noise.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def classify(reference: dict, other: dict) -> dict:
    ra, oa = reference.get("action_sha256") or [], other.get("action_sha256") or []
    ro, oo = reference.get("obs_sha256") or [], other.get("obs_sha256") or []
    n = min(len(ra), len(oa))
    first_action = next((i for i in range(n) if ra[i] != oa[i]), None)
    obs_diff = [i for i in range(min(len(ro), len(oo), n)) if ro[i] != oo[i]]
    if first_action is None:
        if len(ra) == len(oa):
            # Identical behaviour. Observation differences that never changed an action are renderer noise.
            return {"class": "M", "step": None,
                    "reason": f"renderer noise on {len(obs_diff)} steps, actions unaffected" if obs_diff else "",
                    "obs_noise_steps": obs_diff[:20]}
        return {"class": "R", "step": n, "reason": "episode length differs after identical actions"}
    if obs_diff and obs_diff[0] <= first_action:
        return {"class": "R", "step": first_action,
                "reason": f"observations diverged at step {obs_diff[0]} before actions diverged"}
    return {"class": "X", "step": first_action, "reason": "action differs for an identical observation history"}


def attribute(reference: Iterable[dict], other: Iterable[dict]) -> dict:
    key = lambda r: (r["benchmark"], r["task"], r["episode_idx"])  # noqa: E731
    ref = {key(r): r for r in reference if r["valid"]}
    oth = {key(r): r for r in other if r["valid"]}
    per_case = {"|".join(map(str, k)): classify(ref[k], oth[k]) for k in sorted(set(ref) & set(oth))}
    counts = Counter(v["class"] for v in per_case.values())
    return {"M": counts.get("M", 0), "R": counts.get("R", 0), "X": counts.get("X", 0), "per_case": per_case}
