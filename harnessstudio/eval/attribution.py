"""First-divergence attribution of paired episodes (spec 8.1).

For a pair (reference, other) with per-step observation and action hashes:
  M  every step matches;
  R  the first difference is an observation while all earlier actions matched: the environment
     (usually the renderer) branched; the pair is unattributable for mechanism claims;
  X  an action differs although its observation matched: the harness changed behavior where it
     must not (a leak in a null/baseline comparison). Meaningful only for deterministic or
     seeded policies (spec 16.4); stochastic T0 servers can produce X from policy noise.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def classify(reference: dict, other: dict) -> dict:
    ra, oa = reference.get("action_sha256") or [], other.get("action_sha256") or []
    ro, oo = reference.get("obs_sha256") or [], other.get("obs_sha256") or []
    for i in range(min(len(ra), len(oa))):
        if i < len(ro) and i < len(oo) and ro[i] != oo[i]:
            return {"class": "R", "step": i, "reason": "observation differs after identical actions"}
        if ra[i] != oa[i]:
            return {"class": "X", "step": i, "reason": "action differs for an identical observation"}
    if len(ra) != len(oa):
        return {"class": "R", "step": min(len(ra), len(oa)), "reason": "episode length differs after identical steps"}
    return {"class": "M", "step": None, "reason": ""}


def attribute(reference: Iterable[dict], other: Iterable[dict]) -> dict:
    key = lambda r: (r["benchmark"], r["task"], r["episode_idx"])  # noqa: E731
    ref = {key(r): r for r in reference if r["valid"]}
    oth = {key(r): r for r in other if r["valid"]}
    per_case = {"|".join(map(str, k)): classify(ref[k], oth[k]) for k in sorted(set(ref) & set(oth))}
    counts = Counter(v["class"] for v in per_case.values())
    return {"M": counts.get("M", 0), "R": counts.get("R", 0), "X": counts.get("X", 0), "per_case": per_case}
