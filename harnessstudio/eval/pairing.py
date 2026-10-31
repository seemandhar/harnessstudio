"""Pair episodes across arms and summarize (spec 8.1). Only cases valid in every arm are paired."""
from __future__ import annotations

from math import comb
from typing import Iterable


def _key(r: dict) -> tuple:
    return (r["benchmark"], r["task"], r["episode_idx"])


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial test on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(arms: dict[str, Iterable[dict]]) -> dict:
    by_arm = {arm: {_key(r): r for r in recs} for arm, recs in arms.items()}
    cases = set.intersection(*(set(k for k, r in recs.items() if r["valid"]) for recs in by_arm.values())) \
        if by_arm else set()
    invalid = {arm: sorted(str(k) for k, r in recs.items() if not r["valid"]) for arm, recs in by_arm.items()}
    out: dict = {"paired_cases": len(cases), "invalid": invalid, "arms": {}}
    for arm, recs in by_arm.items():
        rs = [recs[k] for k in cases]
        out["arms"][arm] = {
            "successes": sum(bool(r["success"]) for r in rs), "episodes": len(rs),
            "policy_samples": sum(r["accounting"]["method_policy_samples"] + r["accounting"]["native_policy_samples"]
                                  + r["accounting"]["shadow_policy_samples"] for r in rs),
            "llm_calls": sum(r["accounting"]["llm_calls"] for r in rs),
            "wall_seconds": round(sum(r["wall_seconds"] for r in rs), 2),
        }
    base = by_arm.get("baseline")
    if base is not None:
        for arm in by_arm:
            if arm == "baseline":
                continue
            wins = sum(1 for k in cases if by_arm[arm][k]["success"] and not base[k]["success"])
            losses = sum(1 for k in cases if base[k]["success"] and not by_arm[arm][k]["success"])
            out["arms"][arm]["vs_baseline"] = {"wins": wins, "losses": losses, "p": round(sign_test_p(wins, losses), 4)}
        if "null" in by_arm:
            from .attribution import attribute
            out["null_attribution"] = attribute([base[k] for k in cases], [by_arm["null"][k] for k in cases])
    return out
