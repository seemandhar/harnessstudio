"""Arms and the derived-null rules (spec 8.1, ported from v1).

baseline: the native policy (sample 0 stream, full chunk, original instruction).
null: pays the candidate's compute but executes natively.
candidate: the method.
"""
from __future__ import annotations

from enum import Enum


class Arm(str, Enum):
    BASELINE = "baseline"
    NULL = "null"
    CANDIDATE = "candidate"


def num_samples(arm: Arm, k: int) -> int:
    return 1 if arm is Arm.BASELINE else k


def executed_index(arm: Arm, chosen: int) -> int:
    return chosen if arm is Arm.CANDIDATE else 0


def honor_cut(arm: Arm, wants_continue: bool) -> bool:
    """Whether execution continues after a step. Only the candidate honours a cut."""
    return wants_continue if arm is Arm.CANDIDATE else True


def instruction_used(arm: Arm, original: str, rewritten: str) -> str:
    return rewritten if arm is Arm.CANDIDATE else original


def sampler_effective(arm: Arm) -> bool:
    return arm is Arm.CANDIDATE
