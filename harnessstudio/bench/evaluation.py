"""Strict, paired evaluation accounting for frozen-policy harness experiments.

An episode error is missing evidence, never a policy failure. Rates from incomplete
runs are descriptive only. A historical score is a numeric target, not a paired
control and not proof of a state-of-the-art result.
"""

from collections.abc import Mapping
import math
import re


CONTRACT_FIELDS = (
    "seed", "horizon", "initial_observation_sha256", "protocol_sha256",
    "checkpoint_sha256", "action_chunk", "observation_stride", "control_hz",
)
DIGEST_FIELDS = (
    "initial_observation_sha256", "protocol_sha256", "checkpoint_sha256",
)
INTEGER_FIELDS = ("trial", "seed", "steps", "horizon", "action_chunk", "observation_stride")
NUMBER_FIELDS = ("policy_seconds", "wall_seconds", "control_hz")
REQUIRED_FIELDS = frozenset(("task", "harness", "success", *INTEGER_FIELDS,
                             *NUMBER_FIELDS, *DIGEST_FIELDS))


def _number(value, name, *, positive=True):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_episode(record):
    """Return a shallow copy of one validated episode; never coerce values."""
    if not isinstance(record, Mapping):
        raise ValueError("episode must be an object")
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"episode missing fields: {', '.join(sorted(missing))}")
    for field in ("error", "errors"):
        if record.get(field):
            raise ValueError("episode contains an execution error; it is not a policy failure")
    for field in ("task", "harness"):
        if type(record[field]) is not str or not record[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if type(record["success"]) is not bool:
        raise ValueError("success must be a boolean")
    for field in INTEGER_FIELDS:
        if type(record[field]) is not int:
            raise ValueError(f"{field} must be an integer")
        minimum = 1 if field in ("horizon", "action_chunk", "observation_stride") else 0
        if record[field] < minimum:
            raise ValueError(f"{field} must be >= {minimum}")
    if record["steps"] > record["horizon"]:
        raise ValueError("steps exceeds horizon")
    if record["success"] and record["steps"] == 0:
        raise ValueError("a successful episode must execute at least one step")
    if not record["success"] and record["steps"] != record["horizon"]:
        raise ValueError("a failed episode must exhaust its horizon; aborted episodes are not policy failures")
    for field in NUMBER_FIELDS:
        _number(record[field], field)
    tolerance = max(1e-6, record["wall_seconds"] * 1e-9)
    if record["policy_seconds"] - record["wall_seconds"] > tolerance:
        raise ValueError("policy_seconds exceeds episode wall_seconds")
    for field in DIGEST_FIELDS:
        if type(record[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", record[field]) is None:
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return dict(record)


def validate_records(records):
    """Validate one harness arm, rejecting duplicate task/trial outcomes."""
    if not isinstance(records, (list, tuple)):
        raise ValueError("records must be a list of episodes")
    validated, seen, harnesses = [], set(), set()
    for record in records:
        item = validate_episode(record)
        case = (item["task"], item["trial"])
        if case in seen:
            raise ValueError(f"duplicate episode: {case[0]}/{case[1]}")
        seen.add(case)
        harnesses.add(item["harness"])
        validated.append(item)
    if len(harnesses) > 1:
        raise ValueError("one arm contains multiple harness identities")
    return validated


def paired_binomial_p(wins, losses):
    """Exact one-sided sign/McNemar-binomial tail, excluding concordant pairs."""
    if type(wins) is not int or type(losses) is not int or min(wins, losses) < 0:
        raise ValueError("wins and losses must be nonnegative integers")
    n = wins + losses
    if n == 0 or wins == 0:
        return 1.0
    # A single integer division avoids floating-point overflow for large n.
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (1 << n)


def wilson_interval(successes, total, z=1.959963984540054):
    """Wilson 95% interval by default; return null endpoints for zero episodes."""
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("counts must be integers with 0 <= successes <= total")
    _number(z, "z")
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _cases(expected_cases):
    if not isinstance(expected_cases, (list, tuple)) or not expected_cases:
        raise ValueError("a nonempty expected task/trial list is mandatory")
    result = set()
    for case in expected_cases:
        if isinstance(case, Mapping):
            task, trial = case.get("task"), case.get("trial")
            for field in ("seed", "horizon"):
                if field in case and (type(case[field]) is not int or case[field] < (1 if field == "horizon" else 0)):
                    raise ValueError(f"expected case {field} must be a valid integer")
        elif isinstance(case, (tuple, list)) and len(case) == 2:
            task, trial = case
        else:
            raise ValueError("each expected case must contain task and trial")
        if type(task) is not str or not task.strip() or type(trial) is not int or trial < 0:
            raise ValueError("expected cases require a nonempty task and nonnegative integer trial")
        if (task, trial) in result:
            raise ValueError(f"duplicate expected case: {task}/{trial}")
        result.add((task, trial))
    return result


def _case_list(cases):
    return [{"task": task, "trial": trial} for task, trial in sorted(cases)]


def assess(candidate_records, baseline_records, expected_cases, *, min_net_wins,
           max_inference_ratio, alpha=None, expected_contract=None):
    """Apply a predeclared paired gate. Incomplete or unequal contracts never pass.

    ``expected_cases`` accepts task/trial pairs or objects. ``expected_contract``
    maps episode fields to frozen values (e.g. checkpoint_sha256/action_chunk).
    Use a common evaluation protocol ID across arms, and separate harness names.
    Schema/coverage defects are reported as reasons rather than counted as losses.
    Invalid gate configuration raises ValueError.
    """
    expected = _cases(expected_cases)
    case_contracts = {
        (case["task"], case["trial"]): {field: case[field] for field in ("seed", "horizon") if field in case}
        for case in expected_cases if isinstance(case, Mapping)
    }
    if type(min_net_wins) is not int:
        raise ValueError("min_net_wins must be an integer")
    _number(max_inference_ratio, "max_inference_ratio")
    if alpha is not None:
        _number(alpha, "alpha")
        if alpha > 1:
            raise ValueError("alpha must be <= 1")
    if expected_contract is not None:
        if not isinstance(expected_contract, Mapping):
            raise ValueError("expected_contract must be an object")
        if set(expected_contract) - REQUIRED_FIELDS:
            raise ValueError("expected_contract contains unsupported fields")
    summary = {
        "complete": False, "comparable": False, "passed": False, "reasons": [],
        "expected_episodes": len(expected), "candidate_episodes": 0, "baseline_episodes": 0,
        "matched_episodes": 0, "candidate_successes": 0, "baseline_successes": 0,
        "candidate_rate": None, "baseline_rate": None,
        "candidate_wilson_interval": [None, None], "baseline_wilson_interval": [None, None],
        "matched_wins": 0, "matched_losses": 0, "net_wins": 0,
        "gains": [], "losses": [], "paired_p": None,
        "inference_ratio": None, "cost_ratio": None, "wall_time_ratio": None,
        "missing_candidate": [], "missing_baseline": [], "unexpected_candidate": [],
        "unexpected_baseline": [], "contract_mismatches": [],
        "thresholds": {"min_net_wins": min_net_wins,
                       "max_inference_ratio": max_inference_ratio, "alpha": alpha},
    }
    try:
        candidate = validate_records(candidate_records)
        baseline = validate_records(baseline_records)
    except ValueError as exc:
        summary["reasons"].append(str(exc))
        return summary
    c = {(r["task"], r["trial"]): r for r in candidate}
    b = {(r["task"], r["trial"]): r for r in baseline}
    summary.update(candidate_episodes=len(c), baseline_episodes=len(b))
    for arm, records in (("candidate", c), ("baseline", b)):
        summary[f"missing_{arm}"] = _case_list(expected - records.keys())
        summary[f"unexpected_{arm}"] = _case_list(records.keys() - expected)
        if summary[f"missing_{arm}"]:
            summary["reasons"].append(f"{arm} has missing expected episodes")
        if summary[f"unexpected_{arm}"]:
            summary["reasons"].append(f"{arm} has unexpected episodes")
    summary["complete"] = c.keys() == b.keys() == expected
    matched = sorted(expected & c.keys() & b.keys())
    summary["matched_episodes"] = len(matched)
    if c and b and candidate[0]["harness"] == baseline[0]["harness"]:
        summary["reasons"].append("candidate and baseline must have distinct harness identities")
    for case in matched:
        fields = [field for field in CONTRACT_FIELDS if c[case][field] != b[case][field]]
        if fields:
            summary["contract_mismatches"].append({"task": case[0], "trial": case[1], "fields": fields})
    if expected_contract:
        for arm, records in (("candidate", c), ("baseline", b)):
            for case, record in records.items():
                fields = [field for field, value in expected_contract.items()
                          if type(record[field]) is not type(value) or record[field] != value]
                if fields:
                    summary["contract_mismatches"].append({
                        "task": case[0], "trial": case[1], "arm": arm, "fields": fields,
                    })
    for arm, records in (("candidate", c), ("baseline", b)):
        for case in expected & records.keys():
            fields = [field for field, value in case_contracts.get(case, {}).items() if records[case][field] != value]
            if fields:
                summary["contract_mismatches"].append({
                    "task": case[0], "trial": case[1], "arm": arm,
                    "fields": fields, "source": "expected_case",
                })
    distinct_arms = not (c and b) or candidate[0]["harness"] != baseline[0]["harness"]
    summary["comparable"] = bool(matched) and not summary["contract_mismatches"] and distinct_arms
    if summary["contract_mismatches"]:
        summary["reasons"].append("evaluation contracts or initial observations differ")
    if not matched:
        summary["reasons"].append("no matched expected episodes")
        return summary
    # Rates/costs always use the same matched cases, even during partial reporting.
    cs = sum(c[k]["success"] for k in matched)
    bs = sum(b[k]["success"] for k in matched)
    gains = [k for k in matched if c[k]["success"] and not b[k]["success"]]
    losses = [k for k in matched if b[k]["success"] and not c[k]["success"]]
    ratio = sum(c[k]["policy_seconds"] for k in matched) / sum(b[k]["policy_seconds"] for k in matched)
    wall_ratio = sum(c[k]["wall_seconds"] for k in matched) / sum(b[k]["wall_seconds"] for k in matched)
    if not math.isfinite(ratio) or not math.isfinite(wall_ratio):
        summary["comparable"] = False
        summary["reasons"].append("aggregate runtime ratio is not finite")
        return summary
    summary.update(
        candidate_successes=cs, baseline_successes=bs,
        candidate_rate=cs / len(matched), baseline_rate=bs / len(matched),
        candidate_wilson_interval=wilson_interval(cs, len(matched)),
        baseline_wilson_interval=wilson_interval(bs, len(matched)),
        matched_wins=len(gains), matched_losses=len(losses), net_wins=len(gains) - len(losses),
        gains=_case_list(gains), losses=_case_list(losses),
        inference_ratio=ratio, cost_ratio=ratio,
        wall_time_ratio=wall_ratio,
    )
    if summary["comparable"]:
        summary["paired_p"] = paired_binomial_p(len(gains), len(losses))
    if summary["net_wins"] < min_net_wins:
        summary["reasons"].append("net wins below predeclared minimum")
    if ratio > max_inference_ratio:
        summary["reasons"].append("inference cost exceeds predeclared limit")
    if alpha is not None and (summary["paired_p"] is None or summary["paired_p"] > alpha):
        summary["reasons"].append("paired significance threshold not met")
    summary["passed"] = summary["complete"] and summary["comparable"] and not summary["reasons"]
    return summary


def compare_historical_target(successes, total, target_successes, target_total):
    """Describe a numeric score comparison; this cannot establish a SOTA claim."""
    for count, denominator in ((successes, total), (target_successes, target_total)):
        if type(count) is not int or type(denominator) is not int or not 0 <= count <= denominator or denominator <= 0:
            raise ValueError("target comparisons require valid nonzero integer counts")
    return {
        "candidate_successes": successes, "candidate_episodes": total,
        "target_successes": target_successes, "target_episodes": target_total,
        "candidate_rate": successes / total, "target_rate": target_successes / target_total,
        "numerically_exceeds_target": successes * target_total > target_successes * total,
        "paired": False, "sota_claim_supported": False,
        "limitation": "A historical rate alone does not verify comparable conditions or statistical improvement.",
    }
