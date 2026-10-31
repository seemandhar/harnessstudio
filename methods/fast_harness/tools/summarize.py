"""Per-episode FastHarness metrics from HarnessStudio records (records.jsonl / dev_records.jsonl).

usage: python summarize.py LABEL=PATH [LABEL=PATH ...] [--json OUT]
PATH is a records.jsonl / dev_records.jsonl file or an hs eval arm directory containing records.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> list:
    p = Path(path)
    if p.is_dir():
        p = p / "records.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def episode(rec: dict) -> dict:
    ev = rec.get("events") or []
    llm = [e for e in ev if e.get("t") == "llm"]
    usage = [e.get("usage") or {} for e in llm]
    prompt = sum(int(u.get("prompt_tokens", 0)) for u in usage)
    completion = sum(int(u.get("completion_tokens", 0)) for u in usage)
    reasoning = sum(int((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)) for u in usage)
    cached = sum(int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) for u in usage)
    named = lambda n: [e for e in ev if e.get("t") == "event" and e.get("name") == n]  # noqa: E731
    reviews, execs, sched = named("fh_review"), named("fh_execute"), named("fh_schedule")
    start = (named("fh_start") or [{}])[0]
    chunks = len(execs)
    auto = sum(e.get("source") == "harness_auto" for e in execs)
    reasons: dict = {}
    for s in sched:
        reasons[str(s.get("reason"))] = reasons.get(str(s.get("reason")), 0) + 1
    m = rec.get("metrics") or {}
    acc = rec.get("accounting") or {}
    return {
        "case": f"{rec.get('benchmark')}/{rec.get('task')}/{rec.get('episode_idx')}", "arm": rec.get("arm"),
        "valid": rec.get("valid"), "invalid_reason": rec.get("invalid_reason"), "success": rec.get("success"),
        "score": m.get("score"), "steps": acc.get("steps"), "termination": acc.get("termination"),
        "memory": start.get("memory_source"), "cold": start.get("cold"),
        "llm_calls": len(llm), "accepted_reviews": len(reviews), "rejected_reviews": len(named("fh_review_rejected")),
        "failed_reviews": len(named("fh_review_failed")), "transport_errors": len(named("fh_transport_error")),
        "prompt_tokens": prompt, "completion_tokens": completion, "reasoning_tokens": reasoning,
        "cached_tokens": cached, "total_tokens": prompt + completion,
        "llm_seconds": round(sum(float(e.get("seconds", 0)) for e in llm), 1),
        "chunks": chunks, "auto_chunks": auto, "skip_rate": round(auto / chunks, 3) if chunks else None,
        "takeovers": sum(e.get("mode") in ("edit", "eef") for e in execs),
        "recovery_starts": sum(1 for e in named("fh_recovery_transition")),
        "errors_reported": sum(r.get("last_outcome") == "error" for r in reviews),
        "checkpoints": len(named("fh_checkpoint")),
        "passthrough": bool(named("fh_recovery_exhausted_passthrough")),
        "passthrough_chunks": sum(e.get("source") == "recovery_exhausted_passthrough" for e in execs),
        "distinct_stages": len({r.get("stage") for r in reviews}),
        "unknown_context": reasons.get("unknown_context", 0),
        "schedule_reasons": reasons,
        "policy_samples": acc.get("method_policy_samples", 0) + acc.get("native_policy_samples", 0)
                          + acc.get("shadow_policy_samples", 0),
        "wall_min": round(float(rec.get("wall_seconds", 0)) / 60, 2),
    }


def main(argv: list) -> int:
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    rows = []
    for item in argv:
        label, _, path = item.partition("=")
        for rec in load(path):
            rows.append({"label": label, **episode(rec)})
    cols = ("label", "case", "valid", "success", "score", "steps", "accepted_reviews", "llm_calls", "prompt_tokens",
            "completion_tokens", "reasoning_tokens", "chunks", "auto_chunks", "skip_rate", "passthrough_chunks", "takeovers",
            "recovery_starts", "unknown_context", "distinct_stages", "wall_min")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
