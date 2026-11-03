"""Write the two paper cards of hs_reseat_reentry from ONE frozen constant set (python scripts/make_cards.py).
Every behavioural parameter is explicit in the card, so the Campaign freeze hashes it."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1] / "hs_reseat_reentry"
sys.path.insert(0, str(PKG.parent))
from hs_reseat_reentry import controller as C, motion as M, perception as P, planner as PL, triggers_gap as T  # noqa: E402

BASE_PARAMS = {
    "mode": "candidate",
    "execute_prefix": 1,
    "max_perception_calls": C.DEFAULTS["max_perception_calls"],
    "retain_packets": True,
    "articulated_words": list(C.DEFAULTS["articulated_words"]),
    "triggers": dict(T.DEFAULTS),
    "planner": {k: v for k, v in PL.DEFAULTS.items() if k != "scripted_responses"},
    "perception": dict(P.DEFAULTS),
    "motion": dict(M.DEFAULTS),
}

CARD = {
    "name": "reseat-reentry",
    "version": "1",
    "status": ("NEGATIVE (development): dropped at the predeclared pilot (0 rescues / 10 fired failures; Run X "
               "scaleagent-codex-20261103, decision-drop-x-r2-end-search.json); never frozen, no validation/final "
               "exposure. See docs/results/scaleagent-x.md."),
    "paper": {"id": "rmh:scaleagent-codex-20261103/x-r2-reseat-reentry",
              "title": "Reseat re-entry: SCALE decoding with a planner-witnessed open-gripper re-seat after a "
                       "post-release stall",
              "venue": "RoboMetaHarness Run X (original mechanism; sources arXiv:2602.04208 SCALE; Scientist C H1 "
                       "merged with Scientist D xd2)"},
    "categories": {"primary": "agentic_orchestration", "secondary": ["execution_monitoring"]},
    "fidelity": "component",
    "substitutions": [],
    "requires": {"capabilities": ["act", "chunk", "seeded_sampling", "isolated_streams", "decoding_hooks"]},
    # The gap trigger needs the robot proprio `states`; vla-eval LIBERO sends it only with send_state: true.
    "benchmark_params": {"send_state": True},
    # policy_calls: one SCALE sample per decision (<= 520) + margin; llm_calls: codex planner cap per episode;
    # tool_calls: depth back-projection cap; max_tokens 0: codex exec has no output cap (recorded per call).
    "budgets": {"policy_calls": 600, "llm_calls": PL.DEFAULTS["max_calls_episode"],
                "tool_calls": C.DEFAULTS["max_perception_calls"], "max_tokens": 0},
    "null": "shadow",
    "artifacts": [],
    "operators": [],
    "method_local_operators": "FILLED_BELOW",
    "original": {"repo": "none (new method)", "commit": "n/a",
                 "reported": "n/a; predeclared pilot and gates in runs/scaleagent-codex-20261103/reviews/proposal_r2_review.json"},
    "description": "SCALE-decoded frozen OpenVLA; after a post-release stall (proprio), an isolated codex planner points at the "
                   "remaining instruction object, a same-frame marked crop witnesses it, the open gripper is re-seated "
                   "0.10 m above it, and SCALE resumes with the original instruction.",
    "tier1": {"instruction": "put both the alphabet soup and the tomato sauce in the basket", "horizon": 24,
              "chunk_len": 1},
    "disclosure": ["depth + calibration + robot proprio (send_state) are harness-only added public inputs",
                   "planner = gpt-6.1-sol via codex exec (effort low), unseeded, may have seen the benchmark in pretraining",
                   "analytic share: per re-seat budget clip(steps_left - 105, 45, 90) primitive steps (review "
                   "r2_impl_review ruling 2), <= 2 re-seats per episode, gripper never closed by the harness",
                   "trigger: C's measured gap rule made causal (deferral while a width run is in progress); end kinds = "
                   "release or lost (C's measured script; review r2_impl_review ruling 1 freezes it)",
                   "the Campaign shadow null executes greedy native actions (controls overhead vs greedy only); the "
                   "SCALE-shadow null is reseat-reentry-respondoff",
                   "the -home ablation of the review is not built (optional, development-only)"],
}

VARIANTS = {
    "paper.yaml": ("reseat-reentry", {}, "candidate"),
    "paper_respondoff.yaml": ("reseat-reentry-respondoff", {"mode": "respondoff"},
                              "SCALE-shadow null: same trigger, planner + witness calls and back-projection, response "
                              "suppressed; executed actions are the SCALE proposals (must be bitwise plain scale)."),
}


def main() -> None:
    for fname, (name, override, role) in VARIANTS.items():
        card = copy.deepcopy(CARD)
        card["name"] = name
        params = copy.deepcopy(BASE_PARAMS)
        params.update(override)
        card["params"] = params
        card["role"] = role
        pl = params["planner"]
        card["method_local_operators"] = [{
            "name": "planner", "kind": "planner_cli", "backend": "codex", "model": pl["model"], "effort": pl["effort"],
            "timeout_s": pl["timeout_s"], "calls_per_episode": pl["max_calls_episode"],
            "calls_per_escalation": pl["max_calls_escalation"], "binary": pl["codex_binary"],
            "auth": "codex ChatGPT login (no API key)", "seeded": False,
            "implementation": "hs_reseat_reentry/vendor/codex_planner.py (sha256 pinned in vendor/PROVENANCE.json)",
            "calls_made_in_arms": "candidate, respondoff and the shadow null (on fires of their own trajectories); "
                                  "baseline: none; 0 calls in unfired episodes"}]
        header = ("# GENERATED by scripts/make_cards.py from one constant set; edit that script, not this file.\n"
                  "# Frozen scalars from review proposal_r2_review.json merged_spec.frozen_scalars (development data only).\n")
        (PKG / fname).write_text(header + yaml.safe_dump(card, sort_keys=False, width=118))
        print("wrote", fname)


if __name__ == "__main__":
    main()
