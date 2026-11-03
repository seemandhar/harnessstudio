"""Write the four paper cards of hs_witnessed_rescue from ONE frozen-able constant set (run from the package root:
python scripts/make_cards.py). Every parameter is explicit in the card, so the Campaign freeze hashes it."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1] / "hs_witnessed_rescue"
sys.path.insert(0, str(PKG.parent))
from hs_witnessed_rescue import motion as M, perception as P, planner as PL, triggers as T  # noqa: E402

BASE_PARAMS = {
    "mode": "candidate",
    "place_enabled": True,
    "execute_prefix": 1,
    "min_steps_left_place": 170,
    "max_grasp_attempts": 2,
    "max_perception_calls": 30,
    "witness_route": "marked_crop",
    "retain_packets": True,
    "custody_abstain_policy": "measured",
    "closed_destination_words": ["drawer", "drawers", "cabinet", "cupboard", "shelf", "compartment", "microwave",
                                 "oven", "fridge", "refrigerator", "caddy", "lid", "lidded"],
    "triggers": dict(T.DEFAULTS),
    "planner": {k: v for k, v in PL.DEFAULTS.items() if k != "scripted_responses"},
    "perception": dict(P.DEFAULTS),
    "motion": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in M.DEFAULTS.items()},
}

SUBSTITUTIONS = [
    "Harness VLA planner from step 0 (arXiv:2607.08448) -> event-triggered planner calls only after a public proprio "
    "grasp-failure trigger (H2 empty close / H1 settled unlifted clamp; SCALE++ dev replays ported as code)",
    "Claude Code / MCP tool loop -> stateless isolated `codex exec` JSON calls (gpt-6.1-sol, unseeded, read-only "
    "sandbox, zero tool items; vendored CodexPlanner sha256 a37c8f1a...), at most 12 per episode",
    "RPent pi0 primitives -> analytic grasp/lift/place on public proprio with a measured custody test",
    "SAM3 segmentation -> metric-depth back-projection + region growing + same-frame marked-crop witness by the "
    "planner (SAM3 route not built)",
    "native greedy OpenVLA decoding -> SCALE decoding (arXiv:2602.04208, hs_scale ScaleDecoder paper defaults) at "
    "every decision, seeded exactly as the plain `scale` method under the same case key",
    "added harness-only public sensors: agentview + wrist metric depth and camera calibration (libero_rgbd); the "
    "frozen policy and the SCALE/greedy arms never receive them",
]

CARD = {
    "name": "witnessed-rescue",
    "version": "1",
    "status": ("NEGATIVE (development): dropped at the predeclared re-pilot (0 rescues / 10 scenes; Run X "
               "scaleagent-codex-20261103, decision-drop-x-r1.json); never frozen, no validation/final exposure. "
               "See docs/results/scaleagent-x.md."),
    "paper": {"id": "rmh:scaleagent-codex-20261103/x-r1-witnessed-rescue",
              "title": "Witnessed rescue: SCALE decoding with event-triggered, depth-grounded, planner-witnessed "
                       "grasp recovery",
              "venue": "RoboMetaHarness Run X (new method; sources arXiv:2602.04208 SCALE, arXiv:2607.08448 "
                       "Harness VLA, SCALE++ gpp-core negatives)"},
    "categories": {"primary": "agentic_orchestration",
                   "secondary": ["execution_monitoring", "skill_composition", "observation_grounding",
                                 "uncertainty_compute"]},
    "fidelity": "tf_variant",
    "substitutions": SUBSTITUTIONS,
    "requires": {"capabilities": ["act", "chunk", "seeded_sampling", "isolated_streams", "decoding_hooks"]},
    # fix2: the triggers need the robot proprio `states`; vla-eval LIBERO sends it only with send_state: true. A
    # Campaign freeze without --send-state is refused (fail-closed; ext/campaign_rgbd/send_state_check.json).
    "benchmark_params": {"send_state": True},
    # policy_calls: one SCALE sample per decision (<= 520 on libero_10) + margin. llm_calls: the codex planner's
    # per-episode cap (params.planner.max_calls_episode); tool_calls: the depth perception cap
    # (params.max_perception_calls). Both are enforced and accounted by the controller (accounting.extra), because
    # the codex operator is method-local, not a framework OperatorPin (see method_local_operators). max_tokens 0:
    # codex exec has no output cap (recorded per call, not enforceable).
    "budgets": {"policy_calls": 600, "llm_calls": 12, "tool_calls": 30, "max_tokens": 0},
    "null": "shadow",
    "artifacts": [],
    # Framework OperatorPins describe OpenAI-compatible HTTP endpoints with an API key; the planner is an isolated
    # `codex exec` CLI call (ChatGPT login, no key), so it is declared here instead and runs inside the controller.
    "operators": [],
    "method_local_operators": "FILLED_BELOW",
    "original": {"repo": "none (new method)", "commit": "n/a",
                 "reported": "n/a; pre-registered decision rules in runs/scaleagent-codex-20261103/reviews/"
                             "proposal_r1_review.json"},
    "description": "SCALE-decoded frozen OpenVLA; on a public grasp-failure trigger, an isolated codex planner points "
                   "at the instruction's target, a same-frame marked crop witnesses it, a depth-grounded analytic "
                   "grasp runs with a measured custody test, then an open-top analytic place or hand-back.",
    "tier1": {"instruction": "put both the alphabet soup and the tomato sauce in the basket", "horizon": 24,
              "chunk_len": 1},
    "disclosure": ["depth + calibration are harness-only added public sensors",
                   "planner = gpt-6.1-sol via codex exec, unseeded, may have seen the benchmark in pretraining",
                   "report the analytic-control share; > 15% of steps on any block = 'harness with substantial "
                   "analytic control'",
                   "the Campaign shadow null executes greedy native actions (controls overhead vs greedy only); the "
                   "SCALE-shadow null is witnessed-rescue-respondoff"],
}

VARIANTS = {
    "paper.yaml": ("witnessed-rescue", {}, "candidate"),
    "paper_respondoff.yaml": ("witnessed-rescue-respondoff", {"mode": "respondoff"},
                              "SCALE-shadow null: same triggers, planner and perception calls, responses suppressed; "
                              "executed actions are the SCALE proposals (must be bitwise the plain scale method)."),
    "paper_noplace.yaml": ("witnessed-rescue-noplace", {"place_enabled": False},
                           "ablation (descriptive, B3-equivalent): grasp + custody, then always hand back."),
    "paper_analytic.yaml": ("witnessed-rescue-analytic", {"mode": "analytic"},
                            "primitives-only ablation (descriptive, never a Campaign entry): after the first custody "
                            "the planner + primitives complete the rest analytically, no hand-back."),
}


def main() -> None:
    for fname, (name, override, role) in VARIANTS.items():
        card = copy.deepcopy(CARD)
        card["name"] = name
        params = copy.deepcopy(BASE_PARAMS)
        params.update(override)
        card["params"] = params
        pl = params["planner"]
        card["method_local_operators"] = [{
            "name": "planner", "kind": "planner_cli", "backend": "codex", "model": pl["model"],
            "effort": pl["effort"], "timeout_s": pl["timeout_s"], "calls_per_episode": pl["max_calls_episode"],
            "calls_per_escalation": pl["max_calls_escalation"], "binary": pl["codex_binary"],
            "auth": "codex ChatGPT login (no API key)", "seeded": False,
            "implementation": "hs_witnessed_rescue/vendor/codex_planner.py (sha256 pinned in vendor/PROVENANCE.json)",
            "original_model": "Harness VLA paper: Claude Code / Codex agent loop (substituted, see substitutions)",
            "calls_made_in_arms": "candidate, respondoff, noplace, analytic and the shadow null (same triggers on its "
                                  "own trajectory); baseline: none"}]
        card["role"] = role
        header = ("# GENERATED by scripts/make_cards.py from one constant set; edit that script, not this file.\n"
                  "# All thresholds/budgets are explicit params, frozen on development data only (review A1/A2/A4/B4).\n")
        (PKG / fname).write_text(header + yaml.safe_dump(card, sort_keys=False, width=118))
        print("wrote", fname)


if __name__ == "__main__":
    main()
