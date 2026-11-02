"""Write the three vlm-rehome cards from one source (run once before freeze; the cards are then frozen files).

    harnessstudio/.venv/bin/python harnessstudio/methods/vlm_rehome/scripts/make_cards.py
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1] / "hs_vlm_rehome"
sys.path.insert(0, str(PKG.parent))
from hs_vlm_rehome import judge  # noqa: E402

NS = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"
JUDGE_SHA = hashlib.sha256((PKG / "judge.py").read_bytes()).hexdigest()

PARAMS = {
    "gate": "vlm",
    # r4-perception frozen_scalars
    "boundary_min_close": 4, "hold_width_m": 0.004, "empty_width_m": 0.06, "max_boundaries": 2,
    "check_offsets_steps": [10, 30, 50, 80], "close_stop_decisions": 2, "max_calls": 8, "max_rehomes": 2,
    "rehome_tol_m": 0.02, "rehome_tol_deg": 10.0, "rehome_max_steps": 60, "vlm_timeout_s": judge.TIMEOUT_S,
    # re-home law: OSC_POSE output_max scaling (C-V7; osc_pose.json in the LIBERO image: 0.05 m, 0.5 rad), unit gains
    "kp": 1.0, "kr": 1.0, "pos_scale": 0.05, "rot_scale": 0.5, "lift_first": True, "lift_tol_m": 0.02,
    "gripper_open": -1.0,
    # tilt gate (C-V5; inherited from B, not tuned)
    "tilt_gate_deg": 55.0,
    "exec_rows": 5, "zero_chunk_tol": 1e-4, "image_key": judge.IMAGE_KEY, "state_key": "states", "operator": "judge",
    "models_check": True, "norm_stats": "assets/pi0_fast_libero_norm_stats.json", "norm_stats_sha256": NS,
    "judge_sha256": JUDGE_SHA,
}

DISCLOSURES = [
    "Original RoboMetaHarness mechanism (run rtcfpp-20261101, round-4 PERCEPTION proposal P1 'VPR' by agent ad8636fc416aedb01; "
    "review r4-proposal-review.json by critic a118b6025eedbd39e: approve qualification only, screen conditional on Q1 and the forced probe; "
    "amendment A4 in campaign-inputs/preregistration-r1.md). Not a paper port; fidelity 'faithful' to its own specification with an internal "
    "paper id (the enum has no 'new method' value). It is a new mechanism, not an NRP-adapted reproduction.",
    "Runtime operator: NRP qwen3-vl (served Qwen/Qwen3.8-Flash-Next-FP8), thinking off, temperature 0, max_tokens 64, no retries, no fallback, $0. "
    "PIN DEVIATION (disclosed, critic acknowledgement requested by the supervisor): the operator extra_body adds response_format json_schema "
    "{state: enum[abnormal, normal, unsure]} to the registry pin, added before freeze after a 1-call transport smoke on a non-Q1 S2 frame returned "
    "64 tokens of prose for the plain prompt; the prompt text is the proposal's, unchanged.",
    "FRAMEWORK GAP 1 (accepted by the supervisor): ctx.tools['llm'] does not expose the served model id of a reply, so deployed episodes record "
    "no per-reply model id; the served id is recorded on every Q1 reply, and the controller checks at every episode start that the pinned model "
    "is listed by /models (otherwise ctx.stop: infrastructure, the episode finishes natively).",
    "FRAMEWORK GAP 2 (accepted by the supervisor): the framework operator's HTTP timeout is fixed at 120 s; a reply after 60 s is classified as a "
    "timeout abstention (same decision as Q1's 60 s client timeout; only wall time differs).",
    "Delivered-image identity (C-V2): the controller and Q1 send judge.image_png(obs['images']['agentview']) (Pillow PNG compress_level=1, the "
    "encoder of the RTCF S2 stored frames; 454/454 Q1 frames re-encode byte-identically) and judge.build_messages; the request body equals the "
    "framework operator's (unit test).",
    "Native decisions use ctx.policy.sample(1) + ctx.execute(rows[:5]) (the bitwise-native seeded call of the baseline arm, as in redecode), "
    "not ctx.decide(), because the boundary detector needs the chunk's gripper command and zero-chunk status in every arm including the shadow "
    "null. Re-home steps consume the official 520-step horizon and make no policy call; afterwards the framework decision counter (CRN seed index) "
    "differs from the baseline's, which is irrelevant for greedy pi0-FAST decoding (evaluator: seed invariance).",
    "Re-home law: a_pos = clip(kp (target - p) / 0.05), a_rot = clip(kr log(R0 R^T) / 0.5), gripper open, kp = kr = 1 (OSC_POSE output_max "
    "scaling preferred by C-V7, i.e. each step commands the remaining error as the OSC goal offset, clipped), lift first to max(z, z0) holding "
    "x,y, stop at 0.02 m / 10 deg or 60 steps. A realized-gain regression on dev80 identity traces was attempted and rejected as confounded "
    "(cosine command vs realized motion 0.22 position / 0.05 rotation; implementation/vlm-rehome/calibration/gains.json); no outcome labels used.",
    "Shadow null: the controller runs on the native trajectory with the same boundary detection, the same VLM calls and would-be re-homes; "
    "executed actions are native. Public inputs only; no RNG; replies are not bitwise repeatable (FP8 serving, batching): outcomes are one "
    "realization of a stochastic gate, never rerun (C-V9).",
]

CARD = {
    "name": "vlm-rehome",
    "version": "1.0",
    "paper": {"id": "rmh:rtcfpp-20261101/r4p-p1", "title": "VLM-flagged post-boundary home re-entry (VPR) - original RoboMetaHarness mechanism "
              "(rtcfpp-20261101 round-4 PERCEPTION proposal P1), not a paper port", "venue": "none (internal)"},
    "categories": {"primary": "execution_monitoring", "secondary": ["world_model_verification"]},
    "fidelity": "faithful",
    "substitutions": [],
    "requires": {"capabilities": ["act", "chunk", "seeded_sampling"]},
    "budgets": {"policy_calls": 110, "llm_calls": 8, "tool_calls": 0, "max_tokens": judge.MAX_TOKENS},
    "null": "shadow",
    "artifacts": [{"name": "pi0_fast_libero_norm_stats",
                   "path": "/bigdata/sejain/hs2-cache/openpi/openpi-assets/checkpoints/pi0_fast_libero/assets/physical-intelligence/libero/norm_stats.json",
                   "sha256": NS, "license": "Apache-2.0 (openpi assets); byte-identical copy in hs_vlm_rehome/assets"}],
    "operators": [dict(judge.PIN)],
    "params": PARAMS,
    "disclosures": DISCLOSURES,
    "original": {"repo": "none (original method)", "commit": "",
                 "reported": "none; predictions in runs/rtcfpp-20261101/ideation/r4-perception.json (hard48 net +2, 80% interval -1..+5)"},
    "description": "After a public subtask boundary (gripper release after a hold), ask a frozen NRP VLM at +10/30/50/80 steps whether the arm "
                   "is abnormal; on 'abnormal' with the gripper empty, drive the end effector back to the episode's step-0 public pose, then "
                   "return to the native policy.",
    "tier1": {"instruction": "pick up the block and place it in the bowl", "horizon": 40, "chunk_len": 10},
    "changelog": ["v1.0 (2026-11-01): first implementation (task implement-vlm-rehome, coder ac34ce3a7d6d80e7d) under C-V1..C-V10 and A4; "
                  "judge.py frozen for Q1 before the controller (implementation/vlm-rehome/judge-manifest.json)."],
}


def variant(name, gate, title, extra_disclosure, description, llm=True):
    c = copy.deepcopy(CARD)
    c["name"] = name
    c["params"]["gate"] = gate
    c["paper"]["id"] = f"rmh:rtcfpp-20261101/r4p-p1-{gate}"
    c["paper"]["title"] = title
    c["disclosures"] = [extra_disclosure] + c["disclosures"]
    c["description"] = description
    if not llm:
        c["operators"] = []
        c["budgets"]["llm_calls"] = 0
        c["budgets"]["max_tokens"] = 0
        c["params"]["models_check"] = False
    return c


def main():
    cards = {
        "paper.yaml": CARD,
        "paper_tiltgate.yaml": variant(
            "vlm-rehome-tiltgate", "tilt",
            "Admission control for vlm-rehome (C-V5): same boundary detector, schedule and VLM calls (paid, recorded, ignored); re-home iff "
            "decision-time tilt > 55 deg with the gripper empty",
            "CONTROL for A4.3 (net(V) - net(tiltgate) >= +1): identical to vlm-rehome except the re-home decision, which ignores the verdict and "
            "fires iff tilt of the decision-time state > 55 deg (inherited from family B, not tuned) with the gripper empty. The VLM call is made "
            "and paid exactly as in vlm-rehome so cost and latency match.",
            "Control for vlm-rehome: same detector, schedule and VLM calls; re-home iff decision-time tilt > 55 deg with the gripper empty."),
        "paper_forced.yaml": variant(
            "vlm-rehome-forced", "forced",
            "DEVELOPMENT-ONLY forced re-home probe for vlm-rehome (C-V6): re-home at the first eligible check of the first boundary, no VLM call",
            "DIAGNOSTIC, not a candidate or control: for the 4-scene forced probe (A4.2 / C-V6) only; its episodes are disclosed exposure of those "
            "hard48 scenes, never screen outcomes. 0 VLM calls (no operator).",
            "Dev-only diagnostic: re-home to the step-0 pose at the first eligible post-boundary check, then native.", llm=False),
    }
    for fn, c in cards.items():
        (PKG / fn).write_text("# GENERATED by scripts/make_cards.py (frozen with the package; edit the generator, never the card)\n"
                              + yaml.safe_dump(c, sort_keys=False, width=140, allow_unicode=True))
        print(fn, c["name"])


if __name__ == "__main__":
    main()
