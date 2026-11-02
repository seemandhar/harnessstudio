"""Generate the tube-steer cards (hs_tube_steer/paper.yaml, paper_ortho.yaml). Run once before the freeze."""
from __future__ import annotations

from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1] / "hs_tube_steer"
RUN = "runs/rtcfpp-20261101"
NORM_SHA = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"
NPZ_SHA = "fcf8338306294cace8979ede718d2ede7d8b287198144fbf07e21604a5677f0a"
JSON_SHA = "00d2e87bc024ffed09fc04953e389f1bbded1f26c3ba3a4fe2be25d39fb2d180"
PARAMS = {"eps_deg": 15.0, "consecutive": 2, "arm_by_decision": 39, "window_decisions": 4, "release_deg": 7.5,
          "hold_limit_step": 300, "gain_floor": 0.5, "gain_cap": 1.0, "ortho_cap": 6, "exec_rows": 5, "decision_steps": 5,
          "zero_chunk_tol": 0.0001, "state_key": "states", "instruction_key": "task_description",
          "memory_npz": "assets/tube_memory.npz", "memory_npz_sha256": NPZ_SHA,
          "memory_json": "assets/tube_memory.json", "memory_json_sha256": JSON_SHA,
          "norm_stats": "assets/pi0_fast_libero_norm_stats.json", "norm_stats_sha256": NORM_SHA}

DISCLOSE = [
    "New training-free mechanism; not a reproduction of any cited paper. Original RoboMetaHarness method (run rtcfpp-20261101, round-5 MEMORY proposal P1-tube-steer, " + RUN + "/ideation/r5-memory.json; review " + RUN + "/reviews/r5-proposal-review.json C1-C13; amendment A5). Reported as a task-specific success-memory envelope: a tf_variant inspired by RTCF (arXiv 2608.04527, withdrawn v3; audited in runs/rtcf-20261031/audit/paper-audit.md) with the substitution 'DCT low-frequency correction toward retrieved success chunks -> gated restoring rotation toward the nearest same-instruction success-tube orientation'. Never a reproduction or verified SOTA; t8 untouched (no success memory); development evidence stays development evidence (C13).",
    "Memory (C1, C11): assets/tube_memory.npz (sha256 fcf83383...) and tube_memory.json (00d2e87b...), byte-identical copies of the frozen artifact built by " + RUN + "/ideation/scripts-r5-memory/build_tube_memory.py from the 127 SUCCESSFUL S2 memory episodes (runs/rtcf-20261031/eval/s2, init 10-29, seed 7, native collection mode; per task t0 19, t1 20, t2 8, t3 7, t4 17, t5 15, t6 15, t7 19, t8 0, t9 7; trace hashes in the json): decision-time gripper z-axis per decision, keyed by the public instruction. Fixed non-parametric store selected by development success labels at build time (as RTCF's only_success memory); 0 trainable parameters, no fitting. Both hashes are checked at load and a mismatch refuses to start. Never rebuilt or extended with init 30-49 content; validation uses the same memory.",
    "Lookup and gate (C2, C3, pinned to ideation/scripts-r5-memory/dev80_tube.py): exact match of obs['task_description'] to a memory instruction (9 LIBERO-10 instructions hit; t8 abstains = bitwise native). D(q) = min angle (deg) between the decision-time gripper z-axis z = R(states[3:6])[:, 2] and same-instruction memory rows q' in [q-4, q+4] by decision index (clamped; a memory shorter than q-4 contributes its last row). Arm at decision q when D(q-1) > 15 and D(q) > 15 and q <= 39 (step <= 195); re-arming after a disarm follows the same rule (only before step 200). While armed: D <= 7.5 deg disarms; step > 300 disarms; otherwise steer.",
    "Steering (T): u = unit(z x z_ref) (world frame; robosuite 1.4 OSC delta rotation is world-frame, left-multiplied, so a positive rotation about u moves z toward z_ref); c = clip(D/15, 0.5, 1.0); every executed row r < 5: a_rot' = a_rot - (a_rot . u) u + c u, then clip to [-1, 1]. A native openpi zero chunk (tol 1e-4) while armed gets rotation c u with the other rotation components zeroed (translation/gripper native). Translation (dims 0-2), gripper (dim 6) and rows >= 5 are native; every non-steered decision returns the native chunk object (bitwise).",
    "Scalars (C9, A5.1/A5.6): eps 15 deg and the arming window (decision 39 = step 195) were chosen on a 3x3 grid ({10,15,20} x {150,200,250} steps) over S2 leave-one-out and the recorded dev80 identity rollouts (in-sample for the screen; only the validation block init 38-49 is out of sample); consecutive 2 and W 4 decisions fixed before any scan; release 7.5 = eps/2; the gain floor 0.5 (from the measured native away-drive) and the hold limit step 300 (from S2 success lengths) are development calibrations, not structural constants; ortho cap 6 decisions = T's F1 return window, fixed before outcomes. No change after any A5 outcome.",
    "Exposure (offline, gate only): dev80 identity (memory never saw init 30-37) hard excl t8 0/16 successes vs 19/24 failures (t2 6/7, t3 6/6, t5 4/5, t9 3/4, t6 0/2), easy 0/29 vs 2/3; S2 leave-one-out hard 0/52 vs 41/48, easy 0/75 vs 1/5. Closed-loop exposure is measured by the screen (F4).",
    "Inputs and cost: obs['states'] and obs['task_description'] passed to the action_op (decision-time observation) and the native chunk; no RNG, simulator state, reward, success, case id, seed or arm. Inference ratio 1.00, 0 LLM/tool/SigLIP calls; derived null (shadow tube computation, native actions); model_training_steps 0.",
]
ORTHO = ("Ortho control (C6; DEVELOPMENT-ONLY, never a Campaign arm or validation/final scene): identical arming law and "
         "first arming decision; replaces the component along u_perp = unit(z x u) with s c (s = +/-1 from "
         "sha256('tube-steer/ortho' | decision index), a public coin; c as T), keeps the native component along u; "
         "zero-chunk rows get rotation s c u_perp; steers at most 6 decisions or until D <= 7.5 deg (whichever first), then "
         "native for the rest of the episode (no re-arm). Sideways rotation never reduces D, so an uncapped ortho would stay "
         "armed to step 300.")


def card(ortho: bool) -> dict:
    name = "tube-steer-ortho" if ortho else "tube-steer"
    return {
        "name": name, "version": "1.0",
        "paper": {"id": "rmh:rtcfpp-20261101/tube-steer",
                  "title": "tube-steer: gated success-tube orientation steering from a frozen development success memory - "
                           "new training-free mechanism (original RoboMetaHarness method, run rtcfpp-20261101 round 5); not a "
                           "reproduction of any cited paper",
                  "venue": "none (internal)"},
        "categories": {"primary": "memory_retrieval", "secondary": ["execution_monitoring"]},
        "fidelity": "tf_variant",
        "substitutions": ["RTCF (arXiv 2608.04527, withdrawn) retrieval + DCT low-frequency correction toward retrieved success chunks -> nearest same-instruction success-tube orientation (decision-time gripper z-axis memory from reserved init 10-29 successes) and a gated restoring rotation (eps 15 deg, 2 consecutive decisions, arm by step 195)"],
        "requires": {"capabilities": ["act", "chunk"]},
        "budgets": {"policy_calls": None, "llm_calls": 0, "tool_calls": 0, "max_tokens": 0},
        "null": "derived",
        "artifacts": [
            {"name": "tube_memory_npz", "path": str(PKG / "assets/tube_memory.npz"), "sha256": NPZ_SHA,
             "license": "internal development artifact (rtcfpp-20261101), built from S2 init 10-29 successes"},
            {"name": "tube_memory_json", "path": str(PKG / "assets/tube_memory.json"), "sha256": JSON_SHA,
             "license": "internal development artifact (rtcfpp-20261101)"},
            {"name": "pi0_fast_libero_norm_stats",
             "path": "/bigdata/sejain/hs2-cache/openpi/openpi-assets/checkpoints/pi0_fast_libero/assets/physical-intelligence/libero/norm_stats.json",
             "sha256": NORM_SHA, "license": "Apache-2.0 (openpi assets); byte-identical copy in hs_tube_steer/assets"}],
        "operators": [],
        "development_only": ortho,
        "params": {"mode": "ortho" if ortho else "steer", **PARAMS},
        "disclosures": DISCLOSE + ([ORTHO] if ortho else []),
        "original": {"repo": "none (original method)", "commit": "",
                     "reported": "none; predictions in " + RUN + "/ideation/r5-memory.json (hard48 net +4, 80% interval -1..+7)"},
        "description": ("Development-only ortho control of tube-steer" if ortho else
                        "Gated restoring rotation toward the nearest same-instruction success-tube orientation for pi0_fast_libero") + "; bitwise native when not steering.",
        "tier1": {"instruction": "pick up the block and place it in the bowl", "horizon": 40, "chunk_len": 10},
        "changelog": ["v1.0 (2026-11-01): first implementation (task implement-tube-steer, coder a7156c5c79cdb4ce5) under "
                      "review C1-C13 and amendment A5; frozen before any tube-steer episode."],
    }


def main() -> None:
    for ortho, fname in ((False, "paper.yaml"), (True, "paper_ortho.yaml")):
        (PKG / fname).write_text("# GENERATED by scripts/make_cards.py (tube-steer v1.0); edit the generator, not this file.\n"
                                 + yaml.safe_dump(card(ortho), sort_keys=False, width=120))
        print(PKG / fname)


if __name__ == "__main__":
    main()
