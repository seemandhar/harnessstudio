"""Generate the tube-steer-vfb cards (hs_tube_steer_vfb/paper.yaml, paper_ortho.yaml); copied from tube-steer's generator."""
from __future__ import annotations

from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1] / "hs_tube_steer_vfb"
RUN = "runs/rtcfpp-20261101"
NORM_SHA = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"
NPZ_SHA = "fcf8338306294cace8979ede718d2ede7d8b287198144fbf07e21604a5677f0a"
JSON_SHA = "00d2e87bc024ffed09fc04953e389f1bbded1f26c3ba3a4fe2be25d39fb2d180"
VNPZ_SHA = "ccebf9e0d3d80c709c116101835eea6c199fdac2a87fe7746cc2c98c732209a5"
VJSON_SHA = "023e365e8fc164cd18d7af9201967cd3248b31da225aeec934b10bc77bcfb6b0"
PARAMS = {"eps_deg": 15.0, "consecutive": 2, "arm_by_decision": 39, "window_decisions": 4, "release_deg": 7.5,
          "hold_limit_step": 300, "gain_floor": 0.5, "gain_cap": 1.0, "ortho_cap": 6, "exec_rows": 5, "decision_steps": 5,
          "zero_chunk_tol": 0.0001, "state_key": "states", "instruction_key": "task_description",
          "memory_npz": "assets/tube_memory.npz", "memory_npz_sha256": NPZ_SHA,
          "memory_json": "assets/tube_memory.json", "memory_json_sha256": JSON_SHA,
          "norm_stats": "assets/pi0_fast_libero_norm_stats.json", "norm_stats_sha256": NORM_SHA,
          "visual_k": 8, "image_key": "agentview", "encoder_url": "http://127.0.0.1:8149",
          "encoder_model": "google/siglip-so400m-patch14-224", "encoder_revision": "d04cf29fca7b6374f74d8bea1969314492266b5e",
          "encoder_pooling": "pooler_output",
          "encoder_weights_sha256": "30a00c26a045c84a0bd14adc94278d7f1cc941c8bd67e5fa50f7bed34c34e60d",
          "encoder_pinned_files": {"preprocessor_config.json": "bec6ca696ddd6c2050ce0ea9fcb7581b052d267059a3341b6c1456cab54b1d07",
                                   "config.json": "28c6981582dbd0dc6e1c2eeae21a240e6952019bca3cdffbc22b9da2c747cb6b"},
          "encoder_timeout": 30.0,
          "visual_memory_npz": "assets/visual_memory.npz", "visual_memory_npz_sha256": VNPZ_SHA,
          "visual_memory_json": "assets/visual_memory.json", "visual_memory_json_sha256": VJSON_SHA}

TV_DISCLOSE = [
    "tube-steer-vfb (TV) = the pre-registered A5 variant P2 (" + RUN + "/ideation/r5-memory.json P2; exact rule " + RUN + "/reviews/a5-decision.json p2_ruling.exact_rule; review r5-proposal-review.json): a self-contained copy of the frozen tube-steer (T, hs_tube_steer 1.0.0, manifest " + RUN + "/implementation/tube-steer/manifest.json; not imported) plus ONE fallback: when the public instruction has no success memory (LIBERO-10: t8 'put both moka pots on the stove'), the tube at decision q is the decision-time z-axes of the k = 8 visual-memory rows nearest (cosine over SigLIP pooler descriptors, all tasks, no task id, stable sort) to the live agent-view frame; D(q) = min angle to those 8 z-axes (no decision-index window, as visual_tube_loo.py). Arming (2 consecutive D > 15 deg by decision 39), release 7.5, hold limit step 300, gain, steering law, zero-chunk rows and the ortho law are T's unchanged. Instructions with a tube run T's code path: TV == T bitwise outside t8 (replay-tested).",
    "Visual memory (frozen): assets/visual_memory.npz (sha256 ccebf9e0..., = runs/rtcf-20261031/eval/s2/PREP/artifacts/memory.npz pinned by that prepared.json) and visual_memory.json (023e365e...), byte-identical copies; 6176 rows = the 127 S2 successes at init 10-29 (SigLIP d04cf29f pooler_output descriptors stored by RTCF collect mode). At load every row block is checked to be a success at init 10-29 aligned (same key, length) with the tube memory, whose decision-time z-axes supply the row orientations; any mismatch refuses to start. Only the descriptors and z-axes are used (no RTCF chunk codes).",
    "Encoder pin (deviation from the p2_ruling wording, accepted by the supervisor 2026-11-01): harnessstudio/endpoints.yaml has no SigLIP entry and is framework code hashed by Campaign freezes, so the endpoint is pinned in this card instead, following the RTCF precedent (docs/HARNESSSTUDIO_V2_CAMPAIGN.md): http://127.0.0.1:8149 (method-local service harnessstudio/methods/rtcf/scripts/siglip_service.py), model google/siglip-so400m-patch14-224, revision d04cf29fca7b6374f74d8bea1969314492266b5e, pooler_output, weights sha256 30a00c26..., preprocessor_config.json bec6ca69..., config.json 28c69815.... The /health reply must match all of them once per episode (first visual decision) and every /embed reply must echo model/revision/pooling; an unavailable or mismatching encoder or a missing agent-view image raises, i.e. an infrastructure failure, never a silent abstain or a semantic failure. Encoder calls happen only on decisions of instructions without a tube (t8); about 50 ms each, reported separately (policy inference ratio 1.0).",
    "Disclosures of the selection (p2_ruling): TV was built after T failed only the entry rule (Campaign-30 net 0); its decisive evidence is 8 t8 scenes and entry depends on t8 init 30-32; the t8 reference is cross-task (offline the retrieved rows are t2 moka-pot rows), a task-specific development prior, not zero-shot; k 8 and the encoder were frozen in A5 before any A5 episode. Only validation (init 38-49) is out of sample.",
]

DISCLOSE = [
    "New training-free mechanism; not a reproduction of any cited paper. Original RoboMetaHarness method (run rtcfpp-20261101, round-5 MEMORY proposal P1-tube-steer, " + RUN + "/ideation/r5-memory.json; review " + RUN + "/reviews/r5-proposal-review.json C1-C13; amendment A5). Reported as a task-specific success-memory envelope: a tf_variant inspired by RTCF (arXiv 2608.04527, withdrawn v3; audited in runs/rtcf-20261031/audit/paper-audit.md) with the substitution 'DCT low-frequency correction toward retrieved success chunks -> gated restoring rotation toward the nearest same-instruction success-tube orientation'. Never a reproduction or verified SOTA; in T t8 is untouched (no success memory) - in TV t8 uses the visual fallback described above; development evidence stays development evidence (C13).",
    "Memory (C1, C11): assets/tube_memory.npz (sha256 fcf83383...) and tube_memory.json (00d2e87b...), byte-identical copies of the frozen artifact built by " + RUN + "/ideation/scripts-r5-memory/build_tube_memory.py from the 127 SUCCESSFUL S2 memory episodes (runs/rtcf-20261031/eval/s2, init 10-29, seed 7, native collection mode; per task t0 19, t1 20, t2 8, t3 7, t4 17, t5 15, t6 15, t7 19, t8 0, t9 7; trace hashes in the json): decision-time gripper z-axis per decision, keyed by the public instruction. Fixed non-parametric store selected by development success labels at build time (as RTCF's only_success memory); 0 trainable parameters, no fitting. Both hashes are checked at load and a mismatch refuses to start. Never rebuilt or extended with init 30-49 content; validation uses the same memory.",
    "Lookup and gate (C2, C3, pinned to ideation/scripts-r5-memory/dev80_tube.py): exact match of obs['task_description'] to a memory instruction (9 LIBERO-10 instructions hit; in T t8 abstains = bitwise native, in TV t8 takes the visual fallback). D(q) = min angle (deg) between the decision-time gripper z-axis z = R(states[3:6])[:, 2] and same-instruction memory rows q' in [q-4, q+4] by decision index (clamped; a memory shorter than q-4 contributes its last row). Arm at decision q when D(q-1) > 15 and D(q) > 15 and q <= 39 (step <= 195); re-arming after a disarm follows the same rule (only before step 200). While armed: D <= 7.5 deg disarms; step > 300 disarms; otherwise steer.",
    "Steering (T): u = unit(z x z_ref) (world frame; robosuite 1.4 OSC delta rotation is world-frame, left-multiplied, so a positive rotation about u moves z toward z_ref); c = clip(D/15, 0.5, 1.0); every executed row r < 5: a_rot' = a_rot - (a_rot . u) u + c u, then clip to [-1, 1]. A native openpi zero chunk (tol 1e-4) while armed gets rotation c u with the other rotation components zeroed (translation/gripper native). Translation (dims 0-2), gripper (dim 6) and rows >= 5 are native; every non-steered decision returns the native chunk object (bitwise).",
    "Scalars (C9, A5.1/A5.6): eps 15 deg and the arming window (decision 39 = step 195) were chosen on a 3x3 grid ({10,15,20} x {150,200,250} steps) over S2 leave-one-out and the recorded dev80 identity rollouts (in-sample for the screen; only the validation block init 38-49 is out of sample); consecutive 2 and W 4 decisions fixed before any scan; release 7.5 = eps/2; the gain floor 0.5 (from the measured native away-drive) and the hold limit step 300 (from S2 success lengths) are development calibrations, not structural constants; ortho cap 6 decisions = T's F1 return window, fixed before outcomes. No change after any A5 outcome.",
    "Exposure (offline, gate only): dev80 identity (memory never saw init 30-37) hard excl t8 0/16 successes vs 19/24 failures (t2 6/7, t3 6/6, t5 4/5, t9 3/4, t6 0/2), easy 0/29 vs 2/3; S2 leave-one-out hard 0/52 vs 41/48, easy 0/75 vs 1/5. Closed-loop exposure is measured by the screen (F4).",
    "Inputs and cost: obs['states'], obs['task_description'] and (TV visual fallback only) obs['images']['agentview'] passed to the action_op (decision-time observation) and the native chunk; no RNG, simulator state, reward, success, case id, seed or arm. Policy inference ratio 1.00, 0 LLM/tool calls; SigLIP /embed calls only on decisions of instructions without a tube (t8), plus one /health call per such episode, counted in diagnostics (visual.encoder_calls, encoder_s) and reported separately; derived null (shadow tube computation incl. the same SigLIP calls, native actions); model_training_steps 0; all weights frozen (pi0_fast_libero, SigLIP).",
]
ORTHO = ("Ortho control (C6; DEVELOPMENT-ONLY, never a Campaign arm or validation/final scene): identical arming law and "
         "first arming decision; replaces the component along u_perp = unit(z x u) with s c (s = +/-1 from "
         "sha256('tube-steer/ortho' | decision index), a public coin; c as T), keeps the native component along u; "
         "zero-chunk rows get rotation s c u_perp; steers at most 6 decisions or until D <= 7.5 deg (whichever first), then "
         "native for the rest of the episode (no re-arm). Sideways rotation never reduces D, so an uncapped ortho would stay "
         "armed to step 300.")


def card(ortho: bool) -> dict:
    name = "tube-steer-vfb-ortho" if ortho else "tube-steer-vfb"
    return {
        "name": name, "version": "1.0",
        "paper": {"id": "rmh:rtcfpp-20261101/tube-steer-vfb",
                  "title": "tube-steer-vfb: gated success-tube orientation steering with a visual-retrieval tube fallback - "
                           "new training-free mechanism (original RoboMetaHarness method, run rtcfpp-20261101 round 5); not a "
                           "reproduction of any cited paper",
                  "venue": "none (internal)"},
        "categories": {"primary": "memory_retrieval", "secondary": ["execution_monitoring"]},
        "fidelity": "tf_variant",
        "substitutions": ["RTCF (arXiv 2608.04527, withdrawn) retrieval + DCT low-frequency correction toward retrieved success chunks -> nearest same-instruction success-tube orientation (decision-time gripper z-axis memory from reserved init 10-29 successes), or for instructions without memory the z-axes of the k=8 SigLIP-nearest memory rows (RTCF's own descriptors and encoder), and a gated restoring rotation (eps 15 deg, 2 consecutive decisions, arm by step 195)"],
        "requires": {"capabilities": ["act", "chunk"]},
        "budgets": {"policy_calls": None, "llm_calls": 0, "tool_calls": 0, "max_tokens": 0},
        "null": "derived",
        "artifacts": [
            {"name": "tube_memory_npz", "path": str(PKG / "assets/tube_memory.npz"), "sha256": NPZ_SHA,
             "license": "internal development artifact (rtcfpp-20261101), built from S2 init 10-29 successes"},
            {"name": "tube_memory_json", "path": str(PKG / "assets/tube_memory.json"), "sha256": JSON_SHA,
             "license": "internal development artifact (rtcfpp-20261101)"},
            {"name": "visual_memory_npz", "path": str(PKG / "assets/visual_memory.npz"), "sha256": VNPZ_SHA,
             "license": "internal development artifact (runs/rtcf-20261031/eval/s2/PREP, S2 init 10-29 successes); SigLIP descriptors (Apache-2.0 model)"},
            {"name": "visual_memory_json", "path": str(PKG / "assets/visual_memory.json"), "sha256": VJSON_SHA,
             "license": "internal development artifact"},
            {"name": "pi0_fast_libero_norm_stats",
             "path": "/bigdata/sejain/hs2-cache/openpi/openpi-assets/checkpoints/pi0_fast_libero/assets/physical-intelligence/libero/norm_stats.json",
             "sha256": NORM_SHA, "license": "Apache-2.0 (openpi assets); byte-identical copy in hs_tube_steer_vfb/assets"}],
        "operators": [],
        "development_only": ortho,
        "params": {"mode": "ortho" if ortho else "steer", **PARAMS},
        "disclosures": TV_DISCLOSE + DISCLOSE + ([ORTHO] if ortho else []),
        "original": {"repo": "none (original method)", "commit": "",
                     "reported": "none; predictions in " + RUN + "/ideation/r5-memory.json (hard48 net +4, 80% interval -1..+7)"},
        "description": ("Development-only ortho control of tube-steer-vfb" if ortho else
                        "tube-steer with a SigLIP visual-retrieval tube for instructions without success memory (t8) for pi0_fast_libero") + "; bitwise native when not steering.",
        "tier1": {"instruction": "pick up the block and place it in the bowl", "horizon": 40, "chunk_len": 10},
        "changelog": ["v1.0 (2026-11-01): first implementation of the A5 variant TV under a5-decision.json p2_ruling and review r5 "
                      "C1-C13: draft by coder a7156c5c79cdb4ce5 (task implement-tube-steer-vfb, killed before hand-off, unreviewed), "
                      "verified, completed and frozen by coder a50d6f501ee76d694 (task implement-tube-steer-vfb-r2) before any TV episode."],
    }


def main() -> None:
    for ortho, fname in ((False, "paper.yaml"), (True, "paper_ortho.yaml")):
        (PKG / fname).write_text("# GENERATED by scripts/make_cards.py (tube-steer-vfb v1.0); edit the generator, not this file.\n"
                                 + yaml.safe_dump(card(ortho), sort_keys=False, width=120))
        print(PKG / fname)


if __name__ == "__main__":
    main()
