"""Generate the two ReDecode cards (hs_redecode/paper_<variant>.yaml). The tilt_guard params are copied from the frozen
tilt-guard v1.2 cards (tilt-guard-v12-z for 'redecode', tilt-guard-v12-bz for 'redecode-b'); tests check equality.
Run once before the freeze: harnessstudio/.venv/bin/python harnessstudio/methods/redecode/scripts/make_cards.py"""
from __future__ import annotations

from pathlib import Path

import yaml
from harnessstudio.core import registry

PKG = Path(__file__).resolve().parents[1] / "hs_redecode"
NORM_SHA = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"
RUN = "runs/rtcfpp-20261101"
VARIANTS = {"redecode": ("tilt-guard-v12-z", "ReDecode -> Z", "control: tilt-guard-v12-z"),
            "redecode-b": ("tilt-guard-v12-bz", "ReDecode -> Z -> B", "control: tilt-guard-v12-bz")}

DISCLOSE = [
    "New training-free mechanism; not a reproduction of any cited paper. Original RoboMetaHarness method (run rtcfpp-20261101, round-4 CONTROL proposal r4c-p1-redecode, " + RUN + "/ideation/r4-control.json; review " + RUN + "/reviews/r4-proposal-review.json C-R1..C-R11; amendment A4 of campaign-inputs/preregistration-r1.md). Composition of known parts: observation-perturbation diversity (CycleVLA arXiv 2601.02295v2 / 2610.19438v1 / A3 2605.11567v4 use it for consensus at every decision; here it is a decode-repair device used only when the greedy decode provably failed) and the frozen tilt-guard v1.2 Z (and B) rules.",
    "Trigger: the NATIVE chunk (one seeded call, no override = the baseline's call) is the openpi zero chunk for the decision-time state (all rows identical, dims 0-5 = actions mean + s, gripper = actions mean, tol 1e-4). Retries: obs override {'states': s + delta} then {'states': s - delta}, delta_i = (std_i + 1e-6) * 2/256 on all 8 state dims (one z-score state-token bin, openpi tokenizer.py L70; std from the pinned norm stats sha256 4ece6c43...). A retry is valid iff its raw output is NOT the zero chunk for the state actually sent (C-R1). The number of the 8 state tokens a shift changes is logged per retry (public replica of the openpi digitize); a retry with no token change is still executed (it deterministically repeats the failure; frozen, no skip rule).",
    "Re-base (C-R3, a convention, not a correction): openpi AbsoluteActions adds back the overridden state; the first valid retry is executed with dims 0-5 minus (s_sent - s)[:6], so the command references the true state. The difference from the uncorrected output is exactly one state-token bin per dim: (std + 1e-6) * 2/256 = 0.00082/0.00119/0.00296/0.00269/0.00709/0.00254 for x/y/z/rx/ry/rz (gripper dims 0.00011; <= 0.0071 action units, <= 0.8% of the action std, largest on ry). The gripper is executed as decoded. Under m = a - s training the uncorrected output is arguably the model-consistent command; this is disclosed, frozen and unit-tested (tolerance 1e-5).",
    "Fallback and reuse (C-R9): the selected chunk (re-based valid retry, else the native zero chunk) passes through the FROZEN hs_tilt_guard_v12 TiltGuardV12ActionOp (imported read-only, params copied from the frozen v1.2 card) with zero='substitute' (tail, then hold, cap 2) and envelope off ('redecode') or bte ('redecode-b'). With all retries failing the executed rows equal v1.2 'z' / 'bz' bitwise; with no zero chunk they equal identity / v1.2 'b' bitwise. A successful retry is not a zero chunk for the op, so Z does not fire and B's boundary detector counts it as an ordinary policy decision; B may therefore arm differently than in v1.2 'bz'.",
    "Seeds (C-R4): each retry advances the framework decision counter, so later native calls use other CRN seeds than identity/Z; pi0-FAST decodes greedily (temperature 0), so this is harmless only if the served chunk is seed-invariant, which the evaluator's tier-2 check must show on the live server (same obs, two seeds -> bitwise identical chunk at a zero-chunk state and at a normal state) together with a bitwise R-vs-identity action prefix up to the first zero chunk. The framework logs every call's seeds ('sample' events); the method never reads them.",
    "Null: shadow (the controller runs in the null arm, pays its retries on the shadow policy session and logs W1 evidence; executed actions are native). Budget: policy_calls 312 = 3 x 104 native decisions at horizon 520 (worst case: every decision a zero chunk with two retries); expected ratio about 1.03 (dev80 1.9 zero chunks per episode), worst dev episode 1.25 (t8 i33). Inference ratio is measured per arm (candidate and null) by the evaluator. 0 LLM/tool calls; trainable_parameters 0; policy, checkpoint, normalization, images, instruction, cadence (5 rows) and horizon unchanged.",
    "Exposure (dev80 identity, in-sample): zero chunks in 28/32 hard failures vs 2/16 hard successes (t6 i32 at step 55, t9 i37 at 260); all-80 successes exposed 7/45 (t0 i30, t0 i37, t4 i31, t4 i32, t4 i34, t6 i32, t9 i37). Predictions (r4-control.json): hard48 net +2 vs identity (range -1..+4), +1 vs Z (a tie is plausible). All scalars are structural or inherited (one bin, two symmetric retries, tol 1e-4, Z fallback); no scalar was read from outcomes; only the validation block (init 38-49) is out of sample.",
]


def card(variant: str) -> dict:
    tg_entry, comps, control = VARIANTS[variant]
    tg = dict(registry.load(tg_entry).card["params"])
    params = {"shift_bins": [1, -1], "bin_width": 2.0 / 256, "max_retries": 2, "zero_chunk_tol": 0.0001,
              "exec_rows": 5, "state_key": "states", "norm_stats": "assets/pi0_fast_libero_norm_stats.json",
              "norm_stats_sha256": NORM_SHA, "tilt_guard": tg, "tilt_guard_source_entry": tg_entry}
    return {
        "name": variant, "version": "1.0",
        "paper": {"id": "rmh:rtcfpp-20261101/redecode",
                  "title": "ReDecode: fresh valid decode on FAST decode failure via a one-bin state-token shift, with "
                           "tilt-guard v1.2 Z fallback - new training-free mechanism (original RoboMetaHarness method, run "
                           "rtcfpp-20261101 round 4); not a reproduction of any cited paper",
                  "venue": "none (internal)"},
        "categories": {"primary": "uncertainty_compute", "secondary": ["execution_monitoring"]},
        "fidelity": "component",
        "substitutions": [],
        "requires": {"capabilities": ["act", "chunk", "seeded_sampling"]},
        "budgets": {"policy_calls": 312, "llm_calls": 0, "tool_calls": 0, "max_tokens": 0},
        "null": "shadow",
        "artifacts": [{"name": "pi0_fast_libero_norm_stats",
                       "path": "/bigdata/sejain/hs2-cache/openpi/openpi-assets/checkpoints/pi0_fast_libero/assets/physical-intelligence/libero/norm_stats.json",
                       "sha256": NORM_SHA, "license": "Apache-2.0 (openpi assets); byte-identical copy in hs_redecode/assets"}],
        "operators": [],
        "components": comps, "control": control, "development_only": False,
        "params": params, "disclosures": DISCLOSE,
        "original": {"repo": "none (original method)", "commit": "",
                     "reported": "none; predictions in " + RUN + "/ideation/r4-control.json (hard48 net +2 vs identity, +1 vs Z)"},
        "description": f"{comps} for pi0_fast_libero: re-decode FAST zero chunks with a one-bin state shift, then the frozen "
                       f"tilt-guard v1.2 op; bitwise native when the native chunk is valid and v1.2 is inactive.",
        "tier1": {"instruction": "pick up the block and place it in the bowl", "horizon": 40, "chunk_len": 10},
        "changelog": ["v1.0 (2026-11-01): first implementation (task implement-redecode, coder a7156c5c79cdb4ce5) under "
                      "review C-R1..C-R11 and amendment A4; frozen before any ReDecode episode."],
    }


def main() -> None:
    for v in VARIANTS:
        path = PKG / f"paper_{v}.yaml"
        path.write_text("# GENERATED by scripts/make_cards.py (redecode v1.0); edit the generator, not this file.\n"
                        + yaml.safe_dump(card(v), sort_keys=False, width=120))
        print(path)


if __name__ == "__main__":
    main()
