"""Generate the hs_u5z cards (paper_u5z-t.yaml, paper_u5zsign-t.yaml). The component params are COPIED from the frozen,
sha256-verified component cards (no new scalar); the glue re-checks equality at construction. Run once before the freeze:
    harnessstudio/.venv/bin/python harnessstudio/methods/u5z/scripts/make_cards.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

from hs_u5z import glue

PKG = Path(__file__).resolve().parents[1] / "hs_u5z"
RUN = "runs/rtcfpp-20261101"
TS_CARD = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio/methods/tube_steer/hs_tube_steer/paper.yaml"

DISCLOSE = [
    "Composite of two frozen components, not a new mechanism (A6.Z.8). U5 = T = tube-steer 1.0.0 (hs-tube-steer, manifest "
    + RUN + "/implementation/tube-steer/manifest.json sha256 a36e974e..., fingerprint cf66c4b8..., card paper.yaml b02f5e59..., "
    "action_op.py 6d3b1e1c...); Z = tilt-guard v1.2 'z' (hs-tilt-guard-v12, manifest " + RUN + "/implementation/tilt-guard-v12/"
    "manifest.json sha256 b79c35db..., paper_z.yaml e798eca9..., composite.py cdd6cad1..., action_op.py b4348f2f..., geometry.py "
    "d8f7bbb5...). Both are imported read-only from the installed frozen packages; every pinned file (code, card, assets) is "
    "sha256-verified before import at every construction, and every component parameter must equal the frozen card's params "
    "block (no new scalar). Only the precedence glue (hs_u5z/glue.py) is new.",
    "Frozen precedence (" + RUN + "/reviews/a5-decision.json tz_composite_ruling.frozen_spec; review " + RUN + "/reviews/"
    "u5z-proposal-review.json C5/C9; A6.Z.2): at every decision U5 is called FIRST on the NATIVE chunk with the decision-time "
    "observation; if U5 has not armed in this episode after that call, Z processes the SAME native chunk exactly as in v1.2 and "
    "Z's output is executed (Z's state: previous chunk, zero run, latched gripper, decision counter fed on every pre-arming "
    "decision); at U5's first arming decision and every later decision U5's output is executed, U5's own zero-chunk handling "
    "applies unchanged (also after a U5 disarm), and Z is never called or executed again. A zero chunk AT U5's arming decision "
    "belongs to U5. No other precedence (e.g. 'Z whenever U5 is not armed') is admissible this session.",
    "Control u5zsign-t (DEVELOPMENT-ONLY, A6.Z.4(b)): identical glue and precedence with Z replaced by tilt-guard v1.1 'zsign' "
    "(hs-tilt-guard-v11 TiltGuardActionOp from the byte-identical action_op.py b4348f2f..., exactly the params of v1.1 "
    "paper_zsign.yaml sha256 8be57c4d...; guard false, so hs_band_guard is never imported); the glue checks at construction "
    "that these params equal paper_z v1.2's except 'zero' (signctrl vs substitute) and the inert guard fields "
    "(guard_factory, guard_params). signctrl executes z + r*(sub - z) with per-dimension public coins r from sha256(first "
    "public state bytes | decision index).",
    "S_Z (A6.Z.3): " + RUN + "/analysis/scripts-r6/sz.py (sha256 2d379f59...) over the recorded round-5 T arm -> "
    "data-u5z/sz.json (e0df428f...), 25 scenes. Outside S_Z the composite equals U5 by construction (checked offline by "
    "scripts/replay_round5.py and live by the 2 spot checks t2i30, t2i34). U5 = T abstains on t8 (no success memory), so on t8 "
    "the composite equals Z for the whole episode.",
    "Disclosure (A6.Z.8): outcome-informed composition, selected after round-4 and round-5 development outcomes (Z's t2i31 "
    "rescue in the Campaign development grid); Campaign-30 entry and development pass are in-sample confirmations; only "
    "validation (init 38-49) is out of sample. Inputs: obs['states'], obs['task_description'] and the native chunk; no "
    "simulator state, reward, success, case id, seed or arm. Inference ratio 1.00, 0 LLM/tool/VLM calls, trainable "
    "parameters 0, model_training_steps 0; derived null (shadow: glue, U5 and Z run, native actions executed).",
]


def component_ref(name: str) -> dict:
    comp = glue.COMPONENTS[name]
    return {"component": name, "factory": comp["factory"], "params": glue.card_params(name)}


def card(name: str) -> dict:
    u5_name, z_name = glue.VARIANTS[name]
    ts = yaml.safe_load(open(TS_CARD))
    control = name.startswith("u5zsign")
    return {
        "name": name, "version": "1.0",
        "paper": {"id": "rmh:rtcfpp-20261101/u5z",
                  "title": "U5+Z: frozen tube-steer with frozen tilt-guard v1.2 zero-chunk substitution before U5's first "
                           "arming - composition of two original RoboMetaHarness methods (run rtcfpp-20261101, A6.Z); not a "
                           "reproduction of any cited paper",
                  "venue": "none (internal)"},
        "categories": {"primary": "memory_retrieval", "secondary": ["execution_monitoring"]},
        "fidelity": "tf_variant",
        "substitutions": list(ts["substitutions"]),
        "requires": {"capabilities": ["act", "chunk"]},
        "budgets": {"policy_calls": None, "llm_calls": 0, "tool_calls": 0, "max_tokens": 0},
        "null": "derived",
        "artifacts": list(ts["artifacts"]),
        "operators": [],
        "development_only": control,
        "role": ("control U5+Zsign (A6.Z.4(b))" if control else "candidate U5+Z (A6.Z)"),
        "components": {"u5": u5_name, "z": z_name},
        "params": {"variant": name, "u5": component_ref(u5_name), "z": component_ref(z_name)},
        "disclosures": DISCLOSE,
        "original": {"repo": "none (original method)", "commit": "",
                     "reported": "none; offline predictions in " + RUN + "/campaign-proposals/u5z-proposal.json"},
        "description": ("U5+Zsign control" if control else "U5+Z") + ": tube-steer first on the native chunk; tilt-guard "
                       + ("v1.1 zsign" if control else "v1.2 z") + " output executed only before tube-steer's first arming.",
        "tier1": {"instruction": "pick up the block and place it in the bowl", "horizon": 40, "chunk_len": 10},
        "changelog": ["v1.0 (2026-11-01): first implementation (task implement-u5z) under " + RUN + "/reviews/a5-decision.json "
                      "tz_composite_ruling and u5z-proposal-review.json C1-C9; frozen before any composite episode."],
    }


def main() -> None:
    for name in glue.VARIANTS:
        text = ("# GENERATED by scripts/make_cards.py (hs-u5z 1.0); edit the generator, not this file.\n"
                + yaml.safe_dump(card(name), sort_keys=False, width=120, allow_unicode=False))
        (PKG / f"paper_{name}.yaml").write_text(text)
        print(PKG / f"paper_{name}.yaml")


if __name__ == "__main__":
    main()
