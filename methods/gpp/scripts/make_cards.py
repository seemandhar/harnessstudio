#!/usr/bin/env python3
"""Generate the gpp cards (hs_gpp/paper_<entry>.yaml) from ONE frozen constant set, so every entry (candidate, shams,
H2-only fallback, the GE composition) shares the constants exactly. Run once before freezing; the cards are the frozen
artifacts (review proposals-r2-review.json D4: every constant frozen now, no sweep before the Campaign).

    harnessstudio/.venv/bin/python harnessstudio/methods/gpp/scripts/make_cards.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1] / "hs_gpp"
GE_CARD = Path(__file__).resolve().parents[2] / "greedy_escape/hs_greedy_escape/paper.yaml"

H2 = {"min_close": 3, "within": 6, "max_fires": 2, "refire_rise": 0.02, "open_steps": 3, "speed": 0.5,
      "distance": 0.03, "max_move_steps": 8, "stop_dz": 0.001, "stop_from_move": 2}
H1 = {"min_run": 10, "clamp_w": 0.01, "settle": 0.0005, "max_rise": 0.005, "max_descent": 0.02, "max_rot": 1.0,
      "chain_gap": 5, "chain_lift": 0.03, "max_fires": 3, "speed": 0.5, "max_steps": 12, "height": 0.06, "check_at": 4,
      "min_rise": 0.01}
H3 = {"min_step": 25, "identical": 20, "eps": 0.01, "max_move": 0.02, "open_steps": 0, "speed": 0.5, "distance": None,
      "max_move_steps": 8, "stop_dz": 0.001, "stop_from_move": 2}

ENTRIES = {
    "gpp-core": ({"h2": "on", "h1": "on", "h3": "off", "ge": False},
                 "gpp-core candidate: greedy + H2 deeper re-close + H1 clamp-lift (H3 off, greedy-escape off)."),
    "gpp-core-sham": ({"h2": "sham", "h1": "sham", "h3": "off", "ge": False},
                      "gpp-core directional sham (N2): same triggers and step budgets; H2 opens and ASCENDS 3 cm, H1 holds "
                      "closed 12 steps without lifting (veto-only). Development-screen arm only."),
    "gpp-h2": ({"h2": "on", "h1": "off", "h3": "off", "ge": False},
               "gpp-H2 fallback (time guard): greedy + H2 deeper re-close only."),
    "gpp-h2-sham": ({"h2": "sham", "h1": "off", "h3": "off", "ge": False},
                    "gpp-H2 directional sham: H2 opens and ASCENDS 3 cm. Development-screen arm only."),
    "ge-gpp-core": ({"h2": "on", "h1": "on", "h3": "off", "ge": True},
                    "greedy-escape + gpp-core composition (pre-approved outcome-blind, CONDITIONAL on greedy-escape's "
                    "recorded B verdict; review compositions): H2, H1 and greedy-escape's frozen StallMonitor + SCALE windows."),
}

DISCLOSURES = [
    "New training-free mechanism (D2): not a SCALE variant. SCALE is decoder-only; gpp overrides gripper commands and "
    "inserts scripted EE-delta motions computed from public proprio (analytic_control, D1). Every scripted step is "
    "logged (ctx.record gpp phase=scripted) and counted in accounting.extra['gpp'].scripted_steps.",
    "Scripted steps use the robot's own 7-D normalized EE-delta + gripper action space at policy-like magnitude "
    "(|a_z| = 0.5 vs the policy's p5/p95 -0.54/0.514), make no policy call and count against the official 520-step "
    "horizon. No extra environment steps, resets or simulator queries.",
    "Public inputs: executed action history and obs['states'] (eef xyz, axis-angle, finger qpos; send_state: true is "
    "part of the frozen protocol, D3). No object pose, contact flag, reward, success or simulator state.",
    "D4: triggers and constants were designed in-sample on all 100 development scenes over 4 arms (H1's exclusions for "
    "knob twists, placements and blip-split carries remove success exposures). The 0/54 greedy-success exposure holds by "
    "construction and is not retention evidence. There is no unused development slice; validation (init 15-24) is the "
    "first out-of-sample test. Constants frozen 2026-11-02 before any gpp episode; no sweep before the Campaign.",
    "H2 online rule (H2-RC1) fires at max(c+2, c+lat) where the offline replay used c+lat with the whole-run length; "
    "the fire set is recomputed with the online rule on the 100 greedy records (tests/test_replay_devmap.py).",
    "H2 2nd fire (frozen interpretation of 'risen >= 2 cm since the first'): the maximum EE z after the 1st primitive "
    "ended minus the z at the 1st fire >= 2 cm. H1 abort executes the vetoed policy action (the open) as one scripted "
    "step. Shams: H1 veto-only always runs 12 hold steps (no abort); H2 ascend uses the mirrored stop (realized "
    "rise < 1 mm).",
    "Contact stop (v1.1, G1): under OSC delta control a command from rest ramps up (first realized step about a third of "
    "the sustained ~6.5 mm/step, sometimes ~0: 16% of 43 near-rest starts > -1 mm on the map), so the 1 mm stop rule is "
    "tested from move step 2 (free-space descent once moving: 0/209 steps > -1 mm). Each H2/H3 fire logs dz_mm.",
    "Null (shadow, N1): the controller runs and logs primitives on the native trajectory; native actions execute; its "
    "monitors see the actions it requested (exact until the first fire).",
]


def card(name: str, modes: dict, description: str) -> dict:
    ge = modes["ge"]
    caps = ["act", "chunk", "seeded_sampling"] + (["isolated_streams", "decoding_hooks"] if ge else [])
    params = {"modules": modes, "h2": H2, "h1": H1, "h3": H3, "state_key": "states",
              "declarations": ["analytic_control", "send_state"]}
    subs = []
    if ge:
        g = yaml.safe_load(GE_CARD.read_text())["params"]
        params["ge"] = {k: g[k] for k in ("fixed_point_steps", "fixed_point_eps", "empty_close_steps", "empty_width_m",
                                          "empty_closes", "window_decisions", "max_windows", "escape")}
        subs = ["greedy-escape windows (SCALE decoding hook) inside the gpp controller: SCALE decoder state skips scripted "
                "steps; GE window decisions that coincide with a primitive's steps are consumed by the primitive"]
    return {
        "name": name, "version": "1.1",
        "paper": {"id": f"rmh:scalepp-20261102/{name}",
                  "title": "gpp: greedy base + public-proprio scripted primitives (deeper re-close, clamp-lift) - new "
                           "training-free mechanism (RoboMetaHarness run scalepp-20261102, proposal idea-r2b), not a "
                           "reproduction of any cited paper",
                  "venue": "none (internal)"},
        "categories": {"primary": "contact_feedback", "secondary": ["execution_monitoring", "skill_composition"]},
        "fidelity": "tf_variant" if ge else "component",
        "fidelity_note": "new mechanism (review D2); the framework's fidelity enum has no 'new' value, so 'component' "
                         "(as other original RoboMetaHarness methods) is used" + ("; tf_variant because SCALE windows are "
                                                                                 "composed" if ge else ""),
        "substitutions": subs,
        "requires": {"capabilities": caps},
        "method_capabilities": ["analytic_control"],
        "benchmark_params": {"send_state": True},
        "budgets": {"policy_calls": 520, "llm_calls": 0, "tool_calls": 0, "max_tokens": 0},
        "null": "shadow",
        "artifacts": [], "operators": [],
        "params": params,
        "original": {"repo": "none (original method)", "commit": "",
                     "reported": "none; predictions in runs/scalepp-20261102/ideation/idea-r2b.json and the review's "
                                 "screen_specs_amendment_B.gpp_core (critic central Delta_G +3, range 0..+7)"},
        "description": description,
        "disclosures": DISCLOSURES,
        "tier1": {"instruction": "put both moka pots on the stove", "horizon": 60, "chunk_len": 1},
        "changelog": [f"v1.0 (2026-11-02): first implementation (task impl-gpp-core) under review r2 H1-RC1..5, "
                      f"H2-RC1..4, D1-D4, N1-N4; frozen before any gpp episode.",
                      "v1.1 (2026-11-02, task gpp-v11): contact-stop revision G1 - the H2/H3 descent stop (realized step "
                      "> -1 mm) and the ascend sham's mirrored stop (< +1 mm) are tested from move step 2 "
                      "(stop_from_move: 2; move step 1 ends only at the 3 cm distance); per-step dz_mm logged. Reason: OSC "
                      "ramp-up from rest; greedy-map kinematics, critic review gpp-src-review; no gpp episode existed "
                      "(outcome-blind). 1 mm / 3 cm / 8 steps, triggers, fire set and H1 unchanged. G3: the GE module pins "
                      "greedy-escape by fingerprint f03731f1 and runtime file hashes."],
    }


def main() -> None:
    header = "# GENERATED by scripts/make_cards.py (gpp v1.1); edit the generator, not this file.\n"
    for name, (modes, desc) in ENTRIES.items():
        path = PKG / f"paper_{name.replace('-', '_')}.yaml"
        path.write_text(header + yaml.safe_dump(card(name, modes, desc), sort_keys=False, width=120))
        print(path)


if __name__ == "__main__":
    main()
