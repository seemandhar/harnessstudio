"""Generate the twelve tilt-guard v1.1 paper cards (hs_tilt_guard_v11/paper_<variant>.yaml) from one template
(copied from tilt-guard v1.0 scripts/make_cards.py; adds the r2 saturation-arming flag).

Run once before the freeze (the cards are then committed package files and hashed in the freeze manifest):
    harnessstudio/.venv/bin/python harnessstudio/methods/tilt_guard/scripts/make_cards.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1] / "hs_tilt_guard_v11"
NORM_SHA = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"
RUN = "runs/rtcfpp-20261101"

# Copy of hs_band_guard's frozen card params (band-guard v1.0); tests/test_tilt_guard.py checks equality with the
# installed hs_band_guard card, and the freeze manifest records the band_guard manifest sha256.
GUARD_PARAMS = {
    "dim": 3, "state_q01": 1.5003631114959717, "state_q99": 3.2705323696136475,
    "action_q01": -3.306384325027466, "action_q99": -1.4915390014648438, "margin": 0.02, "gate_rows": 5,
    "zero_chunk_tol": 0.0001, "norm_stats": "assets/pi0_fast_libero_norm_stats.json",
    "norm_stats_sha256": NORM_SHA, "state_key": "states", "correction": "restore",
}

COMMON = {
    "guard_factory": "hs_band_guard.action_op:BandGuardActionOp",
    "guard_params": GUARD_PARAMS,
    "exec_rows": 5,
    "theta_deg": 55.0,
    "window_steps": 150,
    "hold_decisions": 4,
    "width_min": 0.004,
    "kappa_per_rad": 0.5,
    "omega_max": 0.2,
    "singular_eps": 0.001,
    "yaw_seed": 20261101,
    "zero_chunk_tol": 0.0001,
    "zero_max_consecutive": 2,
    "norm_stats": "assets/pi0_fast_libero_norm_stats.json",
    "norm_stats_sha256": NORM_SHA,
    "state_key": "states",
    "detector": "first-boundary/hold4-loose",
    "arm_tilt_deg": 25.0,
    "c_th": 0.5,
    "pin_margin": 0.02,
    "axis_guard": 0.17,
}

VARIANTS = {
    #              guard  envelope   zero       sat arming  role
    "gb":         (True, "bte", "off", "off", "candidate component set G+B"),
    "gyaw":       (True, "yawctrl", "off", "off", "DEVELOPMENT-ONLY control: G + yaw-matched norm control of B (domain C9)"),
    "gbz":        (True, "bte", "substitute", "off", "candidate component set G+B+Z"),
    "gz":         (True, "off", "substitute", "off", "candidate component set G+Z"),
    "gbzsign":    (True, "bte", "signctrl", "off", "DEVELOPMENT-ONLY control: G + B + random-sign Z (general P2 F3 on top of G+B)"),
    "gzsign":     (True, "off", "signctrl", "off", "DEVELOPMENT-ONLY control: G + random-sign Z (general P2 F3 on top of G)"),
    "z":          (False, "off", "substitute", "off", "component Z alone (general review C8 standalone rule)"),
    "zsign":      (False, "off", "signctrl", "off", "DEVELOPMENT-ONLY control: random-sign Z alone (general P2 F3)"),
    "gbsatz":     (True, "bte", "substitute", "on", "candidate C*[B->Bsat] = G+Bsat+Z (amendment A1.2)"),
    "gbsat":      (True, "bte", "off", "on", "candidate C*[B->Bsat] = G+Bsat (amendment A1.2)"),
    "gbsatz-yaw": (True, "bte", "substitute", "yawctrl", "DEVELOPMENT-ONLY control C*[B->Bsat_yaw] for G+Bsat+Z (A1.2)"),
    "gbsat-yaw":  (True, "bte", "off", "yawctrl", "DEVELOPMENT-ONLY control C*[B->Bsat_yaw] for G+Bsat (A1.2)"),
}

SAT_DISCLOSE = [
    "v1.1 = tilt-guard v1.0 (frozen, hs_tilt_guard, manifest runs/rtcfpp-20261101/implementation/tilt-guard/manifest.json) plus ONE switchable flag saturation_arming (r2 residual P1, review runs/rtcfpp-20261101/reviews/r2-residual-review.json frozen_spec_p1 / C1-C10, amendment A1). With saturation_arming='off' the outputs equal v1.0 bitwise (test S3); every variant logs the shadow diagnostic sat_only_would_act (and sat_c, sat_tilt_deg, sat_pinned, sat_armed_now, sat_first, sat_window_open, b_window_open, route), including the derived null.",
    "Saturation arming (first event only, from decision 0, independent of the boundary detector, on the NATIVE chunk before Z and G): 8-d finite state; native chunk valid and not the openpi zero chunk (tol 1e-4); |d x u| >= 0.17 (tilt in ~(9.8, 170.2) deg); tilt > 25 deg; c = mean over rows 0-4 of clip(native rotation) . (d x u)/|d x u| > 0.5 (commanded tilt increase); some native[r, i] - s[i] (r < 5, i = 3..5) <= actions q01[i] + 0.02 or >= actions q99[i] - 0.02 (pinned output; pinned norm stats sha256 4ece6c43...). Window: the arming decision and the next 29 decisions; OR'ed with B's boundary window; B's correction law, theta 55 deg and gates unchanged. 'yawctrl' (development-only control): decisions active only through the saturation window get the yaw-matched rows (yaw_coin(20261101, decision)); boundary-window decisions get real B.",
    "Bsat scalars (r2 C5, A1.7): arm tilt 25 deg, c_th 0.5, pin margin 0.02, axis guard 0.17, first-event 30-decision window; chosen among 5 commanded-tilt variants on dev80 (init 30-37) + S2 (init 10-29) including their success exposure, so the screen is in-sample for them; only a fresh validation block (init 38-49) is out of sample. Any change after a Bsat outcome is a new variant with recorded exposure.",
    "Bsat predictions restated (r2 C6): on identity traces the saturation route arms in dev 30/32 hard failures + 1/45 successes (t9 i37) and S2 64/68 + 1/127 (t9 i28); decisions active ONLY through the saturation window (divergence from G+B+Z) occur in dev 29/32 hard failures, 3 easy failures (t4 i35, t4 i37, t7 i35), 0/45 successes; S2 59/68, 3, 0/127. The proposal's net +2 was computed on the narrower Q notion (~11 newly armed episodes). Rollout counts differ once G/B act. No task-, instruction- or object-keyed exception.",
]

DISCLOSE_COMMON = [
    "New training-free mechanism; not a reproduction of any cited paper. Original RoboMetaHarness method (run rtcfpp-20261101, round 1): G = r1g-p1 and Z = r1g-p2 by agent acd380ee23582d182 (" + RUN + "/ideation/r1-general.json sha256 3bcf58eb..., review a8623158c27233d0a C1-C10), B = DOMAIN P1 'boundary tilt envelope' by agent aa2fe8097e1c16a87 (" + RUN + "/ideation/r1-domain.json sha256 87bc1098..., review a92b7c1738d067767 C1-C14). RTCF (arXiv:2608.04527, withdrawn) is cited only as the motivating failure analysis; nothing of RTCF is reused. Novelty is stated only as: not found in the local knowledge catalog or the 2026-11-01 literature delta.",
    "Order (domain C8): (1) Z detects the openpi zero chunk on the NATIVE chunk and, if it substitutes, executes the previous decision's post-G (pre-B) rows 5-9 with the latched gripper (last executed gripper command; the zero chunk's own gripper value if none was executed yet); no re-guarding; cap (general C4) at 2 consecutive substitutions: run 1 tail (hold when no valid tail), run 2 hold (zero delta on dims 0-5 + latched gripper), run >= 3 native passthrough. (2) G = hs_band_guard BandGuardActionOp (imported read-only, frozen band-guard v1.0 params copied into guard_params) on every non-substituted chunk. (3) B last, on rows[:5] of whatever executes, never on an executed native zero chunk. G and B never touch translation (dims 0-2) or the gripper (dim 6). Rows 5-9 of the returned chunk are left as produced by (1)-(2); the candidate executes rows[:5] (profile execute_horizon 5).",
    "Boundary detector (ONE, frozen, domain C2): per decision, g = sign(native row-0 gripper), width = states[6] - states[7] of the decision-time state; a native zero chunk neither increments nor resets the run (never a release); g > 0 and width > 0.004 m: run += 1; g <= 0: boundary if run >= 4 else run = 0; g > 0 with width <= 0.004 (empty close) neither increments nor resets. FIRST boundary only. This is the detector of ideation/scripts-domain/dev_activation.py / tilt_window.py (the predeclared activation table, 28 dev80 activated episodes), NOT the strictly-consecutive text variant (25; differs on t3 i30/i35/i36). Window = the boundary decision and the next 29 decisions (150 env steps at 5 actions per call).",
    "B geometry: u = R(states[3:6])[:, 2] (Rodrigues, valid for |aa| up to 2 pi, so independent of quat_no_antipodal), tilt = angle(u, -e_z), k = normalize(u x -e_z); robosuite 1.4 OSC delta rotation is world-frame and left-multiplied (checked on 15763 dev steps, geometry test T1). Active iff boundary found, decision in window, tilt > 55 deg, 8-d finite state, executed chunk not a native zero chunk. Per executed row: clip to [-1, 1], remove the negative k-part, add push = min(0.5/rad * (tilt - 55 deg), 0.2) about +k, re-clip per component (C7). Deviation (disclosed): if the per-component re-clip made the k-component negative, the step-3 vector is instead scaled uniformly into the box (keeps its direction; counted as b_rows_rescaled). Singularity (C6): |u x d| < 1e-3 -> previous decision's k, else the horizontal projection of the EE x-axis.",
    "Scalars (domain C1, general C1): theta 55 deg, W 150 steps, hold 4 decisions, width 0.004 m, kappa 0.5 per rad, omega_max 0.2 action units, singular eps 1e-3, zero-chunk tol 1e-4, max 2 consecutive substitutions, yaw seed 20261101, G margin 0.02 and band/saturation edges copied from the checkpoint norm stats (sha256 4ece6c43...). theta and W were picked from a 6-point grid {45,55,65} x {100,150} on dev80 (init 30-37, which contains the hard48 futility scenes and the 30 Campaign entry cases) plus S2 (init 10-29); futility and entry are therefore in-sample for these scalars and only a fresh validation block (init 38-49) is out of sample. No retuning after screen outcomes: any change is a new variant/version.",
    "Regression risk (domain C3): the domain review reports that B's gate exposes about 2.9% of successes (dev80 1/45: t9 i37; S2 4/127: t3 i13, t5 i10, t5 i11, t9 i28), including 2/15 S2 t5 successes (book reorientation) with a boundary. Correction found by this implementation's replay (implementation/tilt-guard/replay-dev.json T8): in analysis/s2_compact.npz '_dstep' is the step AFTER a decision's executed rows, so the S2 checks read the next decision's gripper; with the decision's own gripper (what the live op uses) S2 exposes 2/127 successes (t3 i13, t9 i28) and the t5 i10/i11 'boundaries' were next-decision zero-chunk grippers; exposure is then 3/172 (1.7%). dev80 and S2 t5 successes show no boundary, so the hard48 screen cannot see a t5 channel; it remains an untested final-test regression risk. No task- or instruction-keyed exception exists.",
    "G disclosure (general C9): algebraically a gated AbsoluteActions(clip(s)) on rx: pi0_fast_libero (openpi 981483d, extra_delta_transform) outputs m = a - s on dims 0-5 and AbsoluteActions adds the decision-time state back; G uses clip(s_rx, state q01, state q99) when s_rx is outside the state band AND the policy output over rows[:5] is pinned at its training extreme. It targets an out-of-support artifact of the extra-delta parameterization and transfers only to checkpoints with delta-to-state output reconstruction. Z disclosure: it refuses openpi/HF FAST zero-fill decode-failure outputs (FAST arXiv:2501.09747v1 decode behaviour) that the baseline (and the RTCF reference) executes as served; the baseline is not mis-served.",
    "Inputs: obs['states'] passed to the action_op (ctx.observe() at decision start, the observation the policy received), the native chunk and this op's per-episode memory (previous post-G chunk, last executed gripper, zero-run length, boundary detector state, previous k). In the derived null the op runs as shadow (its memory follows what it would have executed, not what executed); diagnostics there are shadow evidence only. No RNG, simulator state, reward, success, case id, seed or arm. Inference ratio 1.00, 0 LLM/tool calls, trainable_parameters 0; policy, checkpoint, normalization and serving path unchanged in every arm.",
]

CONTROL_DISCLOSE = {
    "yawctrl": "Yaw-matched control (domain C9; development-only, never a Campaign arm or validation/final scene): same B gate on its own rollout; per executed row the delta is s * ||BTE(a) - clip(a)|| * e_z added to clip(a) (world vertical axis, tilt-neutral, k-component 0), with one sign per decision s from sha256('tilt-guard/yawctrl' | yaw_seed 20261101 | decision index): a fixed public seed and the decision index only. Not re-clipped (the per-row norm equals the BTE delta norm exactly; robosuite clips per component afterwards; realized norms are logged).",
    "signctrl": "Zero-chunk sign control (general P2 F3 and C3; development-only, never a Campaign arm or validation/final scene): same Z gate and cap on its own rollout; executes z + r * (sub - z) with a per-dimension coin r in {+1, -1} from sha256('tilt-guard/zerosign' | sha256(first public state bytes of the episode) | decision index), gripper clipped to [-1, 1].",
}

SUBSTITUTIONS = [
    "DynaHarness (arXiv 2609.40306v1) execution authority (slow VLM capability proposals + a deterministic fast envelope that grounds, refuses, bounds and substitutes commands over analytic skills) -> a proprio gravity-tilt envelope applied to the frozen policy's executed rotation rows (no VLM, no skills, no learned component)",
    "VLS (arXiv 2602.03973v1) stage-aware feedback (reward-relative stage switching with hysteresis) -> a proprio gripper-boundary gate (first policy release after a >= 4-decision hold with finger width > 0.004 m) that opens a fixed 150-step window",
]


def card(variant: str) -> dict:
    guard, envelope, zero, sat, role = VARIANTS[variant]
    params = {"guard": guard, "envelope": envelope, "zero": zero, "saturation_arming": sat, **COMMON}
    tf = envelope != "off"
    bname = {"off": "B", "on": "Bsat", "yawctrl": "B+yawsat"}[sat] if envelope == "bte" else "Byaw"
    comps = "+".join(x for x, on in (("G", guard), (bname, envelope != "off"),
                                     ("Z" if zero == "substitute" else "Zsign", zero != "off")) if on)
    disclosures = list(DISCLOSE_COMMON) + list(SAT_DISCLOSE)
    if envelope == "yawctrl":
        disclosures.append(CONTROL_DISCLOSE["yawctrl"])
    if zero == "signctrl":
        disclosures.append(CONTROL_DISCLOSE["signctrl"])
    dev_only = envelope == "yawctrl" or zero == "signctrl" or sat == "yawctrl"
    return {
        "name": f"tilt-guard-v11-{variant}",
        "version": "1.1",
        "paper": {"id": "rmh:rtcfpp-20261101/tilt-guard",
                  "title": "tilt-guard: decode-failure substitution, saturation-gated rx band guard and boundary tilt "
                           "envelope for pi0_fast_libero - new training-free mechanism (original RoboMetaHarness "
                           "method, run rtcfpp-20261101 round 1); not a reproduction of any cited paper",
                  "venue": "none (internal)"},
        "categories": {"primary": "execution_monitoring", "secondary": ["uncertainty_compute"]},
        "fidelity": "tf_variant" if tf else "faithful",
        "substitutions": (SUBSTITUTIONS[:1] + [SUBSTITUTIONS[1] + "; OR (saturation_arming on/yawctrl) a pinned-output / commanded-tilt arming route (first event, same 150-step window)"]
                          if sat != "off" else SUBSTITUTIONS) if tf else [],
        "requires": {"capabilities": ["act", "chunk"]},
        "budgets": {"policy_calls": None, "llm_calls": 0, "tool_calls": 0, "max_tokens": 0},
        "null": "derived",
        "artifacts": [{"name": "pi0_fast_libero_norm_stats",
                       "path": "/bigdata/sejain/hs2-cache/openpi/openpi-assets/checkpoints/pi0_fast_libero/assets/physical-intelligence/libero/norm_stats.json",
                       "sha256": NORM_SHA,
                       "license": "Apache-2.0 (openpi assets); byte-identical copies in hs_tilt_guard_v11/assets and hs_band_guard/assets"}],
        "operators": [],
        "development_only": dev_only,
        "role": role,
        "components": comps,
        "params": params,
        "disclosures": disclosures,
        "original": {"repo": "none (original method)", "commit": "",
                     "reported": "none; predictions in " + RUN + "/ideation/r1-domain.json (B on top of G: marginal +1..+3 on hard48) and r1-general.json (Z: hard48 net +2, range 0..+4)"},
        "description": f"tilt-guard {comps} for pi0_fast_libero ({role}); bitwise native whenever no component is active.",
        "tier1": {"instruction": "pick up the block and place it in the bowl", "horizon": 40, "chunk_len": 10},
        "changelog": ["v1.0 (2026-11-01): first implementation (task implement-guard-r1, coder a7156c5c79cdb4ce5) under "
                      "domain review C1-C14 / T1-T10 and general review C1-C10; frozen before any screen episode.",
                      "v1.1 (2026-11-01): new package hs_tilt_guard_v11 (task implement-tilt-guard-v11): adds saturation_arming "
                      "(r2 frozen_spec_p1, C1-C10, S1-S10) and the shadow saturation diagnostic; 'off' = v1.0 outputs."],
    }


def main() -> None:
    for v in VARIANTS:
        path = PKG / f"paper_{v}.yaml"
        text = "# GENERATED by scripts/make_cards.py (tilt-guard v1.1); edit the generator, not this file.\n"
        text += yaml.safe_dump(card(v), sort_keys=False, width=120, allow_unicode=False)
        path.write_text(text)
        print(path)


if __name__ == "__main__":
    main()
