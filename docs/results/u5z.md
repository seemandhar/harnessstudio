# U5+Z (new training-free harness, RTCF++ discovery run) in HarnessStudio v2

Status: **LIBERO-Long, run `runs/rtcfpp-20261101` with Campaign `runs/rtcfpp-campaign4-20261101` (2026-11-01/02).**
Package: `methods/u5z` (hs-u5z 1.0.0, entry point `u5z-t`, fingerprint `541af8ac…`; dev-only sign control `u5zsign-t`).
Components are imported read-only with sha-verified files: `methods/tube_steer` (T) and `methods/tilt_guard_v12` /
`tilt_guard_v11` (Z). Fidelity `tf_variant` (a new harness, not a paper port). Frozen policy: openpi pi0-FAST LIBERO
(`pi0_fast_libero`, served with z-score normalization, `quat_no_antipodal`, 5 executed actions per call, horizon 520).

> **WAIVER NOTICE (applies to every Campaign number below).** U5+Z did **not** pass its pre-registered admission route
> A7: rule 3 failed (3 losses > cap 2; task 0 lost twice), rule 4 failed (one credited win ended before a post-arming
> decision, so the physical witness could not be shown) and the determinism part of rule 5 failed (two scenes diverged
> from identity before activation through renderer branches). A7 had been declared the session's final route. The
> **user**, shown these facts, waived rules 3, 4 and the determinism part of 5 for `u5z-t` only (2026-11-02 01:21 UTC,
> `runs/rtcfpp-20261101/REQUEST.md` Amendment 3; scope confirmed by an independent critic,
> `runs/rtcfpp-20261101/reviews/a7-waiver-review.json`). This is a post-result override of a pre-registered rule by the
> user, not by any agent. U5+Z was also composed after its components' development outcomes were known.

## Outcome

On the 500-episode final test (libero_10 init 0-9 x env seeds 101-105), with the waiver above applying, U5+Z solved
**355/500 (71.0%, Wilson 95% [66.9, 74.8])** against a fresh identity baseline of 320/500 (64.0%, [59.7, 68.1]) and
the RTCF run's recorded 311/500 (62.2%, [57.9, 66.3]) on the same 500 cases.

- **The run's success criterion is NOT met (355 < 359), category "significant but small".** The criterion was
  >= 359/500 (+8 pp over the reproduced 319/500) AND two-sided exact McNemar p < 0.05 vs both the fresh baseline and
  the RTCF records (with net > 0, the pre-registered script's reading). Significance and direction pass; the threshold
  misses by 4 episodes.
- The Campaign's own gates passed (validation and full, alpha 0.05, min net 1), and an independent final critic
  approved the evidence (Campaign status `confirmed_improvement`, `runs/rtcfpp-20261101/reviews/final-review.json`).
  That is a Campaign-gate term for a significant gain over the fresh baseline and the compute-matched null; it does
  not mean the +8 pp criterion was met. Never a state-of-the-art claim.
- The final test has prior exposure: the RTCF run evaluated the same 500 cases, so it is a protocol-matched comparison,
  not an untouched test. The validation block (init 38-49) was unused before this Campaign.

## Final test (init 0-9 x seeds 101-105, 500 paired cases)

| Arm | Successes | Rate (Wilson 95%) | U5+Z wins / losses | Net | Two-sided exact McNemar p |
|---|---|---|---|---|---|
| U5+Z (`u5z-t`) | 355/500 | 71.0% [66.9, 74.8] | - | - | - |
| fresh identity baseline | 320/500 | 64.0% [59.7, 68.1] | 44 / 9 | +35 (+7.0 pp) | 1.2e-6 |
| RTCF recorded (`runs/rtcf-20261031`, fingerprint `a2b34bb7…`) | 311/500 | 62.2% [57.9, 66.3] | 78 / 34 | +44 (+8.8 pp) | 3.9e-5 |

Source: `runs/rtcfpp-20261101/eval/final-compare/result.json`, produced by the pre-registered comparison tool
`runs/rtcfpp-20261101/implementation/final-compare/final_compare.py` (sha256 `22b4ac50…`, frozen before any final
outcome was read; exact case-set equality enforced). The Campaign's paired p for the full gate (6.1e-7) is one-sided.
The fresh identity arm matches the RTCF run's recorded identity arm on 499/500 cases. The RTCF final-test table was
built from the RTCF run's traces (`eval/final-compare/rtcf_traces_to_jsonl.py`, sha `90431512…`; 500 unique valid
traces, obs0 equal to the full-phase records on 500/500, verified by the final critic).

## Validation (init 38-49, env seed 7, 120 scenes; out of sample)

U5+Z 81/120 (67.5%, [58.7, 75.2]) vs fresh baseline 72/120 (60.0%, [51.1, 68.3]): 10 wins / 1 loss, net +9,
**two-sided exact McNemar p = 0.0117** (the Campaign's one-sided paired p is 0.0059). The compute-matched null scored
72/120 and was bitwise equal to the baseline on all 120 cases (M 120/120), so the gain is not from extra inference;
the candidate's inference ratio was 0.91.

**RTCF on the validation cases (descriptive, predeclared):**
<!-- FILL:rtcf_validation -->
Frozen RTCF (`rtcf`, fingerprint `a2b34bb7…`, the RTCF run's development memory) on the same 120 validation cases,
all complete and valid with obs0 matching the Campaign records: **RTCF 72/120 (60.0%, [51.1, 68.3])**, equal to the
fresh baseline (RTCF vs baseline 11 wins / 11 losses, p = 1.0). U5+Z 81 vs RTCF 72: 19 wins / 10 losses, net +9,
two-sided exact McNemar **p = 0.136, not significant**. Descriptive only, not a Campaign gate; single realizations.
Source: `runs/rtcfpp-20261101/eval/rtcf-validation/summary.json` (sha256 `d4098d19…`).
<!-- /FILL:rtcf_validation -->

## Development evidence (in sample; not a promotion basis)

- Campaign development phase (Campaign-30: all 10 tasks x init 30-32, seed 7): 20/30 vs 18/30; null vs baseline
  M 30/30.
- Fresh-seed replication (pre-declared; env seeds 8/9 of init 30-37, 160 pairs): U5+Z 111/160 vs identity 100/160,
  14 wins / 3 losses, p = 0.013; T alone 105/160 (6/1, p = 0.125). Tasks 4 and 6 replay bit for bit across env
  seeds, so 4 of the 14 wins re-count seed-7 episodes (+9 over 144 when each bitwise scene counts once, p = 0.035,
  post hoc).
- **RTCF on the same 160 replication cases** (pre-declared, development-only): U5+Z 111 vs RTCF 101, 24 wins / 14
  losses, net +10, **p = 0.143 (not significant)**; RTCF vs identity 101 vs 100. On development cases U5+Z beats
  identity but was not shown to beat RTCF.
- About 20 arms were screened on the same 80 seed-7 development scenes (libero_10 init 30-37), so development nets are
  in-sample selection evidence.

## The method

U5+Z is one `action_op` that composes two frozen components on the native pi0-FAST chunk:

- **T (tube-steer, `methods/tube_steer`, called U5 in the run):** a per-instruction "success tube" of gripper
  orientation, built only from successful development episodes on reserved init states 10-29 (non-parametric memory,
  `assets/tube_memory.npz`). When public proprio features show the gripper drifting out of the tube early in an
  episode, T arms and steers the commanded rotation back toward the tube for a short, capped window, then hands back.
- **Z (zero-chunk substitution, from `methods/tilt_guard_v12`):** when greedy FAST decoding fails, openpi returns a
  zero-filled chunk that the delta-action policy executes as a 5-step open-gripper command with a large spurious
  rotation. Z detects that exact signature and executes the previous chunk's tail (or a hold) instead. Z acts only
  before T's first arming; after that only T acts.
- Order and precedence live in `hs_u5z/glue.py`; component files are sha-verified before import and parameters must
  equal the frozen component cards (no new scalars). Null: derived (runs the op, executes native actions).
- Training-free: no weights, probes or fitted parameters change; the bounded scalars (window, gain, thresholds) were
  chosen on development data. No runtime LLM, no SigLIP calls.
- Development attribution (trace-based): 9 of the 14 replication wins come from Z and 5 from T. In the 45 scenes where
  identity executed a confirmed zero chunk, identity solved 12/45, U5+Z 19/45 and T 13/45. Known losses: t0i37 (the zero
  chunk happened to release the can at the right moment; Z released 15 steps later and jammed) and s9 t9i34 (T
  straightened a mug that a tilted push would have inserted).

## Negative ideas from the same run (development screens, init 30-37 seed 7 unless noted)

Each package stays in `methods/` as a record; none is promoted. Details: `runs/rtcfpp-20261101/REPORT.md`.

- `band_guard` (rx saturation band guard + sign control): hard48 16/48 = identity (0 wins / 0 losses); fired at
  runaway onset, too late.
- `tilt_guard` G/B (v1.0, composite Z -> G -> B and controls): G+B hard48 16/48 = identity (0/0); late interventions
  (after tilt 30 deg) never flipped an outcome.
- `tilt_guard_v11` (saturation-armed Bsat flag) and `tilt_guard_v12` (B without G): v1.2 B and B-yaw hard48 16/48 =
  identity (0/0), push too weak; Bsat never screened. (`tilt_guard_v12`'s Z is the Z used in U5+Z.)
- `redecode` (re-query zero chunks with a +-1-bin state shift): hard48 +1/-0 vs identity, -1 vs its z control; not
  admitted.
- `vlm_rehome` (NRP qwen3-vl flags a post-boundary runaway, then re-home): VLM qualification failed (17/37 hard-success
  episodes falsely flagged); the controller is partial and unfrozen.
- `tube_steer` alone (T): hard48 +6/-0, all-80 +6/-0, but Campaign-30 net 0; not admitted as a unit (kept as U5+Z's T).
- `tube_steer_vfb` (T with a SigLIP visual-retrieval fallback for task 8): t8 0/8 for TV, TV-ortho and T; not
  admitted.
- `prr` / `prr_v11` (pose restore and re-decide): 1.0.0 restored the pose in 0/8 events; the 1.1.0 correction re-run
  was blocked because no CPU plant reproduced 1.0.0's position divergence (precondition P1 not met); no claim.
- `bcp` (boundary clause prompt): K1 shadow kill test blocked (1 of 40 cases failed a validity check); not scored;
  the K2 candidate entries never ran.
- M1 chunk self-consistency monitor: offline kill test failed (2/14 hits, 6/14 successes fire); no package built.

## Protocol

- Benchmark: vla-eval 0.8.0 LIBERO libero_10, official horizon 520, `--action-timeout 300`, single-episode containers
  with fail-fast resets (`--episodes-per-container 1 --fail-fast-reset`), `--streams-per-gpu 2`.
- Campaign `runs/rtcfpp-campaign4-20261101`, idea `u5z-a7`: phases scene-disjoint by (task, init state); development
  init 30-32 seed 7; validation init 38-49 seed 7 (120); full init 0-9 x seeds 101-105 (500) with recorded prior
  exposure; reserved memory scenes init 10-29. Null-gate tolerance 4 discordant / |net| 3 was set before any idea
  from measured renderer nondeterminism (X = 0 binding).
- Frozen closure: `runs/rtcfpp-campaign4-20261101/harnesses/25-rtcfpp-campaign4-20261101--u5z-a7/source/snapshot`
  (261 files; check with `python -m robometaharness verify --run rtcfpp-campaign4-20261101 --idea u5z-a7`).

## Limitations

- Admitted under a post-result user waiver of three pre-registered rules (notice above).
- The +8 pp success criterion is missed by 4 episodes.
- The final test has prior exposure via the RTCF run; only validation is untouched by selection.
- One suite (LIBERO-Long), one policy (pi0_fast_libero), one realization per case; greedy decoding branches on
  renderer-level image differences, also on a single GPU.
- U5+Z was not shown to beat a fresh RTCF arm: on the 160 development replication cases p = 0.143 and on the 120
  validation cases p = 0.136 (both descriptive); the significant final-test comparison with RTCF uses RTCF's recorded
  outcomes from the earlier run, not a fresh RTCF arm.
- Z's specificity against its sign control failed a seed-7 frame witness and was not re-tested; the Z/T split is a
  trace-based attribution, and U5+Z is tested as a harness against baseline and null, not as evidence for Z's
  mechanism.

## Large asset not in git

`methods/tube_steer_vfb/hs_tube_steer_vfb/assets/visual_memory.npz` (31,925,792 bytes, sha256
`ccebf9e0d3d80c709c116101835eea6c199fdac2a87fe7746cc2c98c732209a5`) is **not committed**: no `.npz` of that size has
been committed to this repository, and the RTCF development memory was likewise kept out of git. Its sha256 equals the
frozen RTCF development memory `ccebf9e0…` (docs/results/rtcf.md), i.e. it is a byte copy of that artifact; its
metadata `visual_memory.json` (sha256 `023e365e…`) is committed. `tube_steer_vfb` (a negative idea) needs the file
restored at that path before it can run; `u5z` and `tube_steer` do not use it.

## Costs

About 99 GPU-h of the run's 160 GPU-h cap before the RTCF validation arm (Campaign 173,205 GPU-s including the full
phase 105,383 GPU-s); NRP runtime calls 496 (VLM qualification only); paid LLM spend $0; Astra never used.
