# pcd-gphase: PCD + grasp-phase completion on frozen OpenVLA-7b (SIMPLER, HarnessStudio v2) — valid negative

Status (2026-11-04, run `runs/pcd-discovery-20261104`): **done, negative**. The development screen PASSED (W7/L0) and the
fresh validation block FAILED its predeclared gate (W4/L1, net +3 < 6, p 0.375 >= 0.025). The final block was **not
opened** (decision d07). The method package is **not** in this repository: it lives in the run at
`runs/pcd-discovery-20261104/harnesses/cand1-gphase/` (with its base `harnesses/pcdwrap-base/`). Nothing here claims an
improvement over the reproduced PCD.

Every number below comes from one of these files. Paths starting with `runs/` are relative to
`/home/sejain/research/cvpr27/RoboMetaHarness-claude/`; other run paths are relative to `runs/pcd-discovery-20261104/`.
- `eval/cand1/FROZEN.json` (sha256 `e271c5a9...`);
- development: `eval/cand1/dev_gate.json` (`df6b4e09...`), `eval/cand1/dev_assessment.json` (`e3cf762f...`), and the
  critic's independent recomputation from 256 raw records, `reports/critic-dev-cand1.review.json` (`313d3f05...`);
- validation (aggregate fields only): `eval/cand1/validation_gate.json` (`33391909...`),
  `eval/cand1/validation_assessment.json` (`403e04eb...`), and the critic's independent recomputation,
  `reports/critic-val-cand1.review.json` (`5b0a151f...`; CONFIRMED FAIL);
- readiness: `eval/cand1/tier2.json`, `eval/cand1/pilot/pilot.json`, `eval/cand1/pilot/label_verdicts.json`,
  `eval/cand1/relabel_verdicts.json`, `eval/cand1/c1_calibration.json`, `eval/cand1/c6_custody.json`;
- dev mechanism: `analysis/cand1/dev_analysis.md` (`bdbd9889...`, dev only);
- cost: `eval/FINAL_RECON.json` (`7c85234a...`);
- closure: `control/research/decisions/d07-no-admissible-validation.json` (`ea8b51b2...`).

## Outcome

| block | pairs | baseline | PCD | **pcd-gphase** | null | W / L vs PCD | exact McNemar p | Newcombe 95% CI (cand - PCD) | gate |
|---|---|---|---|---|---|---|---|---|---|
| development (exposed seeds; reused PCD/baseline records) | 64/64 | 22 | 36 | **43** | 36 (M 64/64) | 7 / 0 | 0.015625 | [+3.3, +18.4] pp, point +10.9 | **PASS** (net +7 >= 4) |
| validation (fresh, 4 fresh arms per seed; alpha 0.025, m = 2) | 64/64 | 22 | 31 | **34** | 31 (M 64/64) | 4 / 1 | **0.375** | [-1.9, +11.2] pp, point +4.7 | **FAIL** (net +3 < 6; p >= 0.025) |
| final | not opened | | | | | | | | no validation PASS (T7) |

- **Validation checks** (`eval/cand1/validation_gate.json` via `validation_assessment.json` `gate.checks`): complete and
  comparable yes; null clean yes; W > L yes; net +3 (needs >= 6) **no**; p 0.375 (needs < 0.025) **no**; candidate >=
  baseline yes (34 vs 22, W17/L5, p 0.0169).
- **Dev to validation:** W7/L0 (+10.9 pp) on exposed seeds with dev-informed constants shrank to W4/L1 (+4.7 pp, not
  significant) on fresh seeds: the winner's curse and optimism the critic predicted
  (`validation_assessment.json` `dev_vs_validation`).
- **Comparisons are paired against PCD re-run on the same seeds**, never against the 56.3% dev number.
- **Scope:** 2 SIMPLER tasks (Pick Coke Can, Close Drawer), OpenVLA-7b, the authors' seeded protocol, any-step success
  (max-step recorded). Never state of the art.

### Per task and max-step (descriptive)

| | dev baseline | dev PCD | dev cand | val baseline | val PCD | val cand |
|---|---|---|---|---|---|---|
| Pick Coke Can any-step (32) | 5 | 12 | **19** (W7 L0) | 8 | 15 | **18** (W4 L1) |
| Close Drawer any-step (32) | 17 | 24 | 24 (identical) | 14 | 16 | 16 (identical) |
| pooled max-step (64) | 22 | 33 | 38 (W5 L0, p 0.0625) | 22 | 29 | 34 (W5 L0, p 0.0625, +7.8 pp [+1.3, +14.1]) |

Max-step is descriptive. The drawer arm is bitwise PCD by the class gate in both blocks, so every discordant pair is a
coke pair. Sources: `per_task_descriptive` and `max_step_descriptive` in the two assessments.

## Method

Card `runs/pcd-discovery-20261104/harnesses/cand1-gphase/source/hs_pcd_gphase/paper.yaml`; entry
`hs_pcd_gphase:pcd_gphase_l3`, fingerprint `408412ec6163cff92d3885527199489610bd59144c2b7a2065663f81e8c50b4e`; params
`lift_steps 3`, `release_enabled true`, `return_masks true`, `max_extra_decodes_per_step 0`, `"null": custom`. Base: the
pcdwrap controller (FIX1 pins) around the unchanged frozen `pcd` (fingerprint `8350a143...`, [pcd.md](pcd.md)).

Per step the frozen PCD runs exactly as in the port. On Pick Coke Can only (PCD parser targets exactly `["coke can"]`,
plus a step-0 single-dominant-component target witness), the controller may rewrite PCD's 7 native tokens:
- **PG (proximity-gated close):** a would-be close with near-band pixel change nb < 0.02 around the can is sent as OPEN;
  the gate releases at nb >= 0.10.
- **WES (window-end sync):** at the last step of an active 15-step sticky window, the replayed physical gripper value is
  sent, so a swallowed reversal is re-armed.
- **LIFT:** after a closed hand stalls at the target (closed >= 5 steps, 8 executed zero-translation steps, nb >= 0.02,
  valid mask, t <= 74, at most one cycle), 3 scripted bin-table +z steps (nearest +0.02 m) with CLOSE; then a
  bottom-edge custody check on frozen SAM2 masks; custody or ambiguous -> hold; no custody -> one native release.
- **Drawer and any other instruction:** bitwise PCD.
- **Null:** executes PCD's action on all 7 dims every step while the whole path runs in shadow.
- **Cost:** 0 extra decodes, 0 LLM calls; masks come from the existing PCD perception session.
- Every constant is traced with a verbatim quote in `harnesses/cand1-gphase/constants.json` (`36788c33...`).

Origin: scientist D's `pcd-gphase-v1` merged with scientist C's `pcd-gpfp-v1` bottom-edge custody
(`ideation/round2/`, `analysis/premises/premises-r2.md`); proposal `ideation/candidates/cand1-pcd-gphase.json`;
reviews `reports/critic-proposal-cand1.review.json`, `reports/critic-cand1-constants.review.json`.

## Fidelity: **tf_variant** of the adapted PCD reproduction

Substitutions (card):
1. PCD's gripper token executed verbatim -> proximity-gated and window-end-synchronized native gripper token from the
   harness's own-action sticky replay plus a public-pixel target-region witness (pick only).
2. PCD's action on a closed-hand at-target stall -> a 3-step scripted bin-table lift (dim 2 nearest +0.02 m, other dims
   zero bins, CLOSE).
3. After a lift with custody or an ambiguous verdict -> hold (zero bins + CLOSE where PCD's dims 0-2 are the zero triple).
4. After a no-custody lift -> one scripted native OPEN release, then PCD again.
5. None on drawer or any instruction whose targets are not `["coke can"]` (bitwise PCD).
6. Inherited from the adapted PCD reproduction: float32 action wire; HarnessStudio transport; env identity (vla-eval
   SIMPLER image); SAM2 `_C` unresolved.

Training-free: frozen policy and perception weights, 0 trainable parameters, 0 training steps; scalars single-point and
frozen at 03:09:42Z before any candidate episode (`eval/cand1/FROZEN.json`).

## Disclosures (bound by the run's critics)

- **D1 custody:** no claim of held lifts beyond the blind evidence. The first blind pass had 0/5 lift-custody packs
  decidable; a reference-line relabel by a new blind labeller gave 3 yes (dev coke 6, 27, 24) / 2 unclear (coke 15, 7;
  post-success lifts) / 0 no. Dev wins 8, 5 and 19 are "official success without blind-validated custody". Validation
  custody was not blind-labelled (verdicts 13 custody / 3 no custody, unvalidated).
- **D2 labeller notes** (notes, not labels): coke 6 "its position barely changes across frames"; coke 24 "the can stays
  essentially in place across frames".
- **D3:** the falsifier "custody verdicts contradict blind labels in > 1 fired episode" is tested on the 5 pilot packs
  (0 contradictions) and untested on screen and validation episodes.
- **D4:** physical effect demonstrated for PG (pilot coke 27, 15 agree; coke 23 hand-empty agree) and for LIFT on dev
  coke 6 and 27 via the relabel.
- **Dev-informed constants:** PG thresholds 0.02/0.10 and zero-run 8 were chosen with sight of the exposed PCD dev traces
  (critic K5/C8); the dev seeds were exposed in the PCD port and the pilot seeds chosen by predicted mechanism.
- **Transient wins (dev):** coke 5 and 19 succeed at some step but not at the last (release and re-close path).
- **Design-time benchmark reads:** the sticky-gripper rule (`simpler_seeded.py` 31d45dd7) and a cached copy of the coke
  success predicate (c067aa46) used to size the lift; LIFT is a metric-aware scripted lift. No runtime success, state,
  reward or privileged input.
- **Action-interface effect:** PG/WES effects are specific to the protocol's relative 15-step sticky gripper.
- **C6 custody qualification:** seed 25 at exactly 5.0 px (zero margin); the C6 labeller (the evaluator) knew the 3/3
  split; thresholds were designed on mp4 frames and a red-pixel proxy, live decisions use raw frames and SAM2 masks.
- The 7 pilot records were used as dev screen pairs and not rerun (critic X3). Dev pairs are against reused port
  records from a different server instance (C-REPRO 12/12 bitwise, `eval/reuse_proof.json`); validation ran 4 fresh arms.
- All effects come from Pick Coke Can; the drawer arm is bitwise PCD.

## Readiness and controls

| check | result | source |
|---|---|---|
| pcdwrap base tier 2 | FAILED first (BinTable kept the server's float64 table vs PCD's float32 cast); FIX1; re-run PASS 4/4 bitwise | `reports/eval-pcdwrap-tier2.report.json`, `reports/eval-pcdwrap-tier2b.report.json` |
| C1 lift steps | n = 3 under every reading (centroid 14.84 px at n = 3) on smoke seed 111 | `eval/cand1/c1_calibration.json` |
| C5 replay of the merged trigger | PASS 12/12 configurations | `reports/analyst-cand1-c5.report.json` |
| C6 custody on real masks | PASS: custody 7, 9, 25; no custody 2, 22, 14; blind labels 6/6 | `eval/cand1/c6_custody.json` |
| `hs verify` tier 0 / tier 1 | tier 0 PASS x4; tier 1 only the structural custom-null line; 31 unit tests, 3 sabotages detected | `harnesses/cand1-gphase/README.md` |
| tier 2 (GPU) | coke null M with divergence at a logged activation; drawer bitwise PCD | `eval/cand1/tier2.json` |
| pilot P1 | null M 7/7; controls bitwise PCD to first success; label rules (b), (d) PASS, (c) not fired | `eval/cand1/pilot/pilot.json`, `reports/critic-cand1-pilot-labels.report.json` |
| null attribution | dev M 64/64 vs reused PCD; validation M 64/64 vs fresh PCD (actions and obs, critic) | the assessments, `reports/critic-val-cand1.review.json` |
| first-divergence audit | dev 26/26 and validation 26/26 changed seeds diverge at a logged override; 38 bitwise PCD in each block | `reports/critic-dev-cand1.review.json`, `validation_assessment.json` `mechanism` |
| validity | dev: one bounded retry (drawer candidate, step-0 act timeout); validation: one bounded null retry (step-0 act timeout) and a 06:45Z launch failure with 0 episodes; no completed outcome rerun | assessments `infrastructure`, `reports/critic-val-cand1.review.json` |

## Activation and mechanism

- **Dev** (`eval/cand1/dev_assessment.json`, `analysis/cand1/dev_analysis.md`): executed overrides LIFT 71, PG 55, WES 5;
  custody 14 / no custody 6; wins by first-diverging component PG 4 (5, 16, 27, 28), LIFT 3 (6, 8, 19); 0 losses.
- **Validation** (aggregates, `validation_assessment.json` `mechanism`): executed overrides LIFT 51, PG 16, WES 9; null
  would-fire LIFT 55, PG 7, WES 10; custody 13 / no custody 3; 0 determinism faults; 0 drawer pairs differ; the 4 wins
  first diverge at LIFT (3) and PG (1); the 1 loss first diverges at WES.
  Disclosure (T7b): these validation mechanism aggregates are published after the validation block closed. They expose the held-out block's mechanism profile, so no future candidate designed with sight of them may use this run's validation block.

## Cost

GPU-hours from lease acquisition to release, idle holds included (T11; `eval/FINAL_RECON.json`):

| block | GPU-h |
|---|---|
| C1/C6 calibration | 0.121 |
| dev (tier 2, pilot, frames, 64-seed candidate + null screen) | 6.112 |
| relabel packs (part of eval-block-a, 256.7 GPU-s) | 0.071 |
| validation (4 fresh arms x 64 seeds) | 9.844 |

- Per step (dev): 2.44 s coke, 2.71 s drawer; null matched (equal policy samples and tool calls); 0 extra decodes; 0 LLM
  calls; $0.

## Lessons and open items

- The validation negative stands; cand1 may not open the final block and no changed cand1 may be re-tested on this
  validation block (`reports/critic-val-cand1.review.json` rulings).
- Dev-only directions (not tested): PG never releases when nb stays 0; re-grasp retention
  (`reports/analyst-cand1-dev.report.json`).
- Composite cand1 + E1 (drawer forward floor) was frozen but **not admitted to its screen**: the predeclared seed-42 sign
  calibration was not confirmed by its blind label (`reports/critic-cand3-calib.review.json` `99f2aa22...`; d07).
- Run-level report: `runs/pcd-discovery-20261104/REPORT.md`. Sibling negative: [pcd-vgps.md](pcd-vgps.md).
