# pcd-vgps: V-GPS value-guided selection over frozen PCD on OpenVLA-7b (SIMPLER, HarnessStudio v2) — valid negative

**Licence disclosure:** auxiliary critic released without a licence; used for research evaluation only (user
authorisation 2026-11-04 03:05Z, "do it, download it its fine. Start", `runs/pcd-discovery-20261104/TASK.md`
Amendment 1 and `QUESTIONS.md` Q4). The V-GPS code and weights are **not** in this repository and must never be
committed, vendored or pushed. Our method package and critic-service adapter also stay in the run
(`runs/pcd-discovery-20261104/harnesses/cand2-vgps/`, `harnesses/vgps-service/`).

Status (2026-11-04, run `runs/pcd-discovery-20261104`): **done, negative**. The development screen was a **loss** vs PCD
(34 vs 36, W10/L12); the candidate did not reach validation, and the final block was not opened (decision d07).

Sources. Paths starting with `runs/` are relative to `/home/sejain/research/cvpr27/RoboMetaHarness-claude/`; other run
paths are relative to `runs/pcd-discovery-20261104/`.
- `eval/vgps/FROZEN.json` (sha256 `7856b524...`), `eval/vgps/e1.json` (`ecd219bc...`), `eval/vgps/r5_pilot.json`
  (`3f5fa490...`), `eval/vgps/r5_attempt1_infra.json` (`a2bdee5d...`);
- dev: `eval/vgps/dev_gate.json` (`4202e8fa...`), `eval/vgps/dev_assessment.json` (`533ff6fa...`),
  `analysis/vgps/dev_analysis.md` (`ab7fa5f8...`, dev only);
- independent recomputation of the dev screen from raw per-step records: `reports/critic-final.review.json` (exact match).
- reviews and builds: `reports/critic-proposal-vgps.review.json`, `reports/critic-preflight-vgps.review.json`,
  `reports/coder-vgps-feasibility.report.json`, `reports/coder-vgps-service.report.json`,
  `reports/coder-vgps-method.report.json`, `reports/coder-vgps-n1.report.json`, `reports/eval-block-a.report.json`;
- cost: `eval/FINAL_RECON.json` (`7c85234a...`); closure `control/research/decisions/d07-no-admissible-validation.json`.

## Outcome

| block | pairs | baseline | PCD | **pcd-vgps** | null | W / L vs PCD | exact McNemar p | Newcombe 95% CI (cand - PCD) | gate |
|---|---|---|---|---|---|---|---|---|---|
| development (exposed seeds; reused PCD/baseline records) | 64/64 | 22 | 36 | **34** | 36 (M 64/64) | 10 / 12 | 0.832 | [-17.0, +10.9] pp, point -3.1 | **FAIL / loss** (net -2; W < L) |
| validation | not run (d05: slot 2 only after a dev PASS) | | | | | | | | |
| final | not opened | | | | | | | | |

| dev, descriptive | baseline | PCD | pcd-vgps | W / L vs PCD |
|---|---|---|---|---|
| Pick Coke Can any-step (32) | 5 | 12 | 15 | 8 / 5 |
| Close Drawer any-step (32) | 17 | 24 | 19 | 2 / 7 |
| pooled max-step (64) | 22 | 33 | 34 | 11 / 10 (p 1.0) |
| vs baseline, any-step | | | 34 vs 22 | 19 / 7 (p 0.029) |

The dev seeds were exposed in the PCD port. The prediction recorded before any outcome was net -4..+5, expected ~+1,
P(dev gate) ~0.2 (`reports/sci-r4-F.report.json`).

## Why it lost (development analysis, `analysis/vgps/dev_analysis.md`)

- **Untargeted perturbation.** The reranker changes dims 0-5 only (the gripper never), fires on 59% (coke) and 64%
  (drawer) of steps, and every episode diverges at step 1.
- **Coke W8/L5:** persistent perturbation moves PCD out of its fixed points and shifts the policy's own close timing; the
  wins overlap with pcd-gphase's dev wins on only 3 seeds; the losses look random.
- **Drawer W2/L7:** larger cumulative drift (|x| up to 2.9 raw) derails precise pushes; lost successes were later PCD
  successes with larger pre-success perturbation.
- **Q barely separates the candidates:** per-step spread over K = 10 is 2.6-6.2% of |Q|; step-0 Q AUC ~0.5; the
  episode-mean Q tracks progress only. On these tasks the Cal-QL reranker acts close to random noise.

## Readiness evidence (before the screen)

**E1 tier 2** on smoke seeds coke 111 and drawer 98 (`eval/vgps/e1.json`, PASS):

| check | coke 111 | drawer 98 |
|---|---|---|
| null vs PCD | M (and M on the rerun) | M (and M) |
| forced-index-0 seam (test only, critic called every step) | bitwise PCD | bitwise PCD |
| rerun after a critic-service restart | identical actions, obs, selections, Q hex | identical |
| selection audit; fired steps | OK; 62/80 | OK; 42/113 |
| s/step PCD / null / candidate | 2.686 / 2.819 / 2.807 | 2.771 / 2.852 / 3.174 |

Service parity: bitwise Q against the authors' `load_vgps_checkpoint` + `get_vgps_action` on fixed public inputs; a
second instance was parity-identical.

**R5 rescue pilot** (predeclared; `eval/vgps/r5_pilot.json`): failures coke 6, 23, drawer 19, 27; controls coke 7, drawer
16. Null M 6/6; all divergences at logged selections; fired on 61-92% of steps; official rescue coke 23 (the record's own
vla-eval success video is referenced in `eval/FINAL_RECON.json`, sha256 `4365c809...`; not re-rendered); both controls
still succeed; gate PASS mechanically. The 6 pilot records are screen pairs (not rerun).

## Method

Entry `hs_pcd_vgps:pcd_vgps`, fingerprint `d64cb1efd4dfdc79045005edb3ed620cbafc845f5c7bc41d457cbf0cbee97140`, frozen
2026-11-04 04:57:54Z before any candidate or null episode. Params: K 10, beta 0, temperature 1.0, top-k 50, Philox key 0,
counter `[0, env_step, 0, 0]`, sampled dims 6, critic timeout 30 s, pins checked, `"null": custom`.

Every step, on both tasks (no gate):
1. Frozen PCD runs exactly as in the port ([pcd.md](pcd.md)).
2. Candidate 0 = PCD's 7 tokens. Candidates 1..10: per-dimension inverse-CDF draws in dims 0-5 from
   `softmax(final[d, action slice] / 1.0)` restricted to the 50 largest logits; dim 6 = PCD's gripper token.
3. One call to the frozen V-GPS critic service with the fixed ordered batch of 11 actions on the real frame and the public
   instruction; Q = min over the 2-member target ensemble.
4. Execute `argmax Q` (first index wins, so PCD wins exact ties); a selection equal to PCD collapses to PCD.
5. Any critic error invalidates the episode (infrastructure); there is never a PCD fallback.
6. Null: executes PCD's action on all 7 dims while paying all 60 draws and the 11-action critic call every step (identical
   batches in both arms, because batch composition changes Q by up to ~1.9e-5).

Critic pins (runtime-checked each episode): repo `nakamotoo/V-GPS@912a290f95a9f5fcf5942fbb5b9e7b4e99905c7b`,
`checkpoint_500000` (`fa732622...`), critic target params `8f4afa46...`, MUSE saved_model `3d03fd03...`, venv freeze
`3366ed41...`; service pins `harnesses/vgps-service/pins.json` (`baa03fde...`). Code and weights live under
`/bigdata/sejain/frozen-harness-bench/models/pcd-discovery-20261104/vgps`, outside this repository and git.

## Fidelity: **tf_variant** of the adapted PCD reproduction; V-GPS (arXiv:2410.13816v2) used as a component

Substitutions (card):
1. V-GPS's K autoregressive samples from the base policy -> PCD's tokens plus K independent per-dimension draws from
   PCD's contrasted rows, each conditioned on its own stream's greedy prefix (joint actions may be rarer than
   autoregressive samples).
2. transformers top-k over the full vocabulary -> top-k 50 within the 256-token action slice; stable tie order; float64
   inverse CDF over Philox uniforms (one disjoint stream per env step).
3. Paper policy `openvla-v01-7b` -> `openvla-7b` (the frozen PCD port's policy).
4. Paper critic (Cal-QL 1M steps, batch 512) -> released `checkpoint_500000` (batch 256), frozen, CPU JAX service.
5. Paper's swept K x beta (best of 6) -> fixed K 10, beta 0, no margin.
6. Sampled gripper -> PCD's gripper token in every candidate.
7. Unseeded SimplerEnv -> HarnessStudio simpler_seeded (inherited).
8. Accelerator critic -> CPU JAX critic service (bitwise-checked, fixed batch of 11).
9. Inherited from the adapted PCD reproduction: float32 action wire; HarnessStudio transport; env identity (vla-eval
   SIMPLER image); SAM2 `_C` unresolved.
10. Null: PCD's shadow null -> custom null executing PCD's combined action while paying perception, both decodes, all
    draws and the critic call.

This is **not** a reproduction of the V-GPS paper. Training-free: the critic and MUSE are frozen released models used for
inference only; 0 training steps; constants from the paper, the repo and transformers defaults, fixed before any outcome.

## Disclosures (bound by the run's critics)

- **Licence:** auxiliary critic released without a licence; used for research evaluation only. MUSE
  (universal-sentence-encoder-multilingual/3) is Apache-2.0 per its TF Hub listing (not re-fetched). Critic conditions
  L1-L3 (`reports/critic-proposal-vgps.review.json`).
- T 1.0 and top-k 50 are transformers defaults (openvla-7b's generation config sets neither), not paper values; K and beta
  are fixed from a swept paper setting, and the paper's best-of-6 is not reproduced (finding V2).
- The critic action rescale uses the authors' hard-coded fractal min/max (p001/p999 ambiguity disclosed); the critic sees
  only the real public frame, never the counterfactual.
- The paper's OpenVLA SIMPLER gains are unpaired, n = 50, best of 6 settings, on `openvla-v01-7b`, and include no drawer
  task.
- N1 was a pre-exposure design fix (the old Philox counter reused 56 of 60 draws between consecutive steps).
- Deviation D1: `_client` is overridden only to record `ctx.embodiment.control_mode`; accepted by critic-preflight-vgps.
- Scientist F read the PCD dev failure analysis and the feasibility obs0 Q probe; no constant was tuned on outcomes.
- NRP and any runtime LLM: not used.
- Build timebox (Amendment 2, 06:30Z, "11 hours of coding, no."): ready for GPU checks at 03:32Z, scope cut, gates kept.

## Infrastructure events

- **R5 attempt 1 (05:49Z):** all 6 null calls invalid twice on the frozen 30 s act timeout at env step 0 (host load 193.5
  on 96 cores, other users' jobs untouched; critic 1.0-5.8 s per call vs 0.5 s in E1; XLA/TF pools sized to all 96
  cores). Three seeds passed on their bounded retry; three were relaunched staggered = **V-GPS technical repair 1 of 2**;
  method, fingerprint, service config and timeout unchanged; no valid record rerun (`eval/vgps/r5_attempt1_infra.json`).
- A CPU rule with the parallel STAR run (`runs/scaleagent-shared/COORDINATION.md`, 2026-11-04T06:08:11Z) limited STAR's
  CPU use during the screen. Dev screen: 0 invalid attempts (`eval/vgps/dev_assessment.json` `infrastructure`).

## Cost

GPU-hours from lease acquisition to release (T11):

| block | GPU-h | source |
|---|---|---|
| E1 tier 2 | 0.716 (2,576.6 GPU-s) | `reports/eval-block-a.report.json` |
| R5 pilot incl. attempt 1 and the repair | 2.967 (lease-ledger segment vgps-r5, T11; eval-block-a's per-step accounting gives 10,503.2 GPU-s = 2.918) | `eval/cost/lease_ledger.jsonl`, `reports/eval-block-a.report.json` |
| dev screen (candidate + null, 64 seeds) | 5.608 | `eval/FINAL_RECON.json` |

- 2.405 s/step mean in the dev screen; CPU critic K = 10 round trip 0.32 s unloaded, peak RSS 8.0 GB; 3.8 GB download;
  0 extra policy decodes, 0 LLM calls, $0.

## Lesson

Proposed (unreviewed) lesson `20261104T103318Z-calql-critic-reranking-around-a-frozen`: do not retry critic reranking on
these tasks without a CPU Q-discrimination check (Q spread vs |Q|, Q-outcome rank agreement on recorded pairs) first
(`reports/analyst-vgps-dev.report.json`). Sibling negative: [pcd-gphase.md](pcd-gphase.md).
