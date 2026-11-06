# PCD (Policy Contrastive Decoding, arXiv:2505.13255) on OpenVLA through HarnessStudio v2

Status (2026-11-03, filled after all GPU work stopped at 22:35:21Z): **done**.
- Setting A primary complete (64/64 pairs).
- Move Near and Setting B complete (descriptive).
- Both gated extensions verified on GPU and **merged into harnessstudio `main`** (2026-11-03, run task conv-merge):
  merge `4f93662` (EXT openvla-contrast, `adc271a`), merge `ecc9b68` (EXT simpler-seeded, `9b2b714`), docs-only
  `0092631`; scored runs used the pinned worktree commits (decision P-4). Merge identity carries one accepted
  docstring-only exception (see "Scoring source").

Every number below comes from one of these sources (all under `runs/pcd-port-20261103/reports/`):
- `primary.md` (sha256 `6b72163a5f733dcd...`);
- `primary_addendum.md` (`5bc97bdde62654e9...`);
- `optional_runs.md` (`dab7e387ffe1441f...`), as corrected by `optional_runs_addendum.md` (critic-optional R1-R4;
  where they disagree the addendum wins);
- `gpu_phase1.md` (`12f99275e199ae3d...`);
- the critic's independent recomputation from the raw per-episode files, `critic-primary.review.json`
  (`d27b1adb219ff7bf...`; verdict `label_stands`).

Two sections draw on further run files:
- the plan, on `runs/pcd-port-20261103/SETTING.md`;
- the pre-GPU CPU validation, on the files cited in its table.

Paths starting with `runs/` are relative to `/home/sejain/research/cvpr27/RoboMetaHarness-claude/`.

## Outcome

**Setting A primary**, on the authors' seeded SIMPLER protocol with OpenVLA-7B base. Every claim carries this scope:
**"subset only: 2 of 9 tasks, chosen for the largest reported gains; 64 of 2700 paired trials".**

| | value |
|---|---|
| pairs | **64 / 64** complete: no invalid record, no infrastructure retry, no timeout, no X |
| pooled any-step success | baseline **22/64** (34.4%) -> PCD **36/64** (56.3%); null 22/64 |
| W / L (PCD-only success / baseline-only success) | **17 / 3** |
| exact two-sided McNemar p | **0.00258** (critic: 0.0025768) |
| 95% CI of (W - L)/n, Newcombe hybrid score (uncorrected phi) | **[+0.090, +0.335]**, point **+0.219**. It contains the paper's pooled **+0.266** |
| sensitivity: corrected phi (reported only, not the decision) | [+0.086, +0.338] |
| **label** (SETTING v2 edit 2) | **"gain reproduced in direction; magnitude compatible"** |
| null attribution | **M 64 / R 0 / X 0**; null actions bitwise equal to the baseline in 64/64 |
| mechanism-inactive episodes (E-4) | PCD 0/64, null 0/64 |
| **Setting A fidelity label** (frozen card, `paper.yaml` `e1432d30`, decided at D1 before any primary episode) | **adapted** (substitutions: float32 action wire [B7 rule violated]; HarnessStudio transport; env identity [vla-eval SIMPLER image vs the authors' vendored copy]; unresolved: SAM2 _C) |

**What the label means.**
- **"Magnitude compatible":** the paper's gain is not excluded by our interval, which is wide, [+9.0, +33.5] pp.
- **The point estimate is lower:** +21.9 pp is below the paper's +26.6. That value is the mean of the per-task gains;
  the critic reports the two-task gain from the paper's figures as +26.7 pp, and both lie inside the CI.
- **Scope:** this is not a reproduction of the 9-task 16.8 -> 25.3, and it is never compared with or extrapolated to
  that number.

**Optional runs** (descriptive; they promote nothing):

| run | label | pairs | baseline -> PCD | W / L | exact McNemar p | Newcombe hybrid (uncorrected phi) 95% CI | null |
|---|---|---|---|---|---|---|---|
| Move Near (SIMPLER, `pcd`) | **predicted no-gain task** | 32/32 | 19 (59.4%) -> 21 (65.6%) | 4 / 2 | 0.6875 | [-0.083, +0.203] | M 32/0/0 |
| Setting B (libero_goal, `pcd-libero-goal`) | **adapted / tf_variant, transfer screen only** (no paper row; never "reproduced") | 30/30 | 23 (76.7%) -> 23 (76.7%) | 3 / 3 | 1.0 | [-0.162, +0.162] | M 30/0/0 |

## Setting A per-task rows

These rows are **descriptive**. The per-task p and CI are uncorrected, and no per-task claim is made: the decision is
the pooled test.

### Any-step success (paper Table 1, v3+, n = 300 per task)

| task | n | baseline | PCD | exact CP 95% (baseline; PCD) | W / L | p | Newcombe CI | paper Base -> PCD-GDINO |
|---|---|---|---|---|---|---|---|---|
| Pick Coke Can | 32 | 5 (15.6%) | 12 (37.5%) | [0.053, 0.328]; [0.211, 0.563] | 9 / 2 | 0.065 | [+0.024, +0.396] | 18.0% -> 45.3% (+27.3 pp) |
| Close Drawer | 32 | 17 (53.1%) | 24 (75.0%) | [0.347, 0.709]; [0.566, 0.885] | 8 / 1 | 0.039 | [+0.048, +0.371] | 47.3% -> 73.3% (+26.0 pp) |

**Baseline consistency:** both baseline CIs contain the paper's baselines, 18.0% and 47.3%. There is therefore no
"baseline-mismatched stack" qualifier.

**Close Drawer note (Addendum 3):** the authors' `dresser` alias masks the **whole cabinet** (22-26% of the frame in
our episodes). The drawer contrast is therefore against a cabinet-free frame, not a named-drawer mask. This is the
authors' code as written.

### Max-step success (paper Table 9, same runs, n = 300 per task)

| task | baseline | PCD | exact CP 95% (PCD) | W / L | p | Newcombe CI | paper Base -> PCD-GDINO |
|---|---|---|---|---|---|---|---|
| Pick Coke Can | 5 (15.6%) | 11 (34.4%) | [0.186, 0.532] | 8 / 2 | 0.109 | [+0.000, +0.362] | 17.3% -> 39.7% |
| Close Drawer | 17 (53.1%) | 22 (68.8%) | [0.500, 0.839] | 6 / 1 | 0.125 | [+0.004, +0.296] | 47.3% -> 72.7% |
| pooled (64) | 22 | 33 | - | 14 / 3 | 0.0127 | [+0.051, +0.283] | two-task +23.8 pp (critic A1) |

- All the paper values lie inside our intervals (critic A1).
- Success is per step, not latched. PCD is successful at some step but not at the last in 3 episodes (coke seed 1,
  drawer seeds 17 and 35). That is why max-step 33 < any-step 36.

**Move Near (descriptive; `optional_runs_addendum.md` R2):**

| | baseline max-step | PCD max-step | difference |
|---|---|---|---|
| paper, Table 9 | 47.3% | 41.3% | -6.0 pp (a regression) |
| ours (n = 32) | 14/32 = 43.8%, exact CP 95% [26.4, 62.3]% | 18/32 = 56.3%, exact CP 95% [37.7, 73.6]% | +12.5 pp (the **opposite direction**) |

- Each of our arm-level exact CIs contains the paper's value for that arm. The critic's paired max-step interval
  [-0.040, +0.277] excludes -6.0 pp. This is a descriptive difference in direction on a small sample, not a test.
- Any-step: 19 -> 21, (W - L)/n = 2/32 = **+6.25 pp** exactly; baseline 59.4% is consistent with the paper's 58.7%,
  and the paper's any-step gain, +0.3 pp, lies inside our interval.

## Null attribution
- **Primary:** `hs pair`, all 64 pairs: **M 64 / R 0 / X 0**. In 64/64, the null's executed action sequences and
  observation sequences are bitwise equal to the baseline's, and so are its successes (any-step and last-step).
- **Order and gates:** each seed ran baseline -> null -> PCD as three separate arm calls on one policy server per GPU.
  64/64 seed gates passed.
- **No server restart:** each shard's server action log is contiguous (seq 1..9265 = 1 probe + 3,088 native + 6,176
  logits requests).
- **Optional runs:** Move Near M 32/0/0; Setting B M 30/0/0, with null bitwise equal to the baseline in 30/30.

## Activation (Setting A primary; `primary_addendum.md` RE2, `eval/primary/activation_table-RE2.json` `d8c74926...`)

| | PCD coke | null coke | PCD drawer | null drawer |
|---|---|---|---|---|
| episodes | 32 | 32 | 32 | 32 |
| **mechanism inactive (E-4)** | **0/32** | **0/32** | **0/32** | **0/32** |
| steps | 2560 | 2560 | 3616 | 3616 |
| target-active steps | 2560 (100%) | 2560 (100%) | 3616 (100%) | 3616 (100%) |
| mean mask area (fraction of frame) | coke can 0.0118 | 0.0123 | top 0.222 / middle 0.245 / bottom 0.258 | 0.226 / 0.254 / 0.265 |
| first-detection step (median / max) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| robot-only first detection | 0 | 0 | 0 | 0 |
| targets never masked (episodes) | 0 | 0 | 0 | 0 |
| tracker None steps | 0 | 0 | 0 | 0 |
| counterfactual frame differs | 2560 | 2560 | 3616 | 3616 |
| contrast skipped / stream-A fallback | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| steps with a changed token | 1595 | 2189 | 2547 | 3032 |
| changed tokens x/y/z/roll/pitch/yaw/gripper | 1101/576/473/411/392/378/**0** | 1823/675/537/400/305/296/**0** | 1615/1199/1009/988/842/913/**0** | 2102/1405/1209/900/779/800/**0** |
| non-action-vocab selections (gripper dimension only) | 3 | 3 | 1 | 0 |
| perception errors / mode | 0 / http | 0 / http | 0 / http | 0 / http |

**Reading the table:**
- **Overall:** PCD changed a token on 4142 of 6176 steps (67.1%). The gripper token never changed in any arm.
- **Null counters:** the null computes the same contrast in shadow but executes native actions. Its counters say what
  the contrast would have chosen.
- **Non-action-vocabulary selections:** all are the native gripper EOS token, not PCD choices.

**Optional runs, activation** (`optional_runs.md`):
- **Move Near:**
  - inactive PCD 0/32 and null 0/32;
  - target active on 2560/2560 steps;
  - tracker None: **PCD 126 of 2560 steps** (8 of 32 episodes; redbull can 49, pepsi can 38, sponge 20, 7up can 18,
    blue plastic bottle 1) and **null 27 of 2560** (5 of 32 episodes) (`optional_runs_addendum.md` R1; the
    `optional_runs.md` line "Tracker None 0" was wrong);
  - changed-token steps PCD 2126/2560;
  - non-action-vocab selections PCD 12 and null 14, gripper only.
- **Setting B:**
  - inactive 0/30 in both arms;
  - changed-token steps PCD 4397/4847;
  - one named target was **never masked** in 6 episodes per arm: "cream cheese" in t6 and "rack" in t9, on all three
    inits. Those episodes masked only the other target;
  - tracker None: PCD 1558 steps over all 30 episodes, of which **1493** are in the six t6 + t9 episodes (remainder 65:
    t3/i1, "bowl"); null 1246, of which **1236** in those six (remainder 10: t2/i0, "wine bottle")
    (`optional_runs_addendum.md` R4).

## Move Near (optional run 1; "predicted no-gain task"; descriptive)

- **Cases:** the Addendum 4 seed list, re-verified on GPU obs0 (`equal_to_addendum4: true`, case list `5794cffb...`):
  32 pairs on GPUs 4/5, method `pcd` (fingerprint `8350a143...`).
- **Result:**
  - baseline **19/32** -> PCD **21/32**, W 4 / L 2 (+6.25 pp), exact two-sided McNemar p **0.6875**, Newcombe CI
    [-0.083, +0.203];
  - max-step baseline 14/32 -> PCD 18/32 (+12.5 pp) vs the paper's Table 9 47.3% -> 41.3% (-6.0 pp): the opposite
    direction, descriptive only;
  - null M 32/0/0;
  - the paper's change is 58.7% -> 59.0% (+0.3 pp), inside our interval.
- **Incident:** the first launch's plans pointed at a stale server record, and both drivers stopped with
  `stop_unhealthy` at 20:15:10Z. This happened before any episode: no attempt was consumed and no outcome was seen. The
  plans were fixed and relaunched. The incident is kept as infrastructure evidence in `eval/optional/CHAIN.log`.

## Setting B (optional run 2; libero_goal transfer screen; adapted / tf_variant; never "reproduced")

- **Setup:**
  - `openvla-7b-finetuned-libero-goal@fa5ae1e`, tasks 0-9 x init 0-2, horizon 300;
  - config `pcd-libero-goal` (fingerprint `941d0fb5...`), with the frozen `libero_goal` parser `eb67f163...`;
  - startup probe `684de28e9cedcce6`;
  - the B smoke on t0/i3 was valid (15/15 checks);
  - shards (`optional_runs_addendum.md` R3): GPU 6 t0-t2 (9 pairs), GPU 7 t3-t5 (9), GPU 4 t6-t7 (6) and GPU 5 t8-t9 (6),
    GPUs 4/5 after their Move Near shards.
- **Env seed** (clarification, env-seed audit
  `/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/repro-next-20261105/audit/env_seed_audit.md` (sha256
  `1c7dabbe…`)): `seed: 7` with no `env_seed`, and vla-eval 0.8.0 drops the OpenVLA server's HELLO `seed: 0`, so all
  arms ran `env.seed(7)`, not the OpenVLA reference `env.seed(0)`. Pairing is unaffected. The libero_goal rates are
  not comparable with OpenVLA's published libero_goal numbers; none is claimed.
- **Result:**
  - baseline **23/30** -> PCD **23/30**, W 3 / L 3, p **1.0**, Newcombe CI [-0.162, +0.162];
  - null M 30/0/0;
  - max-step equals any-step (23 vs 23), because a LIBERO success ends the episode.
- **Wins and losses:**
  - PCD wins: t0/i1, t2/i0, t9/i0;
  - PCD losses: t6/i0, t6/i1, t9/i2.
  - All three losses are in tasks with one target **never masked**: "cream cheese" (t6) and "rack" (t9). One win
    (t9/i0) is also in such a task. This is a perception-coverage observation, not a tested cause.
- **Disclosed stats-view deviation:** the frozen `stats_decision.py` refused the raw B records because
  `metrics.final_success` was missing (only the SIMPLER benchmark writes it).
  - **The view:** `eval/settingb_statsview.py` (`8b6ea22c...`) adds `final_success` to **copies** of the 90 records,
    as the last per-step `success` of each episode log.
    - All 90 records are checked, with `any_step_ne_last_row: 0`.
    - The raw records are unchanged (the pre-view hash list matched after the view).
  - **Effect:** the any-step result, the only B statistic, uses `record.success` and is identical either way. Only the
    descriptive max-step row depends on the view.

## Cost (GPU-hours from lease acquisition to release)

| block | GPU-s | GPU-h |
|---|---|---|
| phase 1 (pre-primary checks and smokes; session 1 7,281 + session 2 2,812) | 10,093 | **2.80** |
| Setting A primary (GPUs 4-7) | 24,071 | **6.69** |
| optional runs (Move Near + Setting B; g4 8753.1, g5 8494.5, g6 5760.6, g7 6118.7) | 29,127.0 | **8.09** |
| **cumulative** | **63,290.9** | **17.58** of the 40 cap |

- **LLM and paid models:** 0 calls in every arm (`llm_calls: 0`), so the paid cost is $0. Perception runs on the
  local GDINO/SAM2/LaMa service over http, with p50 latency 389 ms/step on GPU (`gpu_phase1.md` 2.2).
- **Leases:** all were released at the end. No foreign leases were touched, and GPUs 0-3 were never used.
- **Where these costs come from (critic-final RE-P1):** the table is computed from **both** lease ledgers by
  `runs/pcd-port-20261103/eval/cost_report_v2.py` (294d03b1).
  - **In-place re-freeze:** at 20:13:04Z the bound `eval/manifests/eval_tools-D1.sha256` (fc2e2ef4) was re-frozen in
    place to baeefef7, the list the optional plans pinned. After all GPU work, that list was preserved as
    `eval_tools-optional.sha256` (baeefef7), and the D1 file was restored byte-exactly (fc2e2ef4). The optional runs'
    provenance list is therefore `eval_tools-optional.sha256`, not the restored D1 file.
  - **Ledger split:** the optional leases had been appended to the bound `eval/cost/lease_ledger.jsonl`. It was split:
    lines 1-32 stay in place (def7e772), and lines 33-44 moved to `lease_ledger-optional.jsonl` (482159b3). The
    concatenation is byte-identical to the pre-split file.
  - Sources: `optional_runs.md` "Bookkeeping"; critic-optional A2/A3.

**Per arm, primary** (`primary.md` section 4):

| arm | episodes | steps | wall s | s/step (median) | decodes / step | perception / step |
|---|---|---|---|---|---|---|
| baseline | 64 | 6176 | 3348 | 0.545 | 1 | 0 |
| PCD (candidate) | 64 | 6176 | 8357 | 1.351 | 2 | 1 |
| null | 64 | 6176 | 10435 | 1.675 | 3 (2 shadow + 1 native) | 1 |

- **PCD overhead:** taken from the candidate only, it is 8357 / 3348 s of wall time.
- **Null counter:** the null's `method_policy_samples` counts 5 per step, but 3 decodes are served. This is a known
  framework counter double count (critic A6), to be fixed later, never in this frozen run.
- **Wall clock, optional runs:**
  - Move Near: baseline 1343 s, null 4306 s, PCD 3467 s;
  - Setting B: baseline 1602 s, null 6679 s, PCD 5412 s.

## Timeline (UTC, 2026-11-03)

| time | event |
|---|---|
| 17:04:15 | T0: GPU phase 1, session 1 (`gpu_phase1.md`) |
| 17:43:28 | **user STOP** (TASK.md Amendment 1). The GPU-5 coke smoke was cut: its baseline record is valid, its null was killed with no record, and PCD did not run (kept as `STOPPED.json`; the directory was not reused) |
| 17:43-17:44 | **pause** |
| 17:44 | re-opening on GPUs 6 and 7 (amendment 4; **handover** of GPUs from Run X) |
| 17:50 | widened to GPUs 4-7 (amendment 5): Run X closed for good, **handover** in PROGRESS.md line 65 |
| 18:04:40 | coke seed-99 smoke VALID; the drawer seed-99 smoke is VALID too; numerics decision: the authors' env PRIMARY |
| ~18:09-18:11 | D1: card status items and the Setting A label filled (P-5, `reports/card_d1.md`) |
| 18:22-18:25 | pre-primary edits R1-R6 (`primary.md` section 1): method pin re-frozen (`eval/manifests/method_files-D1.sha256`), per-GPU server and perception identity recorded |
| 18:25-20:05 | Setting A primary on GPUs 4-7 (first scored episode 18:25:22Z) |
| 20:05 | primary complete; the Addendum 6 triggers for Move Near and Setting B fire |
| 20:09:28 | optional runs: first lease acquired |
| 20:15:10 | Move Near run 1 stops (`stop_unhealthy`, stale server record, before any episode); fixed and relaunched |
| 21:43:00 / 21:51:13 / 22:28:18 / 22:34:47 | Setting B shard ends on g6 / g7 / g5 / g4 |
| 22:35:21 | last lease released; all GPU work stopped |

## Gap analysis (descriptive)

- **Point estimate vs paper:** pooled **+21.9 pp** vs the paper's +26.6 (mean of per-task gains) or +26.7 (two-task,
  per the critic). Per task, both of our gains are +21.9 pp (7/32), against the paper's +27.3 and +26.0. The paper's
  values lie inside every one of our intervals.
- **Winner's curse from task selection:** the two tasks were chosen *because* they had the largest reported gains (the
  scope label says so, and about 70% of the paper's 9-task GDINO gain comes from these two tasks, `pcd_audit.md`
  section 6). A smaller replicated gain is the expected regression toward the mean, so a lower point estimate is not
  evidence against the paper.
- **Wide CI:** with n = 64 and 20 discordant pairs, the 95% interval spans +9.0 to +33.5 pp. It cannot separate the
  paper's magnitude from a substantially smaller one.
- **Stack differences** that could move the estimate in either direction (no tested cause):
  - **Float32 action wire:** B7 replays were VALID but over the bound. Coke differed by 2.36e-5 m / 7.63e-5 rad and
    drawer by 7.76e-3 m / 2.85e-2 rad; success flags and sticky state were identical (`gpu_phase1.md` 2.3).
  - **Env identity:** we used the vla-eval SIMPLER image, not the vendored copy; no behavioral difference was found.
  - **SAM2 `_C`:** absent on our side; the authors' state is unknown.
  - **The authors' 300-trial protocol:** it is not reconstructible. The code runs 100 seeds.
- **Baselines** sit on either side of the paper's: coke 15.6% vs 18.0%, drawer 53.1% vs 47.3%.
  - **Predeclared rule:** both are consistent, because our exact CIs, [0.053, 0.328] and [0.347, 0.709], contain 18.0%
    and 47.3%.
  - **The paper's own CIs:** Table 1 (v3+) gives 95% CIs of 18.0±4.3 and 47.3±5.6. Our coke baseline (15.6%) lies inside
    its band, [13.7, 22.3]. Our drawer baseline (53.1%) lies just above its band, [41.7, 52.9].
  - Sources: `runs/pcd-port-20261103/reports/critic-v28.review.json`; `reports/pcd_audit.md`:13 and :103.

## What was ported

| part | where | state |
|---|---|---|
| **Method `pcd`** (Setting A) and **`pcd-libero-goal`** (Setting B: the same card with only `perception_parser: libero_goal` changed) | `harnessstudio/methods/pcd/hs_pcd/`; card `paper.yaml` (per-file pin `eval/manifests/method_files-D1.sha256`, from `runs/pcd-port-20261103/method/card_d1/`); see below | fingerprints `8350a14351a738d8de77fa419c26e3e6bbcc68e5a873931b7aef0191c457092c` / `941d0fb5265d6365a5e98e074da32f1baab3e8033b5e3a6cda2385469b66a6b5` in every PCD record |
| **EXT openvla-contrast** (gated framework extension) | branch `ext2/openvla-contrast` @ `adc271ab7078cc1e3e68eef12e97126499580c0a`; `docs/extensions/openvla-contrast.md`; `runs/pcd-port-20261103/reports/ext_build.md`; see below | GPU checks G1, G2-G5, G7, G8 and the library gate PASS; G6 and G9 measured (`gpu_phase1.md` 2.1); **merged** in `4f93662` |
| **EXT simpler-seeded** (gated, stacked) | branch `ext2/simpler-seeded` @ `9b2b71445eb72cb58c1ed3d672aa2c67e93c1122`; `runs/pcd-port-20261103/reports/simpler_build.md`; see below | GPU checks B1-B7 / S6-S10 done (`gpu_phase1.md` 2.3); **merged** in `ecc9b68` (docs fixes `0092631`) |
| **Perception service** | `harnessstudio/methods/pcd/service/` (own venv); see below | GPU parity PASS: SIMPLER 42/42 and LIBERO 18/18 steps bitwise; memory peak 3.5 GiB (`gpu_phase1.md` 2.2) |

**Method.** A `null: shadow` controller. Per step:
1. the perception `/step` produces a counterfactual frame;
2. two native greedy 7-token decodes run as separate `logits` requests, one on the real frame and one with the
   counterfactual as the first image;
3. the combine is `(1 + 0.8) * s - 0.8 * c` on positions 0-5, with the gripper from the real stream and the authors'
   <7-token fallbacks, then a full-vocabulary argmax;
4. the result is executed through the server's bin table.

**EXT openvla-contrast.** `policies/openvla_logits_hs.py` adds the native-decode `logits` readout, with a token- and
action-level self-check. `openvla_hs.py` stays byte-identical.

**EXT simpler-seeded** adds:
- the authors' seeded SIMPLER contract and benchmark;
- server mode `simpler_pcd_authors`: raw instruction + 29871, cv2 INTER_AREA 224, eager attention, the authors'
  `generate` call.

**Perception service.** A re-derivation of PCD-Mask:
- GDINO-base fp32 (0.25 / 0.3);
- SAM2.1-L camera predictor without re-detection;
- Big-LaMa every step;
- the authors' SIMPLER parser, plus a frozen `libero_goal` table (`eb67f163...`).

**Scoring source** (predeclared decision P-4, `runs/pcd-port-20261103/reports/gpu_schedule_v2.md`):
- Scored runs used the **pinned worktree commits** `adc271a` / `9b2b714`. Their heads and clean status were checked at
  every seed start and after PCD (64/64).
- This deviation from scoring on `main` is disclosed here.

**Merge identity** (verbatim from the independent review `runs/pcd-port-20261103/reports/critic-merge.review.json`,
ruling a):

> Merge identity (decision P-4). harnessstudio main 0092631 = e0aa505 + merge 4f93662 (EXT openvla-contrast, ext2/openvla-contrast @ adc271a) + merge ecc9b68 (EXT simpler-seeded, ext2/simpler-seeded @ 9b2b714) + docs-only commit 0092631. Setting A, the scored reproduction, ran from 9b2b714; its entire harnessstudio/ package is byte-identical to main (git tree d3bc20ba). Setting B (libero_goal) and the EXT openvla-contrast GPU checks G1-G9 ran from adc271a; that package is byte-identical to main at the openvla-contrast merge commit 4f93662 (git tree 5a2ca721). At main HEAD, one file of the adc271a tree differs: harnessstudio/benchmarks/__init__.py (adc271a sha256 4056b1cc..., main fa0a958b...). simpler-seeded's reviewed edit S-8a rewrote only its module docstring. The file is AST-identical without the docstring and has no other statements; no code reads its __doc__; and Setting B used vla-eval's LIBEROBenchmark and did not load it. eval/verify_merge_identity.py therefore reports scored_identical: false for this one file (runs/pcd-port-20261103/merge/merge_identity_compare.json, benchmarks_init_ast_check.json). P-4's byte-identity condition holds for every other scored file in both trees. This file is accepted as a docstring-only, behavior-identical exception by the independent review runs/pcd-port-20261103/reports/critic-merge.review.json (ruling a).

**Post-run metadata edit.** `harnessstudio/methods/pcd/hs_pcd/paper.yaml` and `harnessstudio/methods/pcd/README.md`
were edited at 22:45:45Z by task conv-results-fill (results changelog line and results paragraph), after scoring had
ended at 22:34:46Z. This is post-run results metadata only: no parameter or code changed and the method fingerprints
are unchanged (`8350a143...`, `941d0fb5...`), but the D1 method pins (`eval/manifests/method_files-D1.sha256`, paper.yaml
`e1432d30`) no longer verify against the live tree for those two files (paper.yaml is now `ea55e539...`).

## Released-artifact findings (`runs/pcd-port-20261103/reports/pcd_audit.md`, sha256 `4ffa776e47fbc245...`)

- **The code differs from the paper's Eq. 2, and the port follows the code.** The released OpenVLA path runs two
  *independent* greedy decodes and combines them per position after decoding, `final[:-1] = (1+a)*s[:-1] - a*c[:-1]`.
  - The gripper is not contrasted.
  - The argmax is deterministic, where Alg. 1 says "sample".
  - There is no plausibility cutoff.
  - The <7-token fallbacks are code-only.
- **Point/Box excluded:** both come from simulator ground-truth segmentation, which is privileged. Only the
  GroundingDINO path is ported.
- **Protocol:** Table 1 uses the authors' seeded prepackaged protocol (`simpler_env.make` + `reset(seed)`, full
  TimeLimit, any-step success). That is not vla-eval's visual-matching / variant-aggregation grids. Our runs use the
  authors' protocol through EXT simpler-seeded.
- **The 300-trial ambiguity:** v3+ says 300 trials per task, but the code runs `n_trajs=100`. How 300 were produced is
  not recorded. Author issue #7: results "fluctuate due to hardware, library versions, random seeds".
- **The miniVLA row is not reproducible:** the LIBERO-90 table has no code, checkpoint id, episode count, mask variant
  or alpha. No OpenVLA-LIBERO row exists.
- **Licences:** Koorye/PCD @ `31b027e` has **no licence**, so the port is an independent re-derivation citing file:line.
  The companion Eres-lcy/PCD @ `00c4ac0` is **Apache-2.0**; no code was taken, and the ICLR values were kept.

## Fidelity items and substitutions (card `harnessstudio/methods/pcd/hs_pcd/paper.yaml`)

| item | ours | authors | status |
|---|---|---|---|
| numerics env (Setting A) | **authors' env PRIMARY**: transformers 4.44.2 / torch 2.3.1 / eager, bf16 (`venvs/openvla-authors-numerics`); S8/S9/S7/S6a/S6b PASS; the fallback was not selected (`eval/decisions/numerics_env.json`) | the same | matched; S10 measurement: 8/64 frames differ, numerics vs fallback |
| float32 action wire | HS `protocol.py:61`, vla-eval `predict.py:241` cast actions to float32 | float64 | **substitution**. B7 VALID, rule violated (see Gap analysis). The CPU dry run had predicted a violation |
| HarnessStudio transport | two `logits`-readout requests; combine in numpy float32 on lossless msgpack rows | two in-process `vla.generate(output_scores)` | **substitution**; combine parity bitwise (12,000 cases) |
| env identity | vla-eval SIMPLER image `sha256:f77449671884afee5c997f2b933d0d2db503debe86961d2f6c378d47276d10fe` (SimplerEnv 4ab7178 / ManiSkill2_real2sim ef7a4d4) | the vendored copy at 31b027e | **substitution**; no behavioral difference found; overlay PNG bytes unverifiable |
| SAM2 `_C` | not built; the code's own fallback; `sam2_C: false` on every GPU | unknown, most likely absent | **unresolved fidelity item** |
| whole-cabinet masks | the authors' `dresser` alias masks the whole cabinet | the same code | disclosed next to the Close Drawer rows |
| Setting B partial masks | t6 "cream cheese" and t9 "rack" never masked; t0/t3 whole-cabinet | n/a | disclosed |
| prompt / resize / gripper / rotation (A) | server mode `simpler_pcd_authors` + the seeded contract | the same | **matched**: S6a/S6b tokens and float64 actions byte-equal to the authors' `OpenVLAInference`; post-processing parity 0 / 22,600 steps |
| Setting B protocol | `fa5ae1e`, libero_goal, horizon 300, LIBERO terminate-on-success, the checkpoint's prompt template | n/a | adapted / tf_variant |

## CPU validation (before any GPU run)

| check | result | evidence |
|---|---|---|
| combine parity vs the authors' `openvla_contrast.py` | 12,000 randomized cases including the fallbacks: 0 token and 0 bitwise differences | `runs/pcd-port-20261103/method/parity/parity_result.json`, `method/fixes2/parity/parity_result.json` |
| perception parity (CPU) | 18 LIBERO-Pro + 42 SIMPLER steps: counterfactual frames bitwise equal, masks IoU 1.0 | `runs/pcd-port-20261103/perception/parity/parity_result.json`, `perception/parity_simpler/parity_result.json` |
| post-processing parity | 0 mismatches / 22,600 steps | `runs/pcd-port-20261103/simpler_ext/build/results/authors_post_parity_worktree_image.json` |
| decode vs the authors over all token ids | 0 mismatches in both envs | `runs/pcd-port-20261103/simpler_ext/build/results/decode_vs_authors_openvla-*.json` |
| `hs verify` tier 0 / tier 1 (`pcd`, `pcd-libero-goal`) | PASS / PASS; method tests 42/42 | `runs/pcd-port-20261103/method/card_d1/` |
| framework regress (82 methods) | identical before/after for both extensions | `runs/pcd-port-20261103/ext/build/fixes2/regress-compare.json`, `simpler_ext/fixes4/regress-compare.json` |
| branch-scoped scope tests | as expected in scratch-clone merge simulations; pre-merge check PASSED | `runs/pcd-port-20261103/ext/build/fixes2/scope_sim.jsonl`, `ext/build/fixes5/premerge_scope_check.log`, `simpler_ext/fixes4/scope_sim.txt` |

## Predeclared plan (`runs/pcd-port-20261103/SETTING.md` v2 + Addenda 1-5, sha256 `888ec105...` as cited before the GPU runs)

- **Subset:** Pick Coke Can seeds 0-31 (80 steps) and Close Drawer seeds 0-13, 16, 17, 19-22, 26, 27, 29, 31, 33-39,
  41 (113 steps). 64 pairs; GPU obs0 64/64 distinct, so there was no edit-5 replacement (case list `cf405fe2`).
- **Decision:**
  - exact two-sided McNemar on the 64 pooled pairs, alpha 0.05, with the "Newcombe hybrid score (uncorrected phi)" CI
    from the frozen `eval/stats_decision.py` (sha256 `bfd472cb...` at run time);
  - labels:
    - "gain reproduced in direction" (W > L and p < 0.05);
    - "magnitude compatible" (the paper's +26.6 pp is inside the CI);
    - "not reproduced" (CI upper bound < +13.3 pp);
    - otherwise "inconclusive".
- **Null before PCD**, per seed; X blocks (Addendum 5).
- **Activation:** E-4 "mechanism inactive" if a non-robot target mask is present on fewer than 50% of steps.
- **Optional runs:** Move Near (the Addendum 4 list) and Setting B (30 pairs), on outcome-independent triggers.
- **Triggers:** the A-readiness and A-smoke triggers (edit 4), and the stop time.

The SETTING sha cited here is the value bound when this document was predeclared. Later Addenda (e.g. Addendum 6
triggers) are cited by `primary.md` and `optional_runs.md`.

## Open items

- **Merges:** done (`4f93662`, `ecc9b68`, `0092631`, then the PCD files commit). Merge identity: `scored_identical`
  false with 1 docstring-only mismatch (`harnessstudio/benchmarks/__init__.py`, adc271a tree), accepted by
  critic-merge ruling a (see "Scoring source"). Not pushed: pushing waits for the final critic.
- **Null counter:** the null arm's `method_policy_samples` double count (5/step counted vs 3 served) needs a framework
  fix in a future task (critic A6).
- **Card source binding:** `paper.yaml` binds `gpu_phase1.md` at sha `c9617fd7...`. The file is now `12f99275...`, but
  only the R4 cost lines changed after D1 (critic D3/A5). The note is recorded in the changelog, and the sources entry
  is left as it was frozen.
- **Setting B perception coverage:** "cream cheese" and "rack" are never masked by the frozen `libero_goal` table.
  Changing it would need a new frozen table and a new run, never this one.
- **Setting B stats view:** the frozen `stats_decision.py` cannot read LIBERO records without `final_success`. A
  versioned stats tool is a later task.
- **Vendored binaries in `7361cd4`:** that commit includes 10 vendored upstream LaMa test images (2.98 MB) under
  `methods/pcd/service/third_party/lama_ia_3889cc7/lama/saicinpainting/evaluation/masks/countless/`
  (`images/{gcim.jpg, gray_segmentation.png, segmentation.png, sparse.png}` and 6 `memprof/*.png`).
  - They are evaluation/test images of the blob-identical Apache-2.0 Inpaint-Anything tree, not weights or compiled
    code.
  - The commit message's "no binaries/venv" is therefore inaccurate; correct is "no compiled code, venv or weights".
  - `7361cd4` is not amended, because the regress record and the COORDINATION DONE line bind it (critic-final C3).
- **Remaining scope:** the other 7 SIMPLER tasks and the paper's 300-trial denominators were not run. No 9-task claim
  is made.
