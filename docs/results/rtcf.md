# RTCF (arXiv:2608.04527 v2, withdrawn in v3) in HarnessStudio v2

Status: **LIBERO-Long reproduction, run `runs/rtcf-20261031` (2026-10-31/11-01).** Package:
`methods/rtcf` (entry points `rtcf`; ablations `rtcf-framenn` and `rtcf-timedomain` exist but were not run).
Fidelity is `tf_variant`. All scored RTCF runs used main 222527b, rtcf fingerprint `a2b34bb7…`, and frozen memory
`ccebf9e0…`.

**Every paper number below is a withdrawn claim.** The authors withdrew the paper on 2026-10-28 (arXiv v3: "the
experimental evidence is not yet sufficient to fully support its main conclusions"). A match would not validate the
paper, and a miss does not by itself show a port defect.

## Outcome

<!-- FILL:outcome -->
On LIBERO-Long (complete seeds seed101, seed102, seed103, seed104, seed105; 500 paired episodes), the frozen pi0-FAST baseline solved 319/500 (63.8%) and RTCF solved 311/500 (62.2%): Δ = -1.6 [-7.27, 4.07] pp (clustered 95% CI), 49 wins / 57 losses, exact McNemar p = 0.497. The withdrawn v2 claim is 61.6% → 68.6% (+7.0 pp, n = 500 per arm); ours − paper: baseline +2.2 [-6.77, 11.17], RTCF -6.4 [-15.52, 2.72] pp.
<!-- /FILL:outcome -->

In short:
<!-- FILL:in_short -->
- **The frozen pi0-FAST baseline is consistent with the paper's baseline**: 63.8% (319/500, complete seeds) vs 61.6%, ours − paper +2.2 pp [-6.8, +11.2]. This holds only after serving the checkpoint with the z-score normalization it was trained with (see "Serving fix").
- **The paper's RTCF gain (+7.0 pp on Long) is not reproduced as a significant effect**: pooled Δ -1.6 pp [-7.3, +4.1], 49 rescues / 57 regressions, exact McNemar p = 0.497. The paper's +7.0 pp lies **outside** our 95% CI (upper bound +4.1 pp), so a gain of that size is excluded at this n for this port and protocol (tf_variant: PCA omitted, disjoint 127-memory bank, a-priori constants, z-score serving); it does not test the paper's own, unreleased configuration; ours − paper on the RTCF row is -6.4 pp [-15.5, +2.7]. Seed 103 had the largest loss (57 vs 68, 7/18, uncorrected per-seed p = 0.043; not significant after correction for five per-seed tests; the pooled test is the predeclared headline).
- The mechanism runs as specified: it corrects 99.5% of logged decisions (34017/34187); a memory of the same task is retrieved on 84.1% of decisions (post hoc; task 8 has no memory; 98.4–98.8% in development leave-one-out); b* tracks progress.
- The corrections are small: median per-decision max |Δa| 0.018 action units (seed-101 analysis), a size set by λ0 = 0.1.
<!-- /FILL:in_short -->

## Paper (withdrawn v2 Table 1) vs ours

The paper's matched block uses 500 episodes per suite and condition: 10 tasks × 10 initial states × seeds 101–105.
Ours uses the same inventory per seed: evaluation init states 0–9 (`runs/rtcf-20261031/protocol/run-order.md`,
addenda 1–3).
<!-- FILL:order -->
Seeds were run in the predeclared priority order 101→105. Seed 104 ran early on GPU 6 because its baseline shards were there (critic ruling s4-early-start), so the completion order of the complete seeds was 101, 102, 104, 103, 105 and was set by capacity, not outcomes.
<!-- /FILL:order -->

**Stopping rule (predeclared):** only seeds whose two arms are complete and valid for all 10 tasks enter the pooled
rows. Incomplete seeds are listed separately as partial and never pooled. CIs are clustered by (task, init state),
because the seeds of one (task, init state) are correlated replicates. McNemar is exact and two-sided, on discordant
pairs; the pooled test treats seeds as independent, so lead with the clustered difference. Our n per seed is 100,
against the paper's 500 per suite.

<!-- FILL:suite_table -->
| Suite | Paper baseline (n=500) | Paper RTCF (n=500) | Our baseline | Our RTCF | Paired n | RTCF wins / losses | McNemar p | Δ pp (clustered 95% CI) | ours − paper, baseline / RTCF (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| LIBERO-Long | 61.6 (308) | 68.6 (343) | 63.8 (319/500) [55.91, 71.69] | 62.2 (311/500) [54.03, 70.37] | 500 | 49 / 57 | 0.497 | -1.6 [-7.27, 4.07] | +2.2 [-6.77, 11.17] / -6.4 [-15.52, 2.72] |
| LIBERO-Spatial | 96.4 (482) | 97.4 (487) | incomplete: baseline 70/100 valid, development 160/200, RTCF 0/100 (memory not frozen); not reported (stopping rule) | | | | | | |
| LIBERO-Object | 97.4 (487) | 97.8 (489) | incomplete: baseline 40/100 valid, development 80/200, RTCF 0/100 (memory not frozen); not reported (stopping rule) | | | | | | |
| LIBERO-Goal | 90.0 (450) | 89.8 (449) | incomplete: baseline 20/100 valid, development 40/200, RTCF 0/100 (memory not frozen); not reported (stopping rule) | | | | | | |
<!-- /FILL:suite_table -->

Per seed (complete and partial):

<!-- FILL:seed_table -->
| Suite | Seed | Status (valid baseline / RTCF of expected) | Our baseline | Our RTCF | Paired n | RTCF wins / losses | McNemar p |
|---|---|---|---|---|---|---|---|
| LIBERO-Long | seed101 | complete (100 / 100 of 100) | 63.0 (63/100) [53.49, 72.51] | 62.0 (62/100) [52.44, 71.56] | 100 | 11 / 12 | 1 |
| LIBERO-Long | seed102 | complete (100 / 100 of 100) | 59.0 (59/100) [49.31, 68.69] | 63.0 (63/100) [53.49, 72.51] | 100 | 13 / 9 | 0.523 |
| LIBERO-Long | seed103 | complete (100 / 100 of 100) | 68.0 (68/100) [58.81, 77.19] | 57.0 (57/100) [47.25, 66.75] | 100 | 7 / 18 | 0.0433 |
| LIBERO-Long | seed104 | complete (100 / 100 of 100) | 61.0 (61/100) [51.39, 70.61] | 66.0 (66/100) [56.67, 75.33] | 100 | 12 / 7 | 0.359 |
| LIBERO-Long | seed105 | complete (100 / 100 of 100) | 68.0 (68/100) [58.81, 77.19] | 63.0 (63/100) [53.49, 72.51] | 100 | 6 / 11 | 0.332 |
<!-- /FILL:seed_table -->

S5 (seed-101 rows of Spatial, Object and Goal; run only on capacity not needed by Long shards):
<!-- FILL:s5_status -->
| Suite (seed 101; dev init 10–29 seed 7) | Baseline valid | Development valid | RTCF valid | Memory frozen | Status |
|---|---|---|---|---|---|
| LIBERO-Spatial | 70/100 (70 traces) | 160/200 (160) | 0/100 (0) | no | incomplete: baseline 70/100 valid, development 160/200, RTCF 0/100 (memory not frozen); not reported (stopping rule) |
| LIBERO-Object | 40/100 (40 traces) | 80/200 (80) | 0/100 (0) | no | incomplete: baseline 40/100 valid, development 80/200, RTCF 0/100 (memory not frozen); not reported (stopping rule) |
| LIBERO-Goal | 20/100 (20 traces) | 40/200 (40) | 0/100 (0) | no | incomplete: baseline 20/100 valid, development 40/200, RTCF 0/100 (memory not frozen); not reported (stopping rule) |
<!-- /FILL:s5_status -->

The paper does not report RTCF's own rescue/regression counts or any significance test. On Long, its Frame-NN
ablation shows similar churn (57 rescues / 47 regressions in 500).

## LIBERO-Long per task (pooled complete seeds; paper Fig. 6 values read from a bar chart, approximate)

<!-- FILL:long_tasks -->
| Task | Paper baseline % (n=50, read from Fig. 6, approx.) | Paper RTCF % (approx.) | Our baseline | Our RTCF |
|---|---|---|---|---|
| 0 | 82 | 98 | 98 (49/50) | 90 (45/50) |
| 1 | 100 | 100 | 100 (50/50) | 100 (50/50) |
| 2 | 30 | 38 | 36 (18/50) | 30 (15/50) |
| 3 | 34 | 48 | 46 (23/50) | 38 (19/50) |
| 4 | 96 | 92 | 80 (40/50) | 100 (50/50) |
| 5 | 46 | 68 | 64 (32/50) | 74 (37/50) |
| 6 | 100 | 94 | 90 (45/50) | 70 (35/50) |
| 7 | 90 | 100 | 90 (45/50) | 80 (40/50) |
| 8 | 6 | 0 | 2 (1/50) | 0 (0/50) |
| 9 | 32 | 48 | 32 (16/50) | 40 (20/50) |
<!-- /FILL:long_tasks -->

## Mechanism activity and overhead control

<!-- FILL:activation -->
| Suite | Arm | Decisions logged | Corrected | Skips (empty / end / decode-failed / unusable) | Mean clipped frac (δ_max) | Mean frac at ±1 limit | Proposals with zero-chunk signature | Same-task retrieval (post hoc) | Latency ms enc / PMA / op (median, p95) |
|---|---|---|---|---|---|---|---|---|---|
| LIBERO-Long | candidate | 34187 | 34017 | 0 / 0 / 170 / 0 | 0.035 | 0.097 | 761 | 0.841 | 48.49, 57.31 / 13.709, 16.267 / 61.84, 73.24 |
| LIBERO-Long | null (shadow) | 1283 | 1283 | 0 / 0 / 0 / 0 | 0.033 | 0.083 | 27 | 0.830 | 49.65, 56.27 / 13.877, 16.519 / 63.56, 72.81 |

- LIBERO-Long: derived null vs baseline, 20/20 identical action sequences (0 wins / 0 losses).

Decisions logged exclude each successful episode's final decision (framework logs diagnostics after execution). Null-arm activation is shadow: corrections computed but not executed.
<!-- /FILL:activation -->

**S3 analysis** (seed 101 only; `runs/rtcf-20261031/analysis/s3-analysis.md`, an independent analyst, descriptive only):
- **The correction is λ0-limited.**
  - Max |Δa| per decision: median 0.018 action units, p99 0.038, max 0.051, against a per-entry bound of 0.030–0.079.
  - δ_max clips only 3.7% of the active coefficients, so λ0 = 0.1, not δ_max, sets the size.
- **Rescues and regressions are mirror-symmetric** (11 vs 12).
  - Over the first 100 steps, rescued and regressed episodes cannot be told apart by retrieval cost, residual norm or
    clipping.
  - Trajectories diverge physically by more than 1 cm at a median of about step 45, just before the first grasp.
  - This is consistent with a small, undirected perturbation of a deterministic closed loop. It is not proof: no
    magnitude-matched random-direction control was run.
- **End of memory.**
  - In failing episodes, b* reaches the memory's last position (median decision 46) and keeps applying that terminal
    chunk, about 30 more decisions.
  - The `past_memory_end` skip cannot fire with read offset 0 (positions are capped at L−1).
  - Whether the paper keeps correcting toward a memory's last chunk is unknown.
- **Memory coverage.**
  - Only 127 of the 200 development rollouts succeeded, so the hardest tasks have the sparsest memories: t2, t3 and t9
    have 7–8 memories each.
  - Task 8 has none. RTCF retrieves task 2 (the other moka-pot/stove scene) on all of t8's 1030 decisions. Both arms
    score 0/10 on t8, as does the paper's RTCF.
- **The derived null is M 20/20** (R 0, X 0): with the frozen memory loaded and PMA running in shadow, executed
  actions are identical to the baseline.
- **pi0-FAST itself fails to decode its FAST tokens on about 2% of proposals.** openpi then returns its zero chunk;
  see the "zero-chunk signature" column. These proposals are counted and, following Alg. 1, still corrected, because
  the paper's skip rule applies only to an undecodable *memory* chunk. Memory records with that signature are skipped
  (the "decode-failed" skips).

## Serving fix: pi0_fast_libero must be served with z-score normalization

`repair/diagnosis.md` and `eval/confirm-zscore/report.md`:
- **The cause.**
  - The openpi revision pinned by the vla-eval/HarnessStudio pi0 server (981483d) normalizes every LIBERO FAST config
    with quantile statistics (commits a3d8ea3/540ebe8, 2025-07-05).
  - The `pi0_fast_libero` checkpoint (GCS objects dated 2025-05-28) was trained with z-score normalization.
- **The symptoms.**
  - Position commands up to ±2.2; 42–45% of entries had |a| > 1.
  - The first Long baseline scored 0/10, including the paper's 50/50 tasks.
- **How it was found.**
  - A demo-frame probe emulating z-score through the quantile server cut position MAE from 0.58 to 0.06–0.11.
  - Episodes confirmed it: 4/4 deterministic failures succeeded.
- **The fix.**
  - Serve with `use_quantile_norm=False`: a reviewed run-local wrapper, plus the extension `ext2/openpi-norm`
    (`--args.norm_mode`). Its GPU equivalence check G1 passed bitwise (`eval/g1_ext2/`).
  - Merge status:
<!-- FILL:merge -->
`ext2/openpi-norm` **merged** into main as 605b5f9 on 2026-11-01T10:17Z; `regress.py` on main: ok (port/package/regress_main.json).
<!-- /FILL:merge -->
  - The method's `norm_mode: zscore` must match the server. An off-FAST-token-grid proposal raises.
  - The live FAST integrality error was 9.9e-6 in z-score.
- All stopped quantile-served episodes are kept as audit evidence and never pooled.
- **Diagnostic episodes on evaluation cases (disclosed).** The 4 confirmation episodes were native-policy runs on
  evaluation cases (Long t0, t1 and t6 at init 0, and Spatial t0 init 0, seed 101) before the protocol freeze.
  - They are diagnostic records only.
  - The z-score choice rested on source history and a predeclared prediction, not on those outcomes.
  - S1z re-ran those cases fresh, and they are not substituted (critic review-zscore C2).

## What was ported

No code was released, so the port implements v2 Eq. 5–11 and Algorithm 1.
- **Retrieve in Time: PMA, Eq. 5–9.**
  - Cosine cost, open begin, steps δ ∈ [0, v_max] with prior γ|δ−1|, score J/t.
  - Top-1 over the whole suite bank; no task id is used.
- **Correct in Frequency, Eq. 10–11.**
  - Clipped low-frequency residual on the motion dims, in pi0-FAST's own z-score FAST coefficient space:
    C = s·DCT(A), s = 1/(std+1e-6).
  - Offsets land in the masked DC term. λ0 = 0.1. The gripper is bitwise unchanged.
- **Form:** one `action_op`, a derived null and a development stage. In collection mode the candidate is the native
  policy, and native equivalence was 5/5 identical.

## Fidelity: `tf_variant`, substitutions and why (full list in `methods/rtcf/hs_rtcf/paper.yaml`)

| Paper | Here | Why / effect |
|---|---|---|
| SigLIP features projected on a **fitted** PCA subspace (centering + low-rank truncation), then ℓ2 | raw ℓ2-normalized SigLIP pooled features (1152-d) = full-rank PCA **without** centering | Training-free boundary. This changes the **retrieval geometry**, not only the cost, so neighbour rankings may differ. Measured cost spread: median 0.132, IQR 0.085. |
| SigLIP variant/pooling/camera unreported | `google/siglip-so400m-patch14-224` @ d04cf29, pooler_output, agentview, float32; standalone service with weights and processor sha-pinned and checked through `/health` | The text points to a standalone SigLIP; the paper's compute claims fit the policy's own encoder better. |
| Memory: seed-7 successes, initial states unreported (likely all 50, including evaluation states) | Development init states **10–29** (20/task), seed 7, successes only: 127 memories, 6176 positions, 8 decode-failed; frozen `ccebf9e0…`; disjoint from evaluation init states 0–9 | No evaluation scene in memory, and a smaller bank than the paper's probable ~300. Task 8 has no memory. |
| F_c, δ_max unreported | F_c = 3; δ_max = 1.0 in z-score coefficient units (per-coefficient action bound 0.035–0.091), a priori | Cannot be selected without robot outcomes. |
| v_max, γ unreported | (2, 0.05): the critic's leave-one-out rule on the Long development memory. All 18 configurations tied within 1 pp (same-task top-1 98.44–98.80%), so the rule kept the a-priori value; selected once on Long | Development-only scalar selection, frozen before any evaluation episode. |
| Read offset ambiguous | read at b* | The record at b* is the chunk planned from the aligned observation. |
| "Cannot be decoded → skip" | exact openpi zero-chunk signature; missing, unusable or wrong-shape memory chunks also skip | FAST/openpi decode failures silently return zeros. |
| Replan cadence unreported | **5** executed actions per call (openpi LIBERO client) | The protocol fingerprints match openpi's client. |
| Benchmark | vla-eval LIBERO, **quat_no_antipodal: true**, num_steps_wait 10, horizon 520 | vla-eval's default axis-angle flips rx (robosuite convention). |
| Normalization unspecified | **z-score** (the checkpoint's training normalization) | See "Serving fix". |
| Final action limit unreported | clip changed motion entries to [−1, 1] | Effectively a no-op: OSC clips at ±1. |
| Memory = complete successful sequences | each memory misses its final policy call (≤ 5 steps) | HarnessStudio logs diagnostics after a decision's actions run. |
| — | paper withdrawn (v3) | Every paper number is a withdrawn claim. |

## Protocol

- **Policy:** openpi `pi0_fast_libero` (checkpoint tree `d59b9cc9…`, norm stats `4ece6c43…`), served z-score by
  HarnessStudio's T1 wrapper on 4 A6000s. Profile exec5 (execute_horizon 5). Greedy FAST decoding.
- **Pairing:** each task's baseline and RTCF arms ran on the same z-score server; the critic verified this for every
  arm, sub-shard and repair. One config family: stock `LIBEROBenchmark` with seed overrides. Protocol fingerprints are
  in each stage's `protocol.json`.
- **Config-family exceptions,** each with the same server and a scene-identity witness:
  - S3's slow tails ran as 22 `LIBEROSelect` offset sub-shards with `match_sequential_resets`.
  - The 19 single-case baseline repairs used the same mechanism.
  - Equivalence evidence: `eval/s3/config_equiv` (offset 0, bit-exact), `eval/s2/split/verify_t5_i15.json` (offset
    replay, bit-exact), and obs[0] equal to the paired record for every candidate case.
- **Order:**
  - S1z: Long seed-101 baseline.
  - Native equivalence and parity.
  - S2: development memory, then LOO, then freeze.
  - S3: RTCF seed 101, plus a derived null on 20 cases.
  - S4: seeds 102–105.
  - S5: seed-101 rows of Spatial, Object and Goal, only on idle capacity.
- **Cutoffs:** no Long shard after 10:00 UTC, no S5 shard after 10:30, hard stop 10:50.
- **Timeouts and repairs.**
  - S3 attempt 1 was aborted at 04:09: 3 empty_result traces under vla-eval's 30 s per-action timeout. They are
    archived in `eval/s3/aborted_r1` and have no outcomes.
  - S3 r2 and every later shard used `--action-timeout 300`. S1z and the early S4 baseline shards ran with the 30 s
    default.
  - In sync mode the timeout changes no action; it only decides whether a slow step ends the episode as empty_result.
  - 19 such S4 baseline episodes were replayed case-exactly on their own server. In all 19 the replay's actions equal
    the original's over its executed prefix. Observations are equal except for one step in 2 cases (102/t8/i1 step 140,
    103/t2/i7 step 52; renderer noise, obs[0] equal).
  - Map: `eval/s4/repairs/repair_map.jsonl`. Completed robot failures were never rerolled.

## Verification

| Check | Result |
|---|---|
| Tier 0 / tier 1 (`hs verify rtcf`, fake stack incl. prepare stage and dev-case refusal) | PASS for rtcf, rtcf-framenn and rtcf-timedomain (main 222527b) |
| Unit tests (`methods/rtcf/tests`) | 73 passed: DCT, mask/clip/bound, λ0 = 0 bitwise no-op, gripper bitwise unchanged, PMA vs brute force, zero-chunk signature, Alg. 1 fallbacks, norm_mode refusal (R10), summarize/check_split/LOO guards |
| Blind parity vs an independent re-derivation (`runs/rtcf-20261031/parity/`) | 0 failing parity-scored checks on the frozen files: Eq. 5–8 ≤ 1.4e-14; `rtcf_correct` ≤ 6.7e-16 on 100k cases; (i*, b*) 100%; sabotaged references disagree |
| Native equivalence on LIBERO (E4: collection mode vs baseline, 5 development cases) | 5/5 identical per-step action hashes |
| Live FAST integrality (E5, z-score) | max error 9.9e-6 on real chunks |
| SigLIP service | weights `30a00c26…`, revision d04cf29, pinned processor, `/health` checked by the method |
| LOO (v_max, γ) | output sha `50aac18a…`; (2, 0.05) |
| Derived null vs baseline (S3) | M 20/20, R 0, X 0 |
<!-- FILL:check_split -->
| check_split (dev ⊂ [10, 29], eval ⊂ [0, 9], fingerprint `a2b34bb7…`, memory `ccebf9e0…`) | S3 (seed 101): ok (`eval/s3/checks`); seed 102: ok; seed 103: ok; seed 104: ok; seed 105: ok (`eval/s4/checks/check_split_seed*.json`) |
<!-- /FILL:check_split -->
| Freeze manifest v0.5 (`port/loo/freeze-manifest-v0.5.json`, sha `99748358…`) | check ok at stage boundaries |
| First candidate trace (F9) | corrections applied, b* advancing, changes only in dims 0–5 within the bound |
<!-- FILL:obs0 -->
| obs[0] pairing, all candidate cases incl. partial seeds and repaired baselines (`eval/obs0_final.json`) | seed101: 100/100; seed102: 100/100 (3 repaired baselines); seed103: 100/100 (10 repaired baselines); seed104: 100/100 (4 repaired baselines); seed105: 100/100 (2 repaired baselines) |
<!-- /FILL:obs0 -->

## Costs


- GPU: **40.16 GPU-h** (144,581 GPU-s), counted as the wall time of this run's pi0-FAST servers on each leased A6000
  (GPUs 4–7) plus short extra servers. All episode work ran inside that time (`reports/s4-long-seeds.json`). The total
  includes bring-up, the quantile-served diagnosis, S1z, S2, S3, S4, the ext2 G1 check and partial S5.
- Policy: one forward per chunk in both arms. RTCF adds one SigLIP call (GPU service) and one PMA update per decision
  (latency in the activation table).
- Paid API spend: **$0** (RTCF uses no LLM).


## Gaps, candidate reasons and future work

These are candidate reasons only; none was tested in this run (`analysis/s3-analysis.md` §6,
`runs/rtcf-20261031/QUESTIONS.md` Q4):
1. **Memory differs from the paper's.**
   - Disjoint init states and a smaller bank (127 memories; t8 has none).
   - The paper's bank probably contained near-duplicates of the evaluation scenes.
   - Q4(c): a reproduction-only "leaky" memory. Q4(d): a larger disjoint memory on init states 10–49.
2. **PCA omitted** (retrieval geometry). Q4(b)/Q2: a separately labelled, reproduction-only PCA-fit variant.
3. **A-priori constants** (F_c, δ_max). λ0 = 0.1 makes the correction small, so its effect may sit below our n.
4. **Undirected churn.** Q4(a): a norm-matched random-direction residual control would show whether memory content
   matters at all.
5. **Standalone vs policy-tower SigLIP encoder.** Lossless PNGs of the development frames are kept for this check.
6. **End-of-memory semantics** (keep applying the terminal chunk vs stop).
7. **n:** 500 paired Long episodes, the paper's inventory. The pooled CI excludes +7 pp but cannot exclude gains up to about +4 pp or losses down to about −7 pp.
8. **A withdrawn paper** with internal inconsistencies (its Fig. 5 vs Table 1).
