# TTF-VLA (arXiv:2508.19257v3) on OpenVLA-7B / LIBERO: not reproduced (non-matching baseline regime; adapted)

> **Status: ACCEPTED WITH CONDITIONS by critic-ttf-final.** The review is `reviews/critic-ttf-final.md` (sha256
> `d1170e947d48afbd…`, agent a027b13c594984db6, 2026-11-05T23:37Z). Its conditions C1-C7, required disclosures and
> interpretation are binding on this document. §14 maps each condition to the section that meets it.
>
> **v3 addition:** the section "Diagnostics (D1): decode-path checks (descriptive; changes no label)", after §12,
> reports the separately frozen D1 experiment under critic-ttf-d1-results (ACCEPT WITH CONDITIONS, `fe94642b…`).
> It changes no number, table or label of the sealed result.
>
> **v5 addition:** the section "TTF-C: TTF on a causal decode (separate decode-variant experiment)", after
> "Diagnostics (D1)", reports the separately frozen and separately sealed TTF-C experiment under critic-ttfc-final
> (ACCEPT WITH CONDITIONS, `a5e63bc2…`). TTF-C appears only in that section. It changes no number, table, headline or
> label of the sealed primary result.
>
> **Where this document lives:** it is `harnessstudio/docs/results/ttf_vla.md`. Its source is `results/ttf_vla.v6.md`
> of run `repro-next-20261105`, written by rmh_converter (task conv-ttf-results-1, 2026-11-05, 23:40Z to 23:58Z UTC,
> `date -u`; v2 corrections 2026-11-06 ~03:30Z, task conv-ttf-d1-1 hand-off, critic-ttf-docs D1/D2; v3 D1 diagnostics
> 2026-11-06 from 06:28Z, task conv-ttf-results-2, critic-ttf-d1-results C1-C7; v5 TTF-C section 2026-11-06 from
> 12:16Z, task conv-ttf-results-2, critic-ttfc-final CF-1..CF-8).
> The two files are byte-identical.
>
> **Paths:** relative paths without a `harnessstudio/` prefix are relative to the run folder
> `/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/repro-next-20261105/`.
>
> **Hashes:** sha256, truncated to 16 hex digits; the full values are in the cited files.
>
> **Numbers:** every number comes from the sealed result `sealed/` (SEAL `dda8177cb4fe05cf…`, 17 files) or the files
> cited next to it. No label of the primary result in this document differs from `sealed/labels.json`
> (`ce0f70f8131a244b…`). The TTF-C section's numbers and labels come only from its own seal `ttfc/sealed/` (SEAL
> `9f273f8ba24bbe40…`, 19 files) or from critic-ttfc-final's `recomputed` block.

## 1. Outcome

**Headline (label of record, qualified as C2 and C7 require):** Not reproduced on the released-code (paper-faithful)
decode path, whose paper-baseline arm does not reproduce the paper's baseline (35.0/46.0/40.5/5.0% vs
82.0/66.5/77.0/48.0%); adapted reproduction.

**Effect estimate.** Over the predeclared 800 paired cases (4 suites × 10 tasks × init states 0-19), ttf minus
paper-baseline is **D = −0.62 pp**:
- the Newcombe (1998b, method 10) 95% CI is **[−3.92, +2.67] pp**;
- exact McNemar p = 0.766;
- paper-baseline succeeds 253/800 (31.6%) and ttf 248/800 (31.0%);
- the cells n11/n10/n01/n00 are 160/88/93/459, where n10 is ttf-only success.

**What that means.**
- The CI excludes the paper's +4.0 pp and is consistent with zero.
- The task-clustered bootstrap (a sensitivity check) gives [−3.25, +2.25].
- The fidelity label is `adapted` (§7).

**The label applies only to a baseline regime that differs from the paper's.**
- Paper-baseline is the release's own decode path. It differs from the paper's baseline row on all four suites, by
  20.5 to 47.0 pp (Spatial 47.0, Object 20.5, Goal 36.5, Long 43.0), and pooled it is 31.6% vs the paper's 68.4%.
- The paired effect was therefore measured around a ~32% baseline, not ~68%. An absolute pp effect is not
  transportable across baselines that different.
- So this result says nothing about TTF in the paper's baseline regime.

**The mechanism was active,** so this is not an inactive-pass null:
- the step-pooled fusion rate was 0.44-0.46, against the paper's 0.41-0.45, with 0 fallbacks;
- ttf actions diverged from paper-baseline in 793 of 800 episodes.

**The overhead control was clean:** ttf-shadow, which computes TTF and discards the result, is action-identical to
paper-baseline in 800/800 cases (M 800 / R 0 / X 0).

| Block | Pairs | paper-baseline → ttf | D (pp) | 95% CI (pp) | McNemar p | Target | Label (sealed; C7 wording) |
|---|---|---|---|---|---|---|---|
| **Headline** (4 suites) | 800 | 253 → 248 | **−0.62** | **[−3.92, +2.67]** | 0.766 | +4.0 | **not reproduced (non-matching baseline regime; adapted)** |
| Secondary (Spatial/Goal/Long; never overrides the headline) | 600 | 161 → 152 | −1.50 | [−4.97, +1.97] | 0.452 | +10/3 | not reproduced (non-matching baseline regime; adapted) |

The labels come from the frozen rule (SETTING §5) applied by the frozen `analysis/ttf_labels.py`
(`b334a24e40919727…`). critic-ttf-final recomputed every number twice, with the frozen script and with an
independent scipy/statsmodels implementation, and they agree to ≥ 1e-9. The headline is not "harmful", because
U > 0.

## 2. Baseline rows next to the paper's (C2)

| Suite | paper-baseline (ours) | Wilson 95% | Paper baseline (Table 1) | Baseline-row label | Stock causal reference (unpaired, descriptive) |
|---|---|---|---|---|---|
| Spatial | 70/200 = **35.0%** | [28.7, 41.8] | 82.0 | differs | vla-eval public 85.0 |
| Object | 92/200 = **46.0%** | [39.2, 52.9] | 66.5 | differs | vla-eval public 86.0 (172/200) |
| Goal | 81/200 = **40.5%** | [33.9, 47.4] | 77.0 | differs | vla-eval public 77.0; HS stock 23/30 |
| Long | 10/200 = **5.0%** | [2.7, 9.0] | 48.0 | differs | vla-eval public 53.5; HS stock 29/60 |
| Pooled | 253/800 = **31.6%** | | 68.4 | | |

- **The stock references are not paired with this run.**
  - vla-eval's public stock-server data covers the same init states 0-19, with env_seed 0, harness 0.0.4.dev77 and
    round-robin sharding.
  - The earlier HarnessStudio stock-causal runs used env seed 7 and other cases.
  - They are comparable rates only, and no paired comparison is made against them.
- **Gap sizes.** paper − paper-baseline is 47.0 / 20.5 / 36.5 / 43.0 pp. critic-ttf-final's text says "21 to 43 pp",
  which understates Spatial (47.0). No label depends on this, and this document uses the exact values.
- **The paper's TTF rows** (84.5/72.5/79.0/53.5, average 72.4) are not reproduction targets here. Our ttf rows are
  64/96/76/12 of 200 (pooled 31.0%).

## 3. Per suite (descriptive; power 0.17-0.42)

| Suite | Cells n11/n10/n01/n00 | paper-baseline → ttf | D (pp) | 95% CI (pp) | McNemar p | Target | Sealed per-suite label (descriptive) |
|---|---|---|---|---|---|---|---|
| Spatial | 42/22/28/108 | 70 → 64 | −3.00 | [−9.86, +3.90] | 0.480 | +2.5 | consistent with the paper, not established (descriptive; non-matching baseline regime; adapted) |
| Object | 60/36/32/72 | 92 → 96 | +2.00 | [−6.02, +9.98] | 0.716 | +6.0 | consistent with the paper, not established (descriptive; non-matching baseline regime; adapted) |
| Goal | 56/20/25/99 | 81 → 76 | −2.50 | [−9.00, +4.03] | 0.551 | +2.0 | consistent with the paper, not established (descriptive; non-matching baseline regime; adapted) |
| Long | 2/10/8/180 | 10 → 12 | +1.00 | [−3.40, +5.49] | 0.815 | +5.5 | not reproduced (non-matching baseline regime; adapted); decided by 0.013 pp at a 5% floor |

- **The per-suite labels are descriptive.** Each CI is 13-16 pp wide, too wide to tell zero from the paper's effect.
  "Consistent with the paper, not established" only means that this.
  - It is **not** partial support for TTF.
  - It is **not** evidence of a per-suite effect.
- **The Long label is fragile.** Its upper limit falls 0.013 pp short of the target, at a 5% baseline floor where a
  +5.5 pp target is not comparable (critic-ttf-final F5).

## 4. Object row (C4)

**Prediction confirmed (pb 46% ≪ stock 86%); the paper's 66.5 is not explained (pb differs from it as well).**

- The predeclared rule is in SETTING §6 (E5). paper-baseline Object 92/200 minus the vla-eval stock 172/200 is
  −40.0 pp. The independent-proportions Newcombe hybrid interval is [−47.93, −31.17], so its upper limit is < 0 and the
  prediction holds.
- This comparison is unpaired. Its only role is to test the predeclared prediction, and it is not a paired
  comparison against 86%.
- paper-baseline is degraded on every suite, not only on Object. So the data do **not** support the claim that the
  non-causal prefill explains the paper's Object row.
- Absolute Object rows are not reproduction targets (SETTING §6).

## 5. Null control and activation evidence

**Null control (ttf-shadow vs paper-baseline).**
- The full `hs pair` gives M 800 / R 0 / X 0.
- Full action-hash and obs-hash streams are identical in 800/800 cases, including the two split shards whose pb and
  shadow arms ran in different server processes.
- The E6 activation validity check (exactly one hooked activation record per policy step) passed for all 2,400
  episodes.
- So any ttf − pb difference comes from the fused tokens, not from overhead, the `output_attentions` decode, or
  process changes.

**Activation evidence** (SETTING §8; reported next to the label, never changes it; `sealed/activation.json`):

| Suite | ttf fusion rate, step-pooled | Paper | Non-keyframe rate | Per-episode median [IQR] | Fallbacks | Zero-reuse episodes |
|---|---|---|---|---|---|---|
| Spatial | 0.440 | 0.412 | 0.663 | 0.436 [0.425, 0.447] | 0 | 0 |
| Object | 0.464 | 0.446 | 0.699 | 0.461 [0.456, 0.470] | 0 | 0 |
| Goal | 0.452 | 0.435 | 0.678 | 0.449 [0.439, 0.464] | 0 | 0 |
| Long | 0.457 | 0.420 | 0.687 | 0.460 [0.445, 0.472] | 0 | 0 |

- No suite is flagged: every rate is within ±10 pp of the paper's, and fallbacks are 0.
- ttf's first action divergence from paper-baseline comes at median step 2, always at a non-keyframe step.
- 7 of 800 pairs never diverge.
- The independent cross-check found 264,185 ttf activation records, equal to the ttf trace steps.

## 6. Protocol (frozen SETTING `a6dc411f07124b1d…`)

**Arms**, all on the same server build, GPU and fixtures per task shard:
- **paper-baseline:** the release's `aptube_enabled=False` path, with stock visual tokens and
  `generate(output_attentions=True)`;
- **ttf:** hybrid fusion, K = 3, τ = 0.03, top-k 70, LLM layer-15 text attention, with the same decode call;
- **ttf-shadow:** the full TTF computation, with the stock tokens sent to the projector.

**Cases.** LIBERO Spatial, Object, Goal and Long (libero_10), all 10 tasks each, vla-eval sequential episodes 0-19
(init states 0-19): **800 cases per arm and 2,400 episodes**.
- Benchmark: seed 7, env_seed 0, `num_steps_wait` 10, official horizons 220/280/300/520, any-step success.
- Every failure ran to its official horizon.
- Success is evaluator-only.
- The public inputs are image and language only.

**Policy.** Frozen per-suite OpenVLA-7B LIBERO fine-tunes at pinned HF revisions:
- spatial `962318ce…`, object `287d6cfd…`, goal `fa5ae1e7…`, 10 `80970322…`;
- remote code `openvla/openvla-7b` @ `47a0ec7f…`;
- inference only, with no gradients and no training. TTF has no learnable parameters.

**Software.**
- Server env: torch 2.10.0, transformers 4.40.1.
- Benchmark: vla-eval 0.8.0 @ `41bd8a44`, Docker image digest `d656fe8b…`.
- Executing tree: HarnessStudio main `5a2ce3b` (EXT openvla-vision-token-hook; `HEAD:harnessstudio` `6360a96c…`),
  unchanged during the run.

**Method.** `method/` hs-ttf 0.1.0, MANIFEST `a611af04503c8922…`. Fingerprints:
- ttf `c507c724…`;
- ttf-paper-baseline `c125ad52…`;
- ttf-shadow `78f54245…`.

`harnessstudio/methods/ttf` is byte-identical to the scored `method/` per MANIFEST `a611af04…` at commit 8a24164; since then only one changelog string was appended to `hs_ttf/paper.yaml` and `README.md` was added (metadata; fingerprints unchanged); see §13.

## 7. Required disclosures (critic-ttf-final `required_disclosures`)

1. **Fidelity: `adapted` (the label of record in every table). Substitutions:**
   - vla-eval 0.8.0 + Docker + PIL preprocessing (JPEG round trip, Lanczos 224, center crop) instead of the authors' TF
     pipeline;
   - torch 2.10.0 instead of the authors' 2.2.0;
   - the README mode fix `attention_guided` → `hybrid`, because the README's mode crashes every episode;
   - constants from the paper text (K = 3, τ = 0.03, k = 70, layer 15, text attention);
   - attention taken from the previous step's LLM layer-15 prefill, as in the code, not as in Alg. 1;
   - a float32 action wire;
   - a per-episode state object;
   - env_seed 0;
   - pinned HF revisions.
2. **Released-code decode path.**
   - Every paper arm uses the release's `generate(output_attentions=True)`. Under transformers 4.40.1 the SDPA
     attention falls back to eager **without a causal mask** in the prefill.
   - This is reproduced, not fixed.
   - CPU parity covers the code path only (random weights, 0 differences). **P6, the real-weight parity check, was not
     run.**
3. **Variant and K were selected on the evaluation suites** by the authors (paper Table 3, Fig. 5), so regression to
   the mean is expected.
4. **All four paper-baseline rows differ** from the paper's: 35.0/46.0/40.5/5.0 vs 82.0/66.5/77.0/48.0; pooled 31.6 vs
   68.4.
5. **The stock causal references are unpaired:**
   - vla-eval public 85.0/86.0/77.0/53.5, with env_seed 0, harness 0.0.4.dev77 and round-robin;
   - HarnessStudio stock Long 29/60 and Goal 23/30, with env seed 7 and other cases.
6. **The Long per-suite label is decided by 0.013 pp at a 5% floor.** Per-suite labels are descriptive, with power
   0.17-0.42.
7. **A1 was a cost-only amendment.**
   - It raised the primary cap from 55 to 65 GPU-h. critic-ttf-a1 approved it at 09:16:53Z, with 11.18 primary GPU-h
     used and no outcome viewed.
   - It is monotone.
   - It decided whether a label exists (57.55 > 55), not which label.
8. **Process events.** These are in `results/RESULTS_ADDENDUM.md` (C1):
   - incidents I1-I5, including I2;
   - the E3 per-arm-shard ruling and the retry table;
   - A2, approved and not used;
   - the scheduler hand-off and split shards;
   - the INVARIANTS 8 slip;
   - the missing libero_10 A1 record;
   - the seal identity: the sealed files name eval-ttf-primary-1b / a7583391, but eval-ttf-primary-1c / ac7c496a ran
     the seal;
   - `seal.py`'s 5 unreviewed, refusal-only lines;
   - all exposure events;
   - the 170 GPU-s health smoke.
9. **Wording correction.** The frozen SETTING §2 has a "Disclosure (paper finding)" that states the paper's rows were
   measured with a bidirectional prefill. That is stronger than the evidence. Only this is established: **the released
   code's decode path takes a non-causal prefill under transformers 4.40.1.** Whether the authors' Table 1 runs took
   that path is unknown, and §8 treats it as an open hypothesis. This document follows critic-ttf-final's wording,
   and the SETTING file itself is unchanged.

## 8. The baseline gap (C3): all causes are untested hypotheses

1. **The size of the gap.** All four paper-baseline rows differ from the paper's (§2), and pooled it is 31.6% vs
   68.4%.
2. **Stock causal references are close to the paper's Spatial, Goal and Long baselines.**
   - vla-eval public stock gives 85.0/86.0/77.0/53.5 on init states 0-19. Earlier HarnessStudio stock-causal runs gave
     Long 29/60 and Goal 23/30.
   - Within this stack, the decode path is therefore the **leading candidate** for paper-baseline's gap to stock
     OpenVLA. That inference is **unpaired**.
   - The pre-checks showed paper-baseline ≠ native causal at step 0 in 10/10 real-weight pairs. They did not separate
     the missing mask from eager-vs-SDPA kernel numerics.
3. **Why the paper's baseline does not show this degradation is not identified.** The candidate causes below are
   **untested hypotheses**, and none is established:
   - **(a) The authors' effective run did not take a non-causal prefill.** For example, they may have used a
     transformers version or attention configuration other than the pinned one. Their Spatial, Goal and Long
     baselines lie within about 1 z of the stock causal reference, which this hypothesis predicts. It leaves Object
     66.5 unexplained.
   - **(b) The non-causal path is less harmful under the authors' numerics** (torch 2.2.0, and TF instead of PIL
     JPEG/resize/crop). A gap of 20.5-47.0 pp makes this unlikely as the sole cause. It is not excluded, because an
     out-of-distribution bidirectional prefill may amplify small differences.
   - **(c) A real-weight port defect** that random-weight CPU parity cannot see. P6 was not run. Parity establishes the
     code path, not real-weight GPU numerics.
   - **(d) Env seed.** Unlikely: the primary uses the authors' env_seed 0, and stock references at seeds 0 and 7 are
     both paper-like.
   - **(e) Checkpoint revisions.** These are pinned, and the authors' revisions are not stated. Unlikely to be the main
     cause, since stock references on the same HF repos are paper-like. Unverified.
4. **Object.** The prediction is confirmed (pb 46% ≪ stock 86%), but pb also differs from the paper's 66.5, and pb is
   degraded on every suite. So the data do **not** support "the non-causal prefill explains the paper's Object row"
   (§4).

**D1 update** (v3; descriptive; see "Diagnostics (D1)" after §12; it changes no label):
- **Point 2's within-stack inference now has paired support** (native vs paper-baseline; the stock references and
  the paper's rows stay unpaired). On the same 200 scenes (init states 0-4), native causal OpenVLA succeeds 148/200
  (74.0%) against the sealed paper-baseline's 60/200 (30.0%), D +44.0 pp [35.6, 51.3], and the native rate is in line
  with the unpaired stock references (75.4-75.5%). So the release's decode path accounts for paper-baseline's gap to
  stock OpenVLA in our stack. "Decode path" means
  the missing mask together with eager-vs-SDPA numerics; D1-b does not separate the two over an episode.
- **(a) is more plausible but untested.** The release's decode call keeps the causal mask from transformers 4.41.2
  on, and the causal decode is consistent with the paper's Spatial, Goal and Long rows (paper value inside the
  native arm's Wilson 95% interval, n = 50, unpaired, init states 0-4). D1 observes nothing of the authors'
  environment.
- **(b) is not tested.**
- **(c) is weakened for step 0 only.** Restoring the mask on the release's own code path reproduces the native first
  action in 729/800 start states.
- **Object 66.5 is not consistent:** it lies below the native interval [71.5, 91.7], so the paper's Object row stays
  unexplained.
- **What remains open:** which cause explains the paper's own baseline rows. Every cause (a)-(e) remains an untested
  hypothesis.

**Post-hoc analyst note (descriptive only; not reviewed by a critic; it changes no label).**
- `analysis/ttf_post_hoc.md` (analyst-ttf-1, written after the seal; hash in §15) reads the sealed per-step records
  through proprioceptive proxies. No frames were stored, so these are proxies, not visual evidence.
- **paper-baseline failures:** all 547 ran to the horizon. They show more of the generic OpenVLA failure symptoms than
  successes do (empty grasps, gripper chatter, late-episode freezing). Early-episode action entropy is similar
  between failures and successes.
- **ttf vs paper-baseline:** ttf diverges at a median of step 2, and wins and losses involve the same proxy failure
  classes.
- **Healthier subset:** a symmetric post-hoc stratification by task difficulty shows no stratum with a sign-test
  p < 0.3. Every stratum is still in the non-causal regime, so it says nothing about TTF on stock causal OpenVLA.
- **The analyst's own ranking of causes** (it rates (a) highest) and its version and environment observations are its
  judgement. They are not test results. Under C3, every cause above remains an untested hypothesis.
- **Cheap checks the analyst proposes:**
  - a step-0 decode-variant probe (≈ 0.2 GPU-h);
  - re-running the CPU non-causal-prefill proof across transformers versions;
  - asking the authors for their `pip freeze` and logs.

## 9. What can and cannot be concluded

**Can be concluded** (critic-ttf-final §5):
- This is an adapted run on the release's own non-causal-prefill decode (§6, §7).
- In it, TTF as released, active at the paper's fusion rates, changes OpenVLA's paired success by −0.62 pp, with a
  95% CI of [−3.92, +2.67]. That excludes the reported +4.0 and is consistent with zero, and the secondary block
  agrees.
- The shadow null rules out overhead and decode-path artefacts within the comparison.
- The released-code decode path in our environment gives a heavily degraded OpenVLA: 31.6% pooled, against about
  75-80% for stock causal OpenVLA on comparable stacks.
- (v3, from D1, descriptive) Within our stack, on identical scenes, the release's decode path accounts for
  paper-baseline's gap to stock OpenVLA: native causal 148/200 vs paper-baseline 60/200 on init states 0-4 (paired), with native 74.0% in line with the unpaired stock references (75.4-75.5%).
  At step 0 the missing causal mask accounts for most of the change (v1 vs v3: 14/800 all-7 equal); kernel numerics
  alone change 71/800 (8.9%) first actions and 27/800 first tokens, never the gripper.

**Cannot be concluded:**
- TTF's effect in the paper's ~68% baseline regime, or on a standard causal decode. D1 does not test TTF; TTF-C
  (decision d10) has since run (v5) as a separate frozen decode-variant experiment, reported only in its own section
  ("TTF-C" after "Diagnostics (D1)"), and it does not establish that effect either;
- that the paper's Table 1 is wrong or irreproducible in its authors' environment;
- that Table 1 was produced with a non-causal prefill;
- which cause explains the paper's baseline rows (D1 makes hypothesis (a) more plausible but leaves it untested);
- how mask and kernel effects divide over a closed-loop episode (D1-a is step 0 only);
- anything about TTF's absolute success relative to the paper's TTF row;
- per-suite effects.

**Never to be written from this result:**
- any general claim that TTF is without effect, ineffective or harmful to OpenVLA;
- that the paper's result is false or irreproducible;
- any description of this run as a faithful or exact reproduction;
- a single named cause of the baseline gap;
- a relabelled headline such as "inconclusive". The label of record is unchanged.

## 10. Process, retries and amendments (summary; full disclosure in `results/RESULTS_ADDENDUM.md`)

**Validity.**
- 2,400/2,400 episodes are valid, with 800 unique keys per arm and identical keys in all three arms.
- Retries were infrastructure-only. 30 of 120 arm-shards used one or two retries, with a maximum of 2 per arm-shard
  (the E3 unit ruled by critic-ttf-e3, `56d68d51…`).
- The earliest valid record is kept for each episode, and 219/219 replays were identical.

**Incidents.**
- I1: host simulator wedge.
- I2: defective cap projection, repaired under review.
- I3: v2 scheduler deadlock.
- I4: duplicate claim; nothing was duplicated.
- I5: user-directed GPU kill and reboot.

**Amendments.**
- A1: cap 55 → 65 GPU-h, cost-only.
- A2: approved, not used.

**Exposure.** No in-run pairing included ttf, so success counts were first viewed after the seal.

**Bias.** critic-ttf-final found no path by which these events could bias ttf − pb.

## 11. Cost and session accounting (C5)

**This run's GPU cost.**
- Primary: 207,177 GPU-s = **57.55 GPU-h**, under the A1 cap of 65, including all failed attempts.
- X1 and gate wait: 0.82 GPU-h.
- Pre-checks: 1.93 GPU-h.
- I5 health smoke: 170 GPU-s, in a separate ledger.
- **TTF total: about 60.35 GPU-h of the run's 80 GPU-h authority.**
- All-in cost: 0.299 GPU-s per policy step.
- Paid API $0; NRP calls 0.

**The rest of the authority.** These are the v2 figures, at the seal:
- About 19.65 GPU-h remained.
- Pigey's share under A1 C2 was ≤ 12 GPU-h; its SETTING v3 allowance was 12.25. Pigey had used 0 GPU-h.
- That left about 7.4-7.65 GPU-h unallocated.

**v3 update (D1).**
- D1 used **3.88 GPU-h**: D1-a 0.645 (including the smoke) and D1-b 3.235, from lease GPU-s
  (`d1/cost/lease_ledger.jsonl`), within d07's ~3-5 GPU-h and d08's caps. $0 API; 0 NRP calls.
- **Session total: about 64.2 GPU-h** (60.35 + 3.88). The user raised the authority from 80 to about 92 GPU-h for
  TTF-C (authority-amendment-2, 2026-11-06T06:14:21Z), so **about 27.8 GPU-h remain**; decision d11 sets TTF-C's cap
  at or above its outcome-proof worst case even if the session total then exceeds ~92 GPU-h (actual spend reported).
- Pigey was dropped by decision d10 after using 0 GPU-h.
- TTF-C's budget is set at its own freeze. (v5: FROZEN-C set it at 55 GPU-h; TTF-C has since run, see below.)

**v5 update (TTF-C; CF-6).**
- TTF-C ran and was sealed (section "TTF-C" after "Diagnostics (D1)"). It used **14.30 GPU-h** (51,495 GPU-s:
  47,173 s scored-scheduler leases and 4,322 s gates, including the failed G-C3 r0), from lease GPU-s
  (`ttfc/sealed/lease_ledger.jsonl`, `ttfc/sealed/results.json` `cost`), under its 55 GPU-h cap. $0 paid API; 0 NRP
  calls.
- **Listed separately, not charged and not in the totals below:** 2,054 s (2,054.1 GPU-s) of idle-held reservations
  on GPUs 4-6 under decision d16 (holds, 09:06:11Z to 09:25:38Z; C-A5-a).
- **Session total: about 78.5 GPU-h** (64.2 + 14.30) of the raised ~92 GPU-h authority (authority-amendment-2), so
  **about 13.5 GPU-h remain**. TTF-C cost less than the 33.4-35.1 GPU-h expected at its freeze (FROZEN-C) mainly
  because the user cut its scope from 800 to 200 pairs (d15).

## 12. Follow-ups (C6): each one is a new frozen experiment

- **Nothing here may alter the sealed labels.**
- **Every stock-causal or causal-TTF run is a new experiment:**
  - separately frozen and independently reviewed;
  - reported as descriptive or as a new claim;
  - with the exposure of init states 0-19 disclosed;
  - inside the remaining authority or a user-approved increase.
- **A TTF-on-causal-decode variant is not the release.** It would be labelled `tf_variant`/`adapted`, with matched
  causal baseline and shadow arms, a new freeze, and preferably fresh init states (for example 20-39).

| Option (critic-ttf-final §5) | What it tests | Approximate GPU-h |
|---|---|---|
| P6 analogue: the authors' unmodified script in our env on a subset, or a recorded-frame replay | real-weight port defect (c) | 0.5-2 |
| step-0 decode-variant probe on the 800 reset observations (analyst) | mask vs kernel; the size of the first-action effect | ≈ 0.2 |
| eager-causal arm, one suite | missing mask vs kernel | 2-3 |
| paired stock/native causal baseline, one suite (Object / Spatial) | baseline-gap attribution in our env | 2.8 / 2.1 |
| same, all 800 cases | same, all suites | ≈ 14 |
| TTF-on-causal study, 3 arms × 800 | TTF itself in the strong-baseline regime | 40-45, or ≈ 10 per suite |

The last option does not fit the remaining authority without a user budget increase. (v3: the user has since raised
the authority to about 92 GPU-h for TTF-C, authority-amendment-2. v5: TTF-C ran at 200 pairs after the user's scope
cut, d15; see the section "TTF-C".)

**Status (v3).**
- **D1 is done** ("Diagnostics (D1)" below). It covers two rows only partially:
  - the step-0 decode-variant probe, done in full;
  - "eager-causal arm, one suite": mask vs kernel was measured at step 0 only, not in closed loop;
  - "paired stock/native causal baseline": 200 cases, init states 0-4, all suites, not 800.
- **The P6 analogue is not done.**
- **TTF-C (decision d10)** is a separate frozen experiment (v3: still under design): TTF on a causal decode, a
  disclosed variant of the release. **v5: it ran** (2026-11-06, 07:39Z to 11:56Z) and was sealed at 11:57:20Z
  (`ttfc/sealed/SEAL.json` `9f273f8ba24bbe40…`). Its result is reported only in the section "TTF-C: TTF on a causal
  decode (separate decode-variant experiment)" below, and it leaves the sealed labels unchanged.
  - It needs its own matched causal baseline and shadow arms with its own decode numerics, because per D1-a, eager vs
    SDPA already changes 8.9% of first actions.
  - **The D1-b native arm is not TTF-C's baseline.**
  - TTF-C must disclose the exposure of the sealed init 0-19 outcomes and of the D1-b native outcomes on init
    states 0-4.

## Diagnostics (D1): decode-path checks (descriptive; changes no label)

D1 is a separate experiment under critic-ttf-final C6 (SETTING-D1 rev 3 `fe2de44015bec374…`, frozen
2026-11-06T03:57:15Z; FROZEN-D1.rev3 `aa76af2f4a696378…`), independently recomputed by critic-ttf-d1-results. It
changes nothing above. The label of record stays **not reproduced (non-matching baseline regime; adapted)**, and
`sealed/labels.json` (`ce0f70f8131a244b…`) is unchanged.
- **Exposure:** the init-state 0-19 outcomes of the sealed arms were known before D1. The D1-b cases (init states
  0-4) were fixed by position before any native outcome.
- **Cost:** 3.88 GPU-h (D1-a 0.645, D1-b 3.235, lease GPU-s); $0 API.

**Sources (sha256, first 16 hex digits).**

| File | sha256 |
|---|---|
| `d1/SETTING-D1.md` = `d1/control/SETTING-D1.frozen.md` (rev 3) | `fe2de44015bec374` |
| `d1/control/FROZEN-D1.rev3.json` | `aa76af2f4a696378` |
| `d1/d1a.json` (D1-a analysis) / `d1/d1a.native_xcheck.json` | `bbafbed0142d5314` / `ffd5339e17071cdc` |
| `d1/d1b.json` (D1-b analysis) | `77083792d0c038bd` |
| `d1/R2_reference.json` | `e43c75380508c507` |
| `d1/b/native_assembled.jsonl` (200 native records) | `22038e8dc13b6f50` |
| `d1/probe/all_probe.jsonl` / `d1/cost/lease_ledger.jsonl` | `0673f1f7a3c9afd1` / `d9194101ea21ae15` |
| `reviews/critic-ttf-d1-results.md` / `.review.json` (ACCEPT WITH CONDITIONS, agent a927ad86246cd4624, 2026-11-06T06:28Z) | `fe94642b22665542` / `efaec3f718b99091` |

**D1-a: the release's decode and OpenVLA's first action (800 scored start states, no rollouts).** On every sealed
start state (observation identical to the sealed paper-baseline step-0 input, and release-decode tokens identical to
the sealed paper-baseline tokens, 800/800), three decodes of the first action were compared:
- v1: the release's `generate(output_attentions=True)` under transformers 4.40.1, eager attention without a causal
  mask;
- v2: OpenVLA's native causal SDPA decode;
- v3: eager attention with the causal mask.

| Pair | What differs | All 7 tokens equal | First token equal | Gripper token equal |
|---|---|---|---|---|
| v1 vs v2 | mask and kernel | 15/800 (1.9%) | 144/800 (18.0%) | 765/800 |
| v1 vs v3 | causal mask only | 14/800 (1.8%) | 145/800 (18.1%) | 765/800 |
| v2 vs v3 | kernel only (eager vs SDPA, both causal) | 729/800 (91.1%) | 773/800 (96.6%) | 800/800 |

- **What the mask does:**
  - the missing causal mask changes OpenVLA's first action in about 98% of start states;
  - on the real weights, the release's prefill attends to later positions in every layer, while the eager-causal
    decode puts zero attention there (checked on one start state per task, 40/40 tasks);
  - the release decode also turns the gripper command to "close" at step 0 in 35/800 start states.
- **What the kernel does:** eager-vs-SDPA numerics alone change the first action in 71/800 (8.9%), never the
  gripper. So the mask accounts for most, but not all, of the step-0 change.
- **The port:** restoring the mask on the release's own code path reproduces the native first action in 729/800.
- **Limit:** this is step 0 only. How the two effects compound over an episode was not measured.

**D1-b: native causal OpenVLA vs the sealed paper-baseline on the same 200 scenes.** The cases are 4 suites × 10
tasks × init states 0-4. Both arms ran on the same server build, configs, Docker image and env_seed 0. The first
observation was identical in 200/200, and all 200 native episodes were valid on the first attempt, with official
horizons.

| Suite | Native causal | Paper-baseline (sealed, same cases) | D (pp) | 95% CI (pp) | Native Wilson 95% | Paper's baseline row | vla-eval stock, init 0-4 (R2) / 0-19 (R1) |
|---|---|---|---|---|---|---|---|
| Spatial | 41/50 (82%) | 17/50 (34%) | +48.0 | [29.2, 62.1] | [69.2, 90.2] | 82.0 | 82 / 85.0 |
| Object | 42/50 (84%) | 23/50 (46%) | +38.0 | [20.2, 52.6] | [71.5, 91.7] | 66.5 | 90 / 86.0 |
| Goal | 42/50 (84%) | 18/50 (36%) | +48.0 | [31.4, 60.7] | [71.5, 91.7] | 77.0 | 78 / 77.0 |
| Long | 23/50 (46%) | 2/50 (4%) | +42.0 | [26.0, 55.9] | [33.0, 59.6] | 48.0 | 52 / 53.5 |
| **Pooled** | **148/200 (74.0%)** | **60/200 (30.0%)** | **+44.0** | **[35.6, 51.3]** | [67.5, 79.6] | 68.4 | 75.5 / 75.4 |

- **Statistics:** the cells are n11/n10/n01/n00 = 52/96/8/44, with n10 = native-only success; exact McNemar
  p = 2.8e-20. The CI is Newcombe (1998b) method 10, from the sealed primary's frozen label script. Per-suite rows are
  descriptive.
- **References:** the vla-eval stock references are unpaired and only comparable (another harness version,
  round-robin shards).
  - R1 is the predeclared rate over `episode_id` < 20: 85.0 / 86.0 / 77.0 / 53.5.
  - R2 is the predeclared rate over `episode_id` < 5: 82 / 90 / 78 / 52.
  - Both are unpaired and comparable, not paired.
  - `R2_reference.json`'s field `R1_rate` (86.6 / 86.6 / 79.0 / 52.8) is over all 50 init states. It is not the
    predeclared R1, and it is not used here.
- **A-5 cross-check:** the frozen analysis reported its native-vs-probe cross-check as not comparable (200/200),
  because the script looks for `sample` events while native traces log `native` events. A post-hoc critic check found
  that the native arm's first action equals the D1-a v2 action, and the sealed paper-baseline's first action equals
  the v1 action, in 200/200 cases. That check is post hoc and outside the frozen analysis.
- **Exposure:** the D1-b native outcomes on init states 0-4 are now exposed as well. Any later experiment on these
  cases (TTF-C) must disclose that.

**What D1 shows.**
1. **Within our stack, the release's decode path accounts for paper-baseline's gap to stock OpenVLA.** On identical
   scenes, switching to the native causal decode raises success from 30.0% to 74.0%. The native rates are in line
   with the stock references. "Decode path" here means the missing causal mask together with eager-vs-SDPA numerics:
   D1-a attributes most of the step-0 change to the mask, and D1-b does not separate them over an episode.
2. **The causal decode is consistent with the paper's baseline rows on Spatial, Goal and Long, but not on Object.**
   - On Spatial, Goal and Long, each paper value lies inside the native arm's Wilson 95% interval (n = 50 per suite,
     unpaired, init states 0-4). Intervals this wide cannot establish a match.
   - On Object, the paper's 66.5 lies below the native interval [71.5, 91.7]. Native Object (84%) sits next to stock
     (86-90%), so the paper's Object row stays unexplained.
3. **Hypothesis (a) becomes more plausible but stays untested.** The release's decode call drops the causal mask
   under its pinned transformers 4.40.1 (and 4.40.2) and keeps it from 4.41.2 on (CPU check, SETTING-D1 §3). Together
   with point 2, this makes (a) of §8, that the authors' effective runs decoded causally, more plausible. It stays
   untested: D1 observes nothing of the authors' environment, and hypothesis (b) is not tested either.

**What D1 does not show.**
- **TTF.** It does not test TTF on a causal decode. TTF-C (decision d10), a separate, newly frozen experiment with its
  own causal baseline and shadow arms, has since run (v5); see the next section.
- **The sealed result.** It does not change, replace or qualify away the sealed paired effect (D = −0.62 pp
  [−3.92, +2.67]) or its label.
- **The paper.** It does not establish what produced the paper's Table 1, or that the paper's numbers are wrong.

## TTF-C: TTF on a causal decode (separate decode-variant experiment)

**Status: ACCEPTED WITH CONDITIONS by critic-ttfc-final.** The review is `reviews/critic-ttfc-final.md`
(`c4238f9894a6ec7d…`) with `reviews/critic-ttfc-final.review.json` (`a5e63bc2c5e1ac89…`), by an rmh_critic that wrote
none of the TTF-C setting, schedulers, label or seal tools, or earlier TTF-C reviews (2026-11-06T12:16Z). Its
conditions CF-1..CF-8, required disclosures, must-not-state list and D1-b ruling bind this section.
`results/v5_cf_checklist.md` maps each of them to the lines that meet it.

- **Separate experiment.** TTF-C was frozen separately under critic-ttf-final C6 (SETTING-C `57e69ab0c0561c09`,
  frozen 2026-11-06T06:50:44Z by critic-ttfc-1; FROZEN-C `a9247ab9e6805fb1`; AMENDMENT-C1 as approved, `0f24ec0c367fa681`)
  and sealed separately (`ttfc/sealed/SEAL.json` `9f273f8ba24bbe40…`, 19 files, 2026-11-06T11:57:20Z, evaluator
  ac7c496a55d231956).
- **Primary:** The sealed primary label ('not reproduced (non-matching baseline regime; adapted)') is unchanged;
  TTF-C appears in its own section and never in the primary's table or headline. critic-ttfc-final found the primary
  seal (`dda8177cb4fe05cf…`, 17 files) and `sealed/labels.json` untouched.
- **Numbers:** every number in this section is taken from `ttfc/sealed/` (`RESULTS.md`, `results.json`,
  `ttfc_labels_c1.json`) or from critic-ttfc-final's `recomputed` block, at the sealed rounding.

### Result (CF-1: the approved results sentence, verbatim)

> TTF-C (TTF-VLA on a causal eager decode; a separate decode-variant experiment that leaves the sealed primary label "not reproduced" unchanged): consistent with the paper, not established (causal decode; baseline rows differ from the paper's on libero_object); adapted, decode variant; 200-pair subset (init 0-4 of all 40 tasks) chosen by the user before any TTF-C outcome was read (d15); D = +1.00 pp (ttfc 152/200 vs ttfc-baseline 150/200), Newcombe-10 95% CI [-5.51, +7.51], exact McNemar p = 0.88, against the paper's +4.0 pp, at a power of 0.14-0.18 for a true +4.0 pp.

The label comes from the pinned chain: wrapper `ttfc/analysis/ttfc_labels_c1_c43ef03e.py` (`c43ef03e9a08ee20…`, the
AMENDMENT-C1 scope filter by case key), then the frozen `ttfc/analysis/ttfc_labels.py` (`cd43d106e07cb1b3…`), then the
primary's pinned `analysis/ttf_labels.py` (`b334a24e40919727…`), with the primary's rule, targets, Newcombe method-10
CI, exact McNemar test and task-clustered bootstrap unchanged. critic-ttfc-final reran the pinned chain on the sealed
records (output byte-identical to `ttfc_labels_c1.json`, `0ec0b67359bba660…`) and recomputed every number
independently (agreement to 1e-12).

| TTF-C block | Pairs | ttfc-baseline → ttfc | Cells n11/n10/n01/n00 | D (pp) | Newcombe-10 95% CI (pp) | McNemar p | Target | TTF-C label of record (sealed) |
|---|---|---|---|---|---|---|---|---|
| Headline (4 suites, init 0-4) | 200 | 150 → 152 | 129/23/21/27 | 1.00 | [-5.51, 7.51] | 0.8804 | +4.0 | consistent with the paper, not established (causal decode; baseline rows differ from the paper's on libero_object); adapted, decode variant |
| Secondary (Spatial/Goal/Long; never overrides the headline) | 150 | 103 → 107 | 85/22/18/25 | 2.67 | [-5.57, 10.85] | 0.6358 | +10/3 | consistent with the paper, not established (S/G/L baseline rows consistent with the paper's); adapted, decode variant |

- n10 counts the pairs where only ttfc completed the task. The task-clustered bootstrap 95% interval (a sensitivity
  check) for the headline is [-6.5, 8.5].
- **The rule at the +4.0 pp target:** L = -5.51 ≤ 0 ≤ U = 7.51 and U ≥ 4.0, so the label is "consistent with the
  paper, not established". The interval contains both zero and the paper's +4.0 pp: this result neither establishes
  nor excludes the paper's effect in the causal-decode regime.
- **The secondary block** (Spatial, Goal and Long, whose baseline rows are all consistent with the paper's) gives the
  same label as the headline.

### Scope (d15) and power

- **Scope:** The user cut TTF-C from 800 to 200 pairs (init 0-4 of all 40 tasks; compute-matched shadow null on init
  0-3, 160 cases) at about 08:41Z, before any TTF-C outcome was read. Before choosing, the user was shown that if the
  paper's +4.0 pp is real, the chance of a 'reproduced' label was about 48 % at 800 pairs, 28 % at 400 and 16 % at 200.
  Episodes with init >= 5 that ran before the cut (159 ttfc-baseline and 139 ttfc trace files) were excluded by case
  key, never used, and will not be used to extend the scope.
  - "Excluded by case key" means that each out-of-scope trace file was parsed only to obtain its key; no out-of-scope
    field entered any output (critic-ttfc-final F10).
  - The question, the options shown and the answer are recorded in `ttfc/control/USER-scope-cut-200.json`
    (`5db6247f200d93c8…`). The user was also told before choosing that the predeclared rule, not the choice of
    subset, assigns the label.
- **C-A1-a stays in force (CF-7):** init >= 5 records are never used; any 400- or 800-pair extension is a new
  experiment, with this exposure disclosed.
- **Power:** Exact McNemar power for a true +4.0 pp at 200 pairs was 0.179 (discordance 0.23) and 0.137 (0.33),
  computed at 08:43Z, after the user's cut and before any TTF-C outcome was read (`ttfc/analysis/power_c1.json`,
  `f69a01e2c571d578…`, written 08:43:25Z). 'Consistent with the paper, not established' was the most likely label
  whatever the true effect. Observed discordance: 44/200.

### Per suite (descriptive; 50 pairs each)

| Suite | Cells n11/n10/n01/n00 | ttfc-baseline → ttfc | D (pp) | 95% CI (pp) | McNemar p | Target | Frozen rule's per-suite output (descriptive only) |
|---|---|---|---|---|---|---|---|
| Spatial | 38/5/3/4 | 41 → 43 | 4.00 | [-7.57, 15.80] | 0.727 | +2.5 | consistent with the paper, not established |
| Object | 44/1/3/2 | 47 → 45 | -4.00 | [-14.09, 5.01] | 0.625 | +6.0 | not reproduced |
| Goal | 30/6/7/7 | 37 → 36 | -2.00 | [-15.96, 12.04] | 1 | +2.0 | consistent with the paper, not established |
| Long | 17/11/8/14 | 25 → 28 | 6.00 | [-10.66, 22.13] | 0.648 | +5.5 | consistent with the paper, not established |

- **Per suite:** Per-suite deltas (+4.0 / -4.0 / -2.0 / +6.0 pp, 50 pairs each) are descriptive and not significant;
  the Object and Goal decreases are not claimed as effects.
- The Object per-suite output ("not reproduced", because U = 5.01 < 6.0) is descriptive like every per-suite output.
  It carries no label, and it sits on the suite with the least headroom (next subsection).

### Baseline rows next to the paper's (Wilson 95%, n = 50)

| Suite | ttfc-baseline | Wilson 95% (%) | Paper baseline (Table 1, %) | Baseline row |
|---|---|---|---|---|
| Spatial | 41/50 | [69.2, 90.2] | 82.0 | consistent |
| Object | 47/50 | [83.8, 97.9] | 66.5 | differs (predeclared, SETTING-C §5.3) |
| Goal | 37/50 | [60.4, 84.1] | 77.0 | consistent |
| Long | 25/50 | [36.6, 63.4] | 48.0 | consistent |

- **Baseline rows:** At n = 50 per suite, the causal baseline is consistent with the paper's baseline row on Spatial
  (41/50), Goal (37/50) and Long (25/50), with wide intervals, and differs on Object (47/50 vs 66.5 %, predeclared),
  where little headroom remains for the paper's +6.0 pp.
- "Consistent" means that the paper's value lies inside a wide n = 50 interval. It is overlap, not equality.

### Variant and protocol

- **Variant:** Adapted, decode variant: every arm decodes with an eager-attention copy of the same frozen checkpoint
  under the full causal mask. This is neither the authors' release, whose decode is non-causal under its pin, nor
  stock SDPA OpenVLA. It tests the paper's claim in a regime where three of four baseline rows are consistent with the
  paper's; it does not establish what the authors ran.
- **Arms** (`hs eval … --arms candidate`, profile `openvla-libero-ttf`, contract `libero`):
  - ttfc-baseline (mode off: stock tokens, causal eager decode), the baseline of every comparison;
  - ttfc (mode ttf: hybrid fusion, K = 3, τ = 0.03, top-k 70, LLM layer-15 text attention);
  - ttfc-shadow (mode shadow: the full TTF computation, stock tokens to the projector), the compute-matched null.
- **Method:** `ttfc/method` = hs-ttf-c 0.1.0 (MANIFEST `45d5e5aba1ce5b4d…`), fingerprints `96a558a0…` / `0fb12b36…`
  / `48c111fb…` (ttfc-baseline / ttfc / ttfc-shadow).
  - The fusion files are byte-identical to the primary's hs_ttf 0.1.0. The only change is the hooked decode: the D1-a
    v3 call on an eager copy of the same pinned checkpoint, without the processor's short attention mask.
  - The server is a run-local subclass of the EXT server. There is no framework change: every scored attempt ran on
    harnessstudio HEAD `07c64228` (tree `6360a96c…`), and the 41 configs used equal `configs/primary/` (MANIFEST
    `f4e4288571b71790…`).
- **Benchmark:** the primary's (§6): seed 7, env_seed 0, `num_steps_wait` 10, official horizons 220/280/300/520. Every
  failure ran to its official horizon. The any-step task-completion signal stays evaluator-only, and the public
  inputs are image and language only.
- **Episodes per arm-shard:** paired arm-shards ran 20 episodes before the d15 cut and
  `--benchmark-overrides '{"episodes_per_task": 5}'` after it; the shadow always ran with 4.
- **Not TTF-C's baseline:** the D1-b native SDPA arm (another kernel) and the sealed paper-baseline (non-causal). Both
  appear here only descriptively.
- **Cap:** 55 GPU-h (FROZEN-C), set above the outcome-proof bound of 51.2 GPU-h so that the cap could not bind through
  outcomes.
- **AMENDMENT-C1** (resource placement and scope; `ttfc/control/AMENDMENT-C1.final.md`) is cited "as approved". The
  file is byte-identical to the proposed file and its header still says "proposed"; its finality comes from
  critic-ttfc-c1 (`02c1aa2cae3686b9…`) and decision d17 (critic-ttfc-final F11).

### Null control and activation

- **Null and activation:** Shadow null M 160 / R 0 / X 0 in-run and in the full pairing; libero_goal_t2's shadow ran
  on another GPU (rule (a)) and matched 4/4. TTF was active: the reused visual-slot rate on all in-scope ttfc steps was
  43.2 / 45.8 / 44.9 / 44.4 % (Spatial / Object / Goal / Long), vs the paper's 41.2 / 44.6 / 43.5 / 42.0. Fallback,
  keyframe-phase and zero-reuse descriptives and the ttfc-baseline vs sealed paper-baseline comparison were not computed
  in the sealed analysis.
- **The null, four ways:** the 40 in-run per-task pairings, the full `hs pair` (160 cases, no invalid record), the
  label script and critic-ttfc-final's independent hash-stream recomputation each give M 160 / R 0 / X 0. The shadow
  paid TTF's fusion computation on every policy step while executing native actions.
- **Fusion rates** (sealed `ttfc_labels_c1.json` `activation`; step-pooled over the 42,028 in-scope ttfc steps only):

| Suite | ttfc reused visual-slot rate (in scope) | Paper | Flagged (outside ±10 pp) | Decode violations | Nonzero upper mass |
|---|---|---|---|---|---|
| Spatial | 43.2 % | 41.2 % | no | 0 | 0 |
| Object | 45.8 % | 44.6 % | no | 0 | 0 |
| Goal | 44.9 % | 43.5 % | no | 0 | 0 |
| Long | 44.4 % | 42.0 % | no | 0 | 0 |

- **The actions diverged:** ttfc's action stream differs from ttfc-baseline's in 200/200 episodes, and the first
  divergence always comes at a non-keyframe step (critic-ttfc-final recomputation).

### Validity

- 560/560 required episodes are valid: 200 ttfc-baseline, 200 ttfc and 160 ttfc-shadow. Keys are unique, and episodes
  are exactly 0-4 (shadow 0-3) of all 40 tasks. Every termination is "environment".
- 120/120 arm-shards are done with one attempt each, so there were **no E3 retries**. The four attempts voided under
  d16 contribute only their complete in-scope records (10 ttfc-baseline, 10 ttfc, 0 ttfc-shadow); none was relaunched.
- **V2** (one hooked activation record per policy step) passed for all 560 episodes over 12 scored server logs,
  including the GPU-7 log `g7-libero_goal-0`. **V3** (the `eager_causal_v3` decode tag, zero attention mass above the
  diagonal, 12/12 server self-checks) passed.
- Every server showed the frozen startup probe, the frozen libraries and the frozen policy JSON hash.

### D1-b comparison (descriptive; critic-ttfc-final ruling)

- **D1-b:** Descriptive only, no label (quoted from the sealed ttfc_labels_c1.json; omitted from results.json/RESULTS.md
  by a key mismatch in the frozen seal tool): on the same 200 cases the eager-causal ttfc-baseline scored 150/200 and
  the earlier D1-b native SDPA-causal arm 148/200 (both 123, ttfc-baseline only 27, native only 25, neither 25). D1-b
  outcomes were known before TTF-C; this compares decode kernels in closed loop across separate runs and is not a TTF
  effect.
- **The seal-tool key mismatch:** `ttfc/eval/seal_c1.py` line 54 reads the key `native_d1b`, but the label script
  writes `descriptive_d1b_native_vs_ttfc_baseline`. So the sealed `results.json` has `native_d1b: null`, and the
  sealed `RESULTS.md` omits the comparison. The label is unaffected: critic-ttfc-final reran the pinned chain without
  the D1-b input, and the output differs only in that descriptive key. There is no reseal.

### Run history: d14, d15, d16, K-5 and libero_goal_t2

All times are 2026-11-06 UTC. critic-ttfc-final checked that every mid-run decision preceded the first read of any
TTF-C outcome (11:56:51Z), using the transcripts, file birth times and atime evidence.

| Time (UTC) | Event | What was visible |
|---|---|---|
| 07:12:16Z / 07:27:14Z | d12 (launch route after the narrow review); d13 (after I-C1, rerun G-C3 as r1) | no TTF-C outcome existed |
| 07:39:46Z | scored run launched on GPUs 4-7 (scheduler `5ca404de`), 800 pairs planned | — |
| 08:36:56Z / 08:38:04Z | **d14**: the user asked to free GPU 7 (08:36:56Z); decision 08:38:04Z; resource only | validity progress only |
| 08:40:21Z to 08:42:15Z | **d15**: the user's question (08:40:21Z); the options were shown at 08:41:01Z, the answer "Cut to 200 pairs now" was recorded by 08:42:04Z, and its record file was written at 08:42:15Z. (`decision-d15.json` says "08:44Z approx"; the transcript times above are the precise ones, critic-ttfc-final F7) | the 08:39:55Z progress line only: shard counts, valid-episode counts, null M 12 / R 0 / X 0, GPU-s |
| 08:43:25Z | power at 200 pairs computed (`power_c1.json`) | no outcome read by anyone |
| 08:48:25Z to 08:51:38Z | **d16**: the user asked to kill the GPU-7 work and move it to GPUs 4-6. STOP 08:49:07Z; four GPU-7 attempts stopped 08:49:34Z to 08:51:36Z (I-C2); GPU 7 released 08:51:38Z | validity and monitor events |
| 09:06:11Z to 09:25:38Z | idle-held reservations (holds) on GPUs 4-6 under d16; no process ran under them | resource only |
| 09:08:32Z | **d17**: AMENDMENT-C1 bound as approved (critic-ttfc-c1); route (ii), scheduler S3 `8c6388e6` | review and validity lines |
| 09:18:08Z; stops 09:18:27Z to 09:20:29Z | **d18 / K-5**: approved stop (critic-ttfc-k5) of the out-of-scope tails of four GPU-5 libero_10 attempts, each holding 5/5 valid in-scope episodes when checked | in-scope validity and wall time |
| 09:21:18Z | resume on GPUs 4-6 (scheduler `8c6388e6`) | — |
| 10:34:16Z | finalize and seal tools frozen (`finalize_c1.py` `fd09c73d…`, `seal_c1.py` `ca0538d7…`, mode 0444) | dry test printed counts only |
| 11:20:19Z / 11:29:52Z | **libero_goal_t2, rule (a)**: its baseline and ttfc ran on GPU 7 (void under d16, in-scope records kept); its shadow ran once on GPU 5 (`rebound_from: 7`) and paired M 4 / R 0 / X 0. It is the only task whose arms ran on two GPUs | null attribution only |
| 11:56:22Z | scheduler end (no STOP) | — |
| 11:56:39Z / 11:56:51Z | finalize stage A passed (outcome-free); stage B, the first outcome read | — |
| 11:57:20Z | seal | — |

- At 09:15:52Z the evaluator surfaced per-shard elapsed minutes and episode counts for the four GPU-5 attempts, a
  wall-time proxy. It informed only the K-5 resource stop, which removed out-of-scope episodes and could not change any
  in-scope record (critic-ttfc-final F9, advisory).

### Incidents

- **Incidents:** I-C1 foreign-process OOM during the non-scored G-C3 r0 (archived; r1 passed); I-C2 user-directed kill
  of four GPU-7 attempts (void for retries; their complete in-scope records used); K-5 approved stop of out-of-scope
  tails of four libero_10 attempts (not infrastructure failures); no E3 retries; no genuine infrastructure failure in
  scored work.
- The 22 incidents in the scheduler state reconcile as 8 `job_infrastructure_failure` entries (4 from I-C2, 4 from
  K-5), 4 `attempt_void_d16`, 1 `task_rebind_pending_d16`, 8 `scope_c1_done_from_existing` and 1
  `task_rebound_missing_arms_d16`. None is a robot failure.
- Not in the sealed `RESULTS.md` incident list: G-C0's literal venv check failed on two editable-install lines and was
  resolved before launch (critic-ttfc-narrow Q3-CL); a monitor-pipe delivery gap from 10:57Z to 11:27Z (recorded in the
  post-seal part of `run_notes.md`) left the validity checks unaffected.

### Cost (CF-6)

- **Cost:** 14.30 GPU-h (51,495 GPU-s: 47,173 s scored-scheduler leases, 4,322 s gates including the failed G-C3 r0),
  cap 55 GPU-h, $0 paid API. Separately, 2,054 s of idle-held reservations under d16, not charged.
- **Breakdown** (`ttfc/sealed/results.json` `cost`):
  - scored-scheduler leases 47,172.8 s: GPU 7 4,105.3 s; first scheduler generation on GPUs 4/5/6 16,396.2 s; second
    generation 26,671.3 s;
  - gate leases 4,322.4 s: G-C1 54.7 s; G-C3 r0 1,836.5 s (failed under I-C1, charged); G-C3 r1 2,431.2 s;
  - idle-held d16 holds 2,054.1 s (GPU 4 907.2 s, GPU 5 161.4 s, GPU 6 985.5 s), listed separately and **not charged**
    (C-A5-a).
- The charged time includes the 298 out-of-scope episodes that ran before the cut, the I-C2 GPU-7 work and the
  K-5-truncated tails. The ledger has no open interval. Session accounting: §11, "v5 update".

### Seal notes (CF-5)

- **Seal notes:** The seal binds run_notes.md 28590fba; a section appended at 11:58:05Z is outside the seal.
- `ttfc/eval/run_notes.md` is cited here as sealed `28590fba82468345…`, which is its first 4,376 bytes. The current
  file (`d04c50f65a8040e1…`) appends, 45 s after the seal, the section headed "2026-11-06T11:58:05Z S3 completion,
  finalize and seal". That section is **post-seal and not covered by the seal**. The change is append-only (checked by
  critic-ttfc-final F1). There is no reseal.

### Exposure

- Before TTF-C, the sealed primary outcomes on init states 0-19 and the D1-b native outcomes on init states 0-4 were
  known; FROZEN-C discloses both.
- The TTF-C outcomes of ttfc-baseline and ttfc on init states 0-4, and of ttfc-shadow on init states 0-3, are now
  exposed as well. Any later experiment on these cases must disclose that.

### What TTF-C can and cannot support

**Can be concluded:**
- On a causal eager decode of the same frozen checkpoints, with TTF active at reuse rates near the paper's, the paired
  difference ttfc minus ttfc-baseline on 200 pairs was D = +1.00 pp, 95% CI [-5.51, +7.51]. That interval contains both zero and the
  paper's +4.0 pp, so the TTF-C label is "consistent with the paper, not established". At a power of 0.14-0.18, this
  was the most likely label whatever the true effect.
- The compute-matched shadow null found no overhead or fusion-leak effect: in all 160 shadow cases (init 0-3) its
  per-step action and observation hashes equal ttfc-baseline's (M 160 / R 0 / X 0). Init state 4 had no shadow arm.

**Cannot be concluded:**
- that TTF raises OpenVLA's paired LIBERO outcome by the paper's +4.0 pp on a causal decode (not established);
- that TTF has no effect, or harms OpenVLA;
- any per-suite effect, including the Object and Goal decreases;
- what the authors ran, or why the paper's Object baseline row differs from ours;
- anything that changes the primary result: TTF-C does not relabel, replace or qualify the sealed primary.

**Never to be written from TTF-C** (critic-ttfc-final `must_not_state`): that TTF-C reproduced the paper's gain; that
any baseline row or result "matches" the paper; that it is a faithful reproduction; a TTF-C number stated as a claim
of task success for TTF; that the paper's baseline used a non-causal prefill; that any TTF-C number, or the Awomo
43.96 % (a different benchmark), is verified SOTA; that TTF-C relabels, replaces or qualifies the sealed primary; the
full `hs pair` candidate row (an init 0-3 denominator) as a result.

**Not quoted as values (CF-3):** the full `hs pair` candidate row (it uses the init 0-3, 160-case denominator, not the
label's); the per-mode activation record counts and the V3 record total (both include out-of-scope steps; they are
validity counts only); the sealed E3 replay-determinism field as determinism evidence (it is vacuous: no attempt
was compared, because there were no E3 retries).

### Re-running TTF-C (evaluator-owned; GPU)

- **Schedulers:** `ttfc/eval/ttfc_sched_5ca404de.py` (07:39:46Z to 09:20:39Z, stopped under d16) and
  `ttfc/eval/ttfc_sched_8c6388e6.py` (S3, 09:21:18Z to 11:56:22Z). Their launch records are sealed
  (`ttfc/sealed/launch_record.json`, `launch_record_resume_s3.json`).
- **Arm-shards:** each ran the §13 argv shape with the entry points `ttfc-baseline`, `ttfc` and `ttfc-shadow`, the
  run-local server launcher `ttfc/method/server/openvla_ttfc_hs.py` (policy JSONs `ttfc/servers/policy-<suite>.json`),
  and `--benchmark-overrides` as above.
- **Labels and seal:** `ttfc/eval/finalize_c1.py` (`fd09c73dbbbaf2db…`) assembles the in-scope records, checks
  V2/V3, runs the full `hs pair` and calls the pinned label chain; its exact label argv is in
  `ttfc/sealed/finalize.json`. `ttfc/eval/seal_c1.py` (`ca0538d7722cda89…`) writes the seal.

**TTF-C sources (sha256, first 16 hex digits).**

| File | sha256 |
|---|---|
| `ttfc/sealed/SEAL.json` | `9f273f8ba24bbe40` |
| `ttfc/sealed/RESULTS.md` / `ttfc/sealed/results.json` | `4455823b7b803299` / `eccbf01d1402357a` |
| `ttfc/sealed/ttfc_labels_c1.json` | `0ec0b67359bba660` |
| `ttfc/sealed/stage_a.json` / `finalize.json` / `pair_full_summary.json` | `e0746b1492736958` / `225c098e76db6183` / `3b1d3dbf090e89c9` |
| `ttfc/sealed/` records: ttfc-baseline / ttfc / ttfc-shadow | `81be175f720b9521` / `e2b29c209a579f8d` / `b2cd15a7ec3150cc` |
| `ttfc/sealed/lease_ledger.jsonl` | `cf4fd8e6e5cf936f` |
| `ttfc/control/SETTING-C.frozen.md` / `FROZEN-C.json` | `57e69ab0c0561c09` / `a9247ab9e6805fb1` |
| `ttfc/control/AMENDMENT-C1.final.md` (as approved) / `USER-scope-cut-200.json` / `L-U_user_notice.json` | `0f24ec0c367fa681` / `5db6247f200d93c8` / `bd87e82b1d395c2b` |
| `ttfc/analysis/power_c1.json` | `f69a01e2c571d578` |
| `ttfc/analysis/ttfc_labels_c1_c43ef03e.py` / `ttfc_labels.py` / `analysis/ttf_labels.py` | `c43ef03e9a08ee20` / `cd43d106e07cb1b3` / `b334a24e40919727` |
| `ttfc/eval/finalize_c1.py` / `seal_c1.py` / `k5_stops.json` | `fd09c73dbbbaf2db` / `ca0538d7722cda89` / `65b2b1bb7dfacb3c` |
| `ttfc/eval/run_notes.md`: sealed prefix (4,376 bytes) / current file (post-seal section appended) | `28590fba82468345` / `d04c50f65a8040e1` |
| `ttfc/method/MANIFEST.json` | `45d5e5aba1ce5b4d` |
| `control/research/decision-d14` / `d15` / `d16` / `d17` / `d18` `.json` | `d9b3efc98a2907c9` / `2165c7405a5a57ee` / `e6afc068ce07c485` / `c3be70f6e2a31128` / `1bd202c70c2fa6be` |
| `d1/b/native_assembled.jsonl` (D1-b input to the descriptive comparison) | `22038e8dc13b6f50` |
| `reviews/critic-ttfc-final.md` / `.review.json` | `c4238f9894a6ec7d` / `a5e63bc2c5e1ac89` |
| `reviews/critic-ttfc-1` / `-c1` / `-k5` / `-s3` / `-narrow` / `-f` `.review.json` | `8498197829cf389e` / `02c1aa2cae3686b9` / `7a17fcaaf9946a2e` / `6eea4e434ada6451` / `63156bab34108bd5` / `4d7fd59c57775111` |
| `reports/eval-ttfc.report.json` (the evaluator's report; its power wording and run_notes hash are superseded by this section, CF-4/CF-5) | `49d4810834adb778` |


## 13. Reproducing the scored run (evaluator-owned; GPU)

The method package is `harnessstudio/methods/ttf`: hs-ttf 0.1.0, entry points `ttf`, `ttf-paper-baseline` and
`ttf-shadow`, byte-identical to the scored `method/` per MANIFEST `a611af04…` at commit 8a24164; since then only one changelog string was appended to `hs_ttf/paper.yaml` and `README.md` was added (metadata; fingerprints unchanged).
- It needs the capability `vision_token_hooks` (EXT openvla-vision-token-hook, server
  `harnessstudio/policies/openvla_ttf_hs.py`, profile `openvla-libero-ttf`).
- The scored run imported the package from the run's `method/` with
  `PYTHONPATH=<harnessstudio checkout>:<method dir>`, and the fingerprints are the same from either location.
- **Exact argv**, as recorded in `sealed/sched_v2_state.json` and the server's `server.json`. Each policy server
  (one per GPU, 4 hs-eval workers each) ran in a clean env with `HF_HUB_OFFLINE=1`, `PYTHONSAFEPATH=1` and
  `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64`:

```json
["/home/sejain/.cache/uv/environments-v2/openvla-hs-73218eaab6c13068/bin/python",
 "<harnessstudio>/harnessstudio/policies/openvla_ttf_hs.py", "--port=8000", "--args.chunk_size=1",
 "--args.jpeg_roundtrip=True", "--args.center_crop=True", "--args.model_path=openvla/openvla-7b-finetuned-libero-10",
 "--args.unnorm_key=libero_10", "--args.activation_log=<server dir>/activation.jsonl", "--address", "127.0.0.1:<port>"]
```

One arm-shard (here ttf, libero_10 task 9 on GPU 6):

```json
["<run>/venvs/hs-ttf-verify/bin/hs", "eval", "ttf", "--arms", "candidate", "--policy-url", "ws://127.0.0.1:<port>",
 "--contract", "libero", "--profile", "openvla-libero-ttf", "--config", "<run>/configs/primary/libero_10_t9_g6.yaml",
 "--port", "<port>", "--action-timeout", "300", "--out", "<run>/eval/primary/runs/libero_10_t9/ttf/attempt0"]
```

- The other two arms use the entry points `ttf-paper-baseline` and `ttf-shadow`.
- The configs are the 160 files of `configs/primary/` (MANIFEST `f4e4288571b71790…`).
- Labels come from `eval/primary/tools/finalize.py` (`0e909041…`): it assembles the records, checks E6, runs the full
  `hs pair` and calls `analysis/ttf_labels.py`. The seal comes from `eval/primary/tools/seal.py` (`b20e394d…`).

## 14. Condition map (critic-ttf-final)

| Condition | Where it is met |
|---|---|
| C1 results addendum | `results/RESULTS_ADDENDUM.md` (new file; `sealed/` untouched) |
| C2 headline with qualifier, D and CI, four baseline rows, `adapted` | §1, §2 |
| C3 baseline gap, every cause an untested hypothesis | §8 |
| C4 Object wording | §4 |
| C5 session GPU accounting | §11; addendum §10 |
| C6 follow-ups are new frozen experiments | §12 |
| C7 qualified label in every table cell and README row | §1, §3; the `harnessstudio/README.md` row; `methods/ttf/README.md` |
| critic-ttf-d1-results C1 (own section, citations; sealed tables unchanged) | "Diagnostics (D1)" with its sources table; §1-§3 unchanged |
| critic-ttf-d1-results C2 (paper rows: "consistent with" / "not consistent") | "Diagnostics (D1)" point 2; §8 D1 update |
| critic-ttf-d1-results C3 (D1-a attribution wording; step 0 only) | "Diagnostics (D1)" D1-a; §9 |
| critic-ttf-d1-results C4 (R1 / R2; not the `R1_rate` field) | "Diagnostics (D1)" References |
| critic-ttf-d1-results C5 (A-5 frozen result + post-hoc check) | "Diagnostics (D1)" A-5 cross-check |
| critic-ttf-d1-results C6 (exposure, cost) | "Diagnostics (D1)" Exposure and Cost; §11 v3 update |
| critic-ttf-d1-results C7 (follow-ups; D1-b is not TTF-C's baseline) | §12 Status (v3) |
| critic-ttfc-final CF-1 (approved results sentence, verbatim) | "TTF-C", Result |
| critic-ttfc-final CF-2 (every required disclosure) | "TTF-C": Scope and power, Per suite, Baseline rows, Variant and protocol, Null control and activation, D1-b comparison, Incidents, Cost, Seal notes; the Primary bullet at its top |
| critic-ttfc-final CF-3 (no `hs pair` candidate row, per-mode activation counts or V3 total as values; "no E3 retries") | "TTF-C": Validity; "Not quoted as values" |
| critic-ttfc-final CF-4 (power computed at 08:43Z, after the user's cut and before any outcome was read) | "TTF-C": Scope and power |
| critic-ttfc-final CF-5 (`run_notes.md` sealed 28590fba = first 4,376 bytes; post-seal section marked) | "TTF-C": Seal notes; its sources table |
| critic-ttfc-final CF-6 (holds of 2,054 s reported separately) | "TTF-C": Cost; §11 v5 update |
| critic-ttfc-final CF-7 (C-A1-a: init >= 5 never used; any extension is a new experiment) | "TTF-C": Scope and power |
| critic-ttfc-final CF-8 (TTF-C only in its own section; primary table, headline and label unchanged) | "TTF-C" only; §1-§8 unchanged; only pointers in the header, §9, §12 and "Diagnostics (D1)", and the session cost in §11 (CF-6) |
| critic-ttfc-final line-level map | `results/v5_cf_checklist.md` |

## 15. Sources (sha256, first 16 hex digits)

| File | sha256 |
|---|---|
| `sealed/SEAL.json` | `dda8177cb4fe05cf` |
| `sealed/labels.json` | `ce0f70f8131a244b` |
| `sealed/results.json` / `sealed/RESULTS.md` | `fa392a8b278137f0` / `85032c03afdf9bb6` |
| `sealed/activation.json` | `47dcfed957af3037` |
| `SETTING.md` = `control/setting/SETTING.frozen.md` | `a6dc411f07124b1d` |
| `control/setting/FROZEN.json` (post-A2) | `b85669d1b933ac93` |
| `control/setting/AMENDMENT-A1.md` / `AMENDMENT-A2.md` | `37fa52cd95615c3d` / `a3b02efd00fdca7a` |
| `reviews/critic-ttf-final.md` | `d1170e947d48afbd` |
| `reviews/critic-ttf-e3.md` / `critic-ttf-a2.md` / `critic-ttf-i2.review.json` | `56d68d51c3d5451a` / `408e913e076a1adb` / `a8487a96f8298560` |
| `eval/primary/incidents.md` / `eval/primary/LAUNCH_RECORD.json` | `4324c3b637680a54` / `8e4e2e90353d3249` |
| `reports/eval-ttf-primary-1c.report.json` | `ed05f2a45bdd03b9` |
| `analysis/ttf_labels.py` | `b334a24e40919727` |
| `analysis/ttf_post_hoc.md` / `.json` (post-hoc, descriptive, file as of 23:42Z) | `c832da6f2fdcc2d5` / `82bdac71fac62570` |
| `method/MANIFEST.json` / `configs/primary/MANIFEST.json` | `a611af04503c8922` / `f4e4288571b71790` |
| D1 files (v3) | listed in the sources table of "Diagnostics (D1)" |
| TTF-C files (v5) | listed in the sources table of "TTF-C: TTF on a causal decode" |

## 16. Changelog

- v1 (2026-11-05, conv-ttf-results-1): first results document, written after the seal and the final review. The card
  `methods/ttf/hs_ttf/paper.yaml` gets one appended changelog line, which is metadata only and leaves the fingerprints
  unchanged, following the STAR precedent.
- v2 (2026-11-06, critic-ttf-docs D1/D2; no number, label or conclusion changed):
  - D1: §6 and §13 no longer call `harnessstudio/methods/ttf` byte-identical to the scored `method/` at the pushed
    commit. It is byte-identical per MANIFEST `a611af04…` at commit 8a24164; the later docs commit appended one
    changelog string to `hs_ttf/paper.yaml` and added `README.md` (metadata; fingerprints unchanged).
  - D2: the header's writing window is the actual `date -u` window, 23:40Z to 23:58Z (v1 said "23:40Z to 00:30Z",
    an estimate). `results/RESULTS_ADDENDUM.md` (sha256 `a7e00d39a491d085…`) has the same estimated window
    "between 23:40Z and 00:30Z" in its header; it is left unedited because reviews cite it by hash, and its actual
    window is likewise 23:40Z to 23:58Z.
  - v1 is kept unchanged as `results/ttf_vla.v1.md` (sha256 `d8b1a0be7a8c00bd…`), cited by hash in
    critic-ttf-docs and as evidence of playbook lesson 20261105T235020Z. This file cannot contain its own hash;
    the v2 sha256 is in the commit message that adds it and in `reports/conv-ttf-results-1.report.json`.
- v3 (2026-11-06, conv-ttf-results-2, critic-ttf-d1-results C1-C7; no sealed number, table or label changed):
  - New section "Diagnostics (D1): decode-path checks (descriptive; changes no label)" after §12. It is the critic's
    wording block, plus the D1 sources table, the R1/R2 definitions (not `R2_reference.json`'s `R1_rate`) and the
    D1-b exposure note.
  - §8: a D1 update. It gives paired support within our stack, says (a) is more plausible but untested and (c) is
    weakened for step 0 only, and that Object is not consistent. All causes remain untested hypotheses.
  - §9: one can-conclude item and two cannot-conclude items added.
  - §11: D1's 3.88 GPU-h added; session about 64.2 GPU-h of the raised ~92 GPU-h authority (authority-amendment-2),
    about 27.8 left; TTF-C's cap per d11; Pigey dropped (d10).
  - §12: status; TTF-C (d10) under design; D1-b is not TTF-C's baseline.
  - §14: D1 condition rows. §15: a pointer to the D1 sources. The header notes v3.
  - The headline, the §1 table, §2, the §3 labels and every C7 wording are unchanged.
  - v2 (`d770c4356953a3a3…`) and v1 (`d8b1a0be7a8c00bd…`) are kept unchanged in `results/`. This file cannot
    contain its own hash; the v3 sha256 is in the commit message and in `reports/conv-ttf-results-2.report.json`.
- v4 (2026-11-06, conv-ttf-results-2, critic-ttf-docs-v3 F1/F2; no number, label or conclusion changed; diff
  `results/ttf_vla_v3_to_v4.diff`):
  - §8 and §9: the D1 update says that the native vs paper-baseline comparison is paired while the stock references and
    the paper's rows stay unpaired, and adds native 74.0% in line with the unpaired stock references (75.4-75.5%).
  - §11 and §12: the authority raised to ~92 GPU-h (authority-amendment-2), about 27.8 GPU-h left, TTF-C's cap per d11.
  - v3 (`b868323ca39150c1…`) is kept unchanged in `results/`.
- v5 (2026-11-06, conv-ttf-results-2, critic-ttfc-final CF-1..CF-8; no number, table, headline or label of the
  primary changed; diff `results/ttf_vla_v4_to_v5.diff`):
  - New section "TTF-C: TTF on a causal decode (separate decode-variant experiment)" after "Diagnostics (D1)": the
    approved results sentence verbatim, the TTF-C headline and secondary block, the per-suite and baseline-row tables,
    the null and fusion rates, the D1-b comparison quoted from the sealed `ttfc_labels_c1.json` per the critic's
    ruling, every required disclosure, the run history (d14-d18, K-5, libero_goal_t2), cost with the holds separate,
    seal notes, exposure, what TTF-C can and cannot support, and its sources table.
  - Header: the source pointer is `results/ttf_vla.v5.md`; a v5 note; the numbers note names the TTF-C seal.
  - Pointers that TTF-C ran, with no primary number changed: §9 (cannot conclude), §11 (TTF-C cap and the v5 cost
    update: 14.30 GPU-h, holds of 2,054 s listed separately, session about 78.5 GPU-h of ~92), §12 (follow-up note and
    status), "Diagnostics (D1)" (what D1 does not show).
  - §14: critic-ttfc-final rows. §15: a pointer to the TTF-C sources. `results/v5_cf_checklist.md` maps each
    condition and required disclosure to its lines.
  - v4 (`af6b69ad5c88c207…`, the published `harnessstudio/docs/results/ttf_vla.md` before v5) is kept unchanged in
    `results/`. This file cannot contain its own hash; the v5 sha256 is in the task report.
- v6 (2026-11-06, conv-ttf-results-2, critic-ttf-docs-v5 `8fe68995…`; no number or label changed): FIX-1 rewords the
  TTF-C shadow-null bullet in "What TTF-C can and cannot support" (160 shadow cases, init 0-3; init state 4 had no
  shadow arm); FIX-2 corrects the TTF-C launch time 07:39:47Z to 07:39:46Z in the run history and in "Re-running
  TTF-C". The header's source pointer is `results/ttf_vla.v6.md`. v5 (`a7121d91cb260e10…`) is kept unchanged in
  `results/`.
