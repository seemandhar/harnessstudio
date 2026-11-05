# TTF-VLA (arXiv:2508.19257v3) on OpenVLA-7B / LIBERO: not reproduced (non-matching baseline regime; adapted)

> **Status: ACCEPTED WITH CONDITIONS by critic-ttf-final.** The review is `reviews/critic-ttf-final.md` (sha256
> `d1170e947d48afbd…`, agent a027b13c594984db6, 2026-11-05T23:37Z). Its conditions C1-C7, required disclosures and
> interpretation are binding on this document. §14 maps each condition to the section that meets it.
>
> **Where this document lives:** it is `harnessstudio/docs/results/ttf_vla.md`. Its source is `results/ttf_vla.v1.md`
> of run `repro-next-20261105`, written by rmh_converter (task conv-ttf-results-1, 2026-11-05, 23:40Z to 00:30Z UTC).
> The two files are byte-identical.
>
> **Paths:** relative paths without a `harnessstudio/` prefix are relative to the run folder
> `/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/repro-next-20261105/`.
>
> **Hashes:** sha256, truncated to 16 hex digits; the full values are in the cited files.
>
> **Numbers:** every number comes from the sealed result `sealed/` (SEAL `dda8177cb4fe05cf…`, 17 files) or the files
> cited next to it. No label in this document differs from `sealed/labels.json` (`ce0f70f8131a244b…`).

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

`harnessstudio/methods/ttf` is byte-identical to that package; see §13.

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

**Cannot be concluded:**
- TTF's effect in the paper's ~68% baseline regime, or on a standard causal decode;
- that the paper's Table 1 is wrong or irreproducible in its authors' environment;
- that Table 1 was produced with a non-causal prefill;
- which cause explains the baseline gap;
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

**The rest of the authority.**
- About 19.65 GPU-h remains.
- Pigey's share under A1 C2 is ≤ 12 GPU-h; its SETTING v3 allowance is 12.25. Pigey has used 0 GPU-h so far.
- That leaves about 7.4-7.65 GPU-h unallocated.

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

The last option does not fit the remaining authority without a user budget increase.

## 13. Reproducing the scored run (evaluator-owned; GPU)

The method package is `harnessstudio/methods/ttf`: hs-ttf 0.1.0, entry points `ttf`, `ttf-paper-baseline` and
`ttf-shadow`, byte-identical to the scored `method/` per MANIFEST `a611af04…`.
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

## 16. Changelog

- v1 (2026-11-05, conv-ttf-results-1): first results document, written after the seal and the final review. The card
  `methods/ttf/hs_ttf/paper.yaml` gets one appended changelog line, which is metadata only and leaves the fingerprints
  unchanged, following the STAR precedent.
