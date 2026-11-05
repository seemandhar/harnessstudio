# STAR (arXiv:2610.41009v1) on OpenVLA-7B / SimplerEnv Google Robot: NOT REPRODUCED (VM primary)

> **Status: ACCEPTED WITH CONDITIONS by critic-final-1** (`reviews/critic-final-1.md`, sha256 `2d552fac…`, agent
> a9079106f71bac202). The conditions are the 17 disclosures of its §9, all carried here; §15 maps each condition to
> its section.
>
> **Merge:** the M2 merge-diff review `reviews/critic-m2-1.md` (sha256 `e4c14ba7…`) returned GO WITH CONDITIONS; D1-D3
> are applied in this version.
>
> This is `harnessstudio/docs/results/star.md`. Its source is `results/star.v6.md` of run star-port-20261104, written
> by rmh_converter (task conv-commitc-14, 2026-11-05 ~00:10 UTC; earlier versions v4 and v5).
> - Relative paths without a `harnessstudio/` prefix are relative to the run folder
>   `/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/star-port-20261104/`.
> - Every number cites a file. Numbers I recomputed are marked "recomputed" (`results/recompute_conv-docs-11.json`).

## 1. Outcome first

STAR was **not reproduced**. On the predeclared primary block (SimplerEnv visual matching, the paper's 4 VM tasks,
756 paired cases):
- final-step success was 189/756 (25.0%) for the baseline and 190/756 (25.1%) for STAR, a +0.13 pp change;
- W 70 / L 69, exact McNemar p = 1.0, Newcombe 95% CI [−2.93, +3.19] pp;
- the paper-implied pooled gain of +5.57 pp lies outside that interval.

On the VA base secondary block (5 tasks, 216 pairs, descriptive only):
- STAR scored 71/216 against the baseline's 75/216 (−1.9 pp), W 23 / L 27, p 0.672;
- the paper's 5-task average is 30.32 → 35.96 %, ours 25.12 → 23.79 %.

**This was not an unfired port.** The mechanism was active: actions diverge in 740/756 VM and 216/216 VA pairs. The
overhead control `star-shadow` is action-identical to the baseline: M 756/756 and 216/216, X 0.

| block | label (frozen stats v2) | valid / expected pairs | final-step success, baseline → STAR | W / L | exact McNemar p | Newcombe 95% (STAR − baseline) | star-shadow vs baseline | paper |
|---|---|---|---|---|---|---|---|---|
| **VM primary** (T1-T4) | **not reproduced** | 756 / 756 (0 excluded; C4 cap 37) | 189 (25.0%) → 190 (25.1%) | 70 / 69 | 1.0 | [−2.93, +3.19] pp | M 756 / R 0 / X 0 | implied pooled +5.57 pp (outside the CI); 4-task avg 32.35 → 38.48 |
| VA base secondary (V1-V5) | descriptive (no reproduction label) | 216 / 216 (0 excluded) | 75 (34.7%) → 71 (32.9%) | 23 / 27 | 0.672 | [−8.22, +4.54] pp | M 216 / R 0 / X 0 | implied pooled +5.33 pp; 5-task avg 30.32 → 35.96 |
| T5 PCD seeds (optional, descriptive) | not run | — / 64 | — | — | — | — | — | — |

Notes on the table:
- **Scope string** (every claim): "SimplerEnv VM, 4 of the paper's 5 OpenVLA tasks, our OpenVLA-7B wrapper; OpenVLA
  STAR code unreleased (re-derivation)".
- **T5 status:** no T5 episode is recorded in PROGRESS.md, and the supervisor declared all GPU work finished at
  23:36Z.

Sources:
- **Final labels:** `eval/null/null_pass_assessment.json` (sha256 `6e91bc6f…`) and `reports/eval-null-8.report.json`
  (`0522678b…`).
- **Label manifests:** `stats/manifests/manifest_vm.v7.json` (`97234e18…`) and `manifest_va.v3.json` (`16301527…`).
  Both have baseline + STAR + star-shadow arms. The derived null is in separate descriptive variants (`a4dbfd7a…`,
  `166484ff…`).
- **Tool:** the frozen `stats/star_stats.v2.py` (`51570371…`), run activity-only first, then full (activity
  reproduced). Outputs: `sealed/final/vm.v7_activity.json` `13ef7299…`, `vm.v7_full.json` `112f0742…`,
  `va.v3_activity.json` `1a863c74…`, `va.v3_full.json` `6db3330d…`; cited by hash only.
- **Per-task counts, intervals and any-step:** PROGRESS.md :94 (VM) and :105 (VA); `reviews/critic-res-1.md` §2-§5
  (VM, CONFIRMED); `reports/eval-va-7.report.json` and `eval/va/va_block_assessment.json` (`db7a0d43…`).
- **Label change since the previous draft:** none. The VM label was "not reproduced (pending null-pass attribution)"
  under critic-res-1's condition C1. It is final now: the label manifest v7 adds the shadow arm, 756/756 pairs stay
  valid, and the label is unchanged.

## 2. What was ported

- **Paper and code:**
  - STAR is a proprioception-modulated EMA over the residual updates of the last third of a VLA's decoder layers.
    It uses one forward pass and no training.
  - The paper is arXiv:2610.41009 (v1 only).
  - The code is `PKU-VaLuE-Lab/STAR@0eaa15f` (Apache-2.0). It contains only the openpi π0.5 / LIBERO-PRO path; its
    README says the OpenVLA, GR00T, SimplerEnv and real-robot code is **not released**.
- **Port** (`methods/star`, package `hs-star` 0.2.1, frozen snapshot
  `eval/preflight/method-snapshot-6d618f8` manifest `47d097df…`):
  - `hs_star.star_core` holds the authors' scheduler, config and layer controller verbatim, plus seven declared
    changes [c1]-[c7]:
    - local config;
    - position selector;
    - axis-angle adapter (alternative only);
    - constant sets;
    - quaternion decompose override (geodesic d_rot);
    - shadow mode;
    - logging-only edit ratio.
  - `hs_star.server_hook.StarResidualHook` is the server-side hook. It reads only `states` and uses:
    - the Table 7 OpenVLA constants: β 0.25-0.55, w_c 1.0, w_v 1.0, w_j −0.25, w_ω −0.75, κ 4.75;
    - ρ = 2/3 (layers 22-32 of 32);
    - the 7 action-emitting positions;
    - a per-forward EMA reset;
    - β once per control step.
- **Form:**
  - It is a SCALE-style slot method (K = 1, the STAR hook as the server-side `sampler`).
  - The framework's derived null sends no sampler.
  - The overhead control `star-shadow` computes everything and writes nothing back.
  - Scored fingerprints: star `ff5c90eb…`, star-shadow `a9a82d28…`.
- **Extension:** EXT openvla-residual-hook, branch `ext2/openvla-residual-hook` @ 6d618f8 (0526759, 01e4a8e, 6d618f8
  on main f7dcd0f; tree `52e2e3c4…`). It adds:
  - the capability `residual_hooks`;
  - the server `policies/openvla_simpler_star_hs.py`;
  - `policy_ext/residual_hooks.py`;
  - the benchmark `benchmarks/simpler_vm.py`: official SimplerEnv@eddb569 VM and VA-base grids, `states` computed
    from the robot, the OpenVLA post-processing, two-stage Place in Drawer, `vm_close_drawer` refused;
  - the contract `simpler_vm` and the profile `openvla-simpler-star`.

  Reviews: critic-ext-1/2/3.

  **Merged** (`merge/M1_results.json` sha256 `fa10e663…`; M2 review `reviews/critic-m2-1.md` `e4c14ba7…`):
  - **A `d9e2e8b`:** no-ff merge of 6d618f8 into main 29e705c. A fast-forward was impossible because main had gained
    other runs' docs commits; a rebase would have dropped the scored commit. critic-m2-1 accepted this in place of
    critic-ext-2's "fast-forward merge" wording.
  - **B `34c17c8`:** `methods/star`, byte-identical to the frozen snapshot manifest `47d097df…`, with no
    `__pycache__`.
  - **Identity:** A 8/8 and B 11/11 PASS. `HEAD:harnessstudio` and `HEAD:tests` equal 6d618f8's subtrees.
  - **Regress:** ok before (84 methods at 29e705c) and after (86 methods at 34c17c8, incl. tube-steer-vfb(-ortho),
    star and star-shadow; 0 status changes).
  - **Tests:** SimplerEnv server tests 61 OK + 1 expected skip; method tests 11/11.
  - **Fingerprints:** star `ff5c90eb…` and star-shadow `a9a82d28…`, equal to the scored runs.
  - **Docs commit C:** README row, EXT status and this document, checked by `merge/stageC_results.json`.

## 3. Released-artifact findings (`audit/code_inventory.md`, `audit/paper_audit.md`, `audit/code_vs_paper.md`)

- STAR acts only on the π0.5 action-expert stream: the last 6 of 18 layers, a per-forward reset, and β per control
  step from a server-side, past-only scheduler.
- **Code-only details:** the warm start at the first mixed layer (m = Δh), and the constants z clip ±2.5, sample std,
  min-count gating and sigmoid clip ±60.
- **README vs paper:** the README's π0.5 command constants (κ 4.25, w_c 1, w_j −0.375, w_ω −2.0) differ from the
  paper's Table 7 π0.5 row (κ 2.30, w_c 0.25, w_j −0.125, w_ω −0.25).
- **Unspecified in the paper:** VM or VA, per-task N, OpenVLA layers and positions, T_w and ε. Our choices U1-U18
  are in `audit/paper_audit.md` §6.
- **Denominators:** inferred from the rounding lattice as 27/27/300/60/27 (OD/PID/Coke/MN/CD, 441 pairs).
- **Protocol hint:** the paper's Native OpenVLA row looks VA-like (CogACT v1 Table 1: OpenVLA coke VM 18.0 vs VA
  60.8). Section 7 bears this out.

## 4. Validation (all before any scored episode)

| check | result | evidence |
|---|---|---|
| Parity vs the authors' modules (CPU) | scheduler 27,322 steps x 5 constant sets: 0 diffs; quaternion scheduler 7,692: 0; torch layer step 10,660: 0; wrapper 48 forwards: 0. Bitwise equal to the authors' torch controller; equal to their JAX path up to fp32 rounding (425/1170 non-bitwise, max 9.5e-7; float64 max 1.8e-15) | `parity/parity_result.v3.json` |
| OpenVLA re-derivation, tiny 32-layer Llama, real `generate` + KV cache | hooks == the authors' wrapper transplanted (tokens and logits bitwise); STAR-off and star-shadow native; Eq. 3-5 recomputed (fp32 rel 6e-8); edit ratio vs float64 (fp32 1.7e-7, bf16 1.4e-3); logging on/off bitwise | `parity/openvla_hooks_result.v3.json` |
| Extension tests (SimplerEnv server env) | 61/61; method tests 11/11 | `ext/logs/fix4_ext_tests_authors_numerics.log` |
| regress | pytest 103 passed; every method's status = the f7dcd0f baseline (tube-steer-vfb npz untracked on main, so absent in the worktree) | `ext/regress_compare.fix4.json` |
| Merge (executed after all GPU work) | rehearsal in a scratch clone first (`merge/rehearsal*`). Then on main: A d9e2e8b (no-ff merge of 6d618f8 into 29e705c) and B 34c17c8 (snapshot byte-identical); identity A 8/8, B 11/11 PASS; regress 84 → 86 methods ok (tube-steer-vfb(-ortho), star, star-shadow); server tests 61 OK + 1 expected skip; method tests 11/11; fingerprints = scored runs; M2 GO WITH CONDITIONS | `merge/M1_results.json` (`fa10e663…`), `reviews/critic-m2-1.md` (`e4c14ba7…`) |
| tier 0 / tier 1 (`star`, `star-shadow`) | PASS (fake-t3) | `ext/verify_tier01.fix4.json` |
| B1 `states` in the pinned image | 9/9 env families; fork formula ≤ 1.5e-8; vla-eval fallback ≤ 1.2e-7; repeats across processes | PROGRESS 07:13 UTC |
| N1 server equivalence | STAR-off = parent 9/9; startup fingerprint 8cd3b024770ffa2e | eval-pre-3 |
| N2 arm equivalence | T1-T4 and V1-V5: baseline = derived null = star-shadow bitwise | eval-pre-3, `eval/preflight/preflight_report.v2.json` |
| B4 logging on/off (GPU) | STAR and shadow bitwise | `preflight_report.v2.json` |
| tier 2 (`hs verify star --tier 2`, T1) | PASS; null vs baseline M1/R0/X0 | job2; critic-pre-2 |
| Reference episodes | 16 complete valid job1 episodes on T1-T4 (eval-ref-5c) | PROGRESS 07:21 UTC |

## 5. Predeclared protocol (SETTING.md rev 3, sha256 `c98f718d…`, frozen 2026-11-04T04:29Z before any GPU run)

- **Policy:** `openvla/openvla-7b@31f090d`, bf16, eager attention, greedy 7-token `generate`, run through the PCD-port
  SimplerEnv wrapper. The same server process serves every arm of a case.
- **VM primary (756 pairs):** the official visual-matching enumeration of SimplerEnv@eddb569 at the official
  horizons: T1 Pick Coke Can 300 (80 steps), T2 Move Near v0 240 (80), T3 Open Drawer 108 (113), T4 Place in
  Drawer 108 (200, two-stage).
  - VM Close Drawer is refused: SETTING §3.4 isolates it from the PCD discovery run's held-out drawer scenes.
- **VA base secondary (216 pairs, descriptive):** V1 Open Drawer 27, V2 Close Drawer 27, V3 Place in Drawer 27,
  V4 Move Near 60, V5 Coke 75. Four of the five N equal the paper's inferred N; Coke is 75 vs the paper's 300.
- **Success:** final-step success is primary; any-step is secondary. A short decode counts as a completed policy
  failure in every arm.
- **Statistics:** pooled exact two-sided McNemar, α 0.05, Newcombe CI.
- **Labels:** "gain reproduced in direction" (+ "magnitude compatible"), "not reproduced", "reversed"; "mechanism
  inactive" overrides them.
  - C4: at most 37 excluded pairs, with a robust worst case.
  - SETTING's null is star-shadow. critic-pre-4 ruled that the framework's derived null is an extra determinism
    control, reported separately, that never changes a pair or a label.
- **Execution:** the frozen tool runs activity-first. Each block's manifest is written before its episodes. Gate v2
  separates infrastructure (one bounded exact-fixture retry) from attribution (R/X are never retried).

## 6. VM primary results (final)

| task | N | baseline k (%) [CP 95%] | STAR k (%) [CP 95%] | Δ pp [Newcombe 95%] | W / L | exact McNemar p | paper Native → STAR (Δ) |
|---|---|---|---|---|---|---|---|
| T1 Pick Coke Can | 300 | 48 (16.0) [12.0, 20.6] | 43 (14.3) [10.6, 18.8] | −1.7 [−6.12, +2.76] | 20 / 25 | 0.551 | 65.3 → 70.0 (+4.7) |
| T2 Move Near v0 | 240 | 110 (45.8) [39.4, 52.4] | 120 (50.0) [43.5, 56.5] | +4.2 [−2.33, +10.59] | 37 / 27 | 0.260 | 56.7 → 61.7 (+5.0) |
| T3 Open Drawer | 108 | 31 (28.7) [20.4, 38.2] | 27 (25.0) [17.2, 34.3] | −3.7 [−13.54, +6.21] | 13 / 17 | 0.585 | 7.4 → 18.5 (+11.1) |
| T4 Place in Drawer | 108 | 0 (0.0) [0, 3.4] | 0 (0.0) [0, 3.4] | 0 [−3.43, +3.43] | 0 / 0 | 1.0 | 0.0 → 3.7 (+3.7) |
| **pooled** | 756 | 189 (25.0) | 190 (25.1) | +0.13 [−2.93, +3.19] | 70 / 69 | 1.0 | implied +5.57 |
| **4-task average (RC4)** | | 22.63 | 22.33 | −0.30 | | | 32.35 → 38.48 (+6.13) |

The per-task rows are descriptive; there are no per-task claims.

Sources: CP intervals recomputed; per-task Newcombe and p from critic-res-1 §2.

Notes:
- **Any-step (secondary):** baseline 218, STAR 217, W 68 / L 69, p 1.0, Newcombe [−3.17, +2.90] pp. The one unknown
  STAR any-step value (vm211 short decode) gives W/L 68/69 or 69/69; p = 1.0 either way.
- **Short decodes:** 1 in STAR (S32 Move Near vm211, decision 52: 6 tokens for 7 dims; CRN seed matches). It is a
  completed STAR failure; that case's baseline also failed. The baseline had 0.
- **Mechanism activity** (computed before any success field):
  - actions diverge in 740/756 pairs (97.9%); the step-0 action changes in 533/756 (70.5%);
  - β is non-constant in 755/755 joined STAR episodes, within [0.25, 0.55];
  - the edit ratio at layer 22 is O(eps) (median 5.5e-4, max 3.2e-3); medians at layers 23 / 27 / 32 are 0.516 /
    0.459 / 0.423;
  - `r_t_tilde` is heavy-tailed, because the W = 6 IQR is small (U8).
- **C4:** 0 pairs excluded (cap 37). With 28 or more exclusions counted as STAR losses, the worst case would be
  "reversed", so the label would have become "incomplete" under C4 (critic-res-1 condition C1). 0 pairs were
  excluded, so "not reproduced" is final. In the other direction, 26 extra STAR wins would have been needed for
  p < 0.05.
- **Power:** the planned power for the paper's effect at 756 pairs was about 0.89 under PCD-like discordance and
  about 0.70 under independence (`audit/power.md` :113-114, :501).

**Baseline vs the paper's Native row** (two-sided Boschloo; recomputed with the frozen `boschloo_p`, as in
critic-res-1 §5):

| task | ours | paper (inferred) | Boschloo p | matched? |
|---|---|---|---|---|
| T1 Coke | 48/300 (16.0%) | 196/300 (65.3%) | 1.7e-36 | **no** (far lower) |
| T2 Move Near | 110/240 (45.8%) | 34/60 (56.7%) | 0.147 | yes |
| T3 Open Drawer | 31/108 (28.7%) | 2/27 (7.4%) | 0.018 | **no** (higher) |
| T4 Place in Drawer | 0/108 | 0/27 | 1.0 (D12) | yes |

## 7. VA base secondary results (final; descriptive, never a label)

| task | N | baseline k (%) [CP 95%] | STAR k (%) [CP 95%] | Δ pp | W / L | exact McNemar p | paper Native → STAR | baseline vs paper Native, Boschloo p |
|---|---|---|---|---|---|---|---|---|
| V1 Open Drawer | 27 | 2 (7.4) [0.9, 24.3] | 3 (11.1) [2.4, 29.2] | +3.7 | 2 / 1 | 1.0 | 7.4 → 18.5 | 1.0 (2/27 vs 2/27) |
| V2 Close Drawer | 27 | 5 (18.5) [6.3, 38.1] | 4 (14.8) [4.2, 33.7] | −3.7 | 1 / 2 | 1.0 | 22.2 → 25.9 | 0.772 (vs 6/27) |
| V3 Place in Drawer | 27 | 0 (0.0) [0, 12.8] | 0 (0.0) [0, 12.8] | 0 | 0 / 0 | 1.0 | 0.0 → 3.7 | 1.0 (D12) |
| V4 Move Near (VA) | 60 | 27 (45.0) [32.1, 58.4] | 23 (38.3) [26.1, 51.8] | −6.7 | 7 / 11 | 0.481 | 56.7 → 61.7 | 0.219 (vs 34/60) |
| V5 Pick Coke Can | 75 | 41 (54.7) [42.7, 66.2] | 41 (54.7) [42.7, 66.2] | 0 | 13 / 13 | 1.0 | 65.3 → 70.0 | 0.094 (vs 196/300) |
| **pooled** | 216 | 75 (34.7) | 71 (32.9) | −1.85 [Newcombe −8.22, +4.54] | 23 / 27 | 0.672 | implied +5.33 | — |
| **5-task average** | | 25.12 | 23.79 | −1.33 | | | 30.32 → 35.96 (+5.64) | |

- **Sources:**
  - per-task counts, W/L for V4 and V5, pooled numbers and averages: PROGRESS.md :105 (eval-va-7, frozen tool on
    `manifest_va.v1`);
  - final pooled label numbers: `eval/null/null_pass_assessment.json` on `manifest_va.v3`;
  - CP intervals, the per-task McNemar values for V4/V5 and every Boschloo value: recomputed.
  - V1/V2 W/L (2/1, 1/2): the source is the frozen tool's full VA output, `sealed/va/analysis/va_full.json`
    (sha256 `d9166e5b40c53c10949149f6a5076723f13e65fcd37bf5556c4f288c3125f39e`, cited by hash only), relayed by the
    evaluator.
    - `reports/eval-va-7.report.json` does not carry per-task W/L.
    - The values are consistent with the per-task counts (2 − 1 + 2 = 3; 5 − 2 + 1 = 4) and sum with V4/V5 to the
      pooled W 23 / L 27; critic-m2-1 §3 checked this too.
    - Their exact McNemar p = 1.0 is recomputed.
  - Newcombe [−8.22, +4.54] pp from `reviews/critic-final-1.md` §2 (PROGRESS :105 rounds it to [−8.2, +4.5]).
- **Baseline consistency:** the VA baseline matches the paper's Native row on all 5 tasks (minimum Boschloo
  p = 0.094, Coke). The VM baseline matched on only 2 of 4 tasks.
- **Any-step:** 88 vs 88 (W 21 / L 21, p 1.0).
- **Short decodes:** 1, in the baseline (A04 va018, decision 75). It is a completed baseline failure. It repeats
  bitwise in the derived null and star-shadow (`null_pass_assessment.json`, gate_v2 line; critic-final-1 §2). STAR
  had 0. W/L is unchanged.
- **Mechanism activity:** actions diverge in 216/216 pairs and the step-0 action changes in 204/216. β is
  non-constant in 216/216, range 0.25-0.55. The whole-request edit-ratio median is 0.460 (critic-final-1 §4).
- **Mixed line** (the only one allowed, labelled "not comparable to Table 1"): VM T1-T4 + VA V2 Close Drawer, ours
  21.81 % → 20.83 % (critic-final-1 §1, the frozen tool's `mixed` command). There is no 5-task VM average, because
  VM Close Drawer was refused.
- **Power:** the predeclared power for the paper's effect was about 0.34 (PROGRESS.md :105). That is why VA carries
  no label.

## 8. Null / star-shadow pass (final)

- **Volume:** VM 1,512 + VA 432 episodes (derived null + star-shadow per case), plus 11 bounded retries.
- **star-shadow** (SETTING's null, the overhead control): VM **M 756 / R 0 / X 0**, VA **M 216 / R 0 / X 0**. The
  overhead control is action-identical to the baseline, and there is no nondeterminism evidence (R = 0).
- **Derived null** (extra determinism control): VM M 756/756, VA M 216/216, R 0, X 0.
- **Gates and exclusions:** gate v2 passed every shard. The final per-arm record is a2 for the 11 INC-1 arms. 0 infra
  exclusions in either block.
- **Orphans:** the VM label residual is 4, exactly the 4 declared INC-1 partial shadow groups (S14 vm097/vm098,
  S15 vm062/vm063; `eval/null/expected_orphans.v2.json` `7bd66dbf…`). The VA label residual is 0.
- **Latency (descriptive, from records; critic-final-1 §6):** median policy seconds per step were baseline
  0.3445, STAR 0.3531 (+2.5%), star-shadow 0.3502 (+1.7%), derived null 0.3421. These ratios are **noisy** on a shared
  host (critic-pre-1b G7) and are not a controlled overhead measurement. The paper reports +3.3% for π0.5 only; we
  do not compare with it.
- **Retry records:** the a1 records of the 11 INC-1 arms are kept, and each retried arm has exactly one a2 record
  (no a3). The a2 gate v2 files list the *non-retried* arm of each retried case as "infra/retry" only because that
  arm is absent from the a2 directory by design. That is an artifact, not a second retry (critic-final-1 §5).
- **Identity:** 57 null-pass "P6 after EQUAL"; startup probe `8cd3b024770ffa2e` on all 107 logged launches; the
  highest launch-time load1 over 111 launches was 59.2, under the gate of 64; 2 workers per GPU.

## 9. Incidents and retries (`eval/null/incidents.v4.jsonl` `d3185ac3…`)

**INC-1: host/driver transient, 19:08:29-19:08:42Z, GPUs 5-7, null pass only.**
- **Errors:** Vulkan `vk::Device::createFenceUnique: ErrorOutOfHostMemory` at simulator env step (record invalid,
  `empty_result`), and NVML `Failed to initialize NVML: Unknown Error` at the benchmark reset's GPU-isolation check
  (no record).
- **Affected arms (11):**
  - S14: vm097, vm098 shadow; vm099 null.
  - S15: vm062, vm063 shadow; vm064, vm065 null.
  - S16: vm034, vm035 shadow; vm036, vm037 null.
- **Host state:** 257 GiB free; no OOM/oomd entry; GPU 5 at 15-16 of 48 GiB. The scored 2-arm records were
  unaffected.
- **Partial records:** for every partial a1 record, the action-hash prefix equals the baseline's and obs0 is
  equal. No completed or divergent episode was retried (critic-final-1 §5).
- **Classification and retry:** gate v2 classified all 11 as infrastructure, with no R/X. S14 (obs0 equal) and
  S15 were `empty_result` (Vulkan). S16's two shadow arms had record count 0 (NVML); its two null arms were
  `empty_result`. Decision d11 allowed one bounded exact-fixture retry per arm in a2 directories, under manifests
  hashed first.
- **Result:** all 11 retries ran 23:24-23:35Z on GPU 4 (runner v3, driver v5); **all 11 are M**. A recurrence
  watcher (vulkan_watch v2) saw no new burst.

**INC-2: failed first retry launches and lease sharing, 21:50-21:53Z.**
- **Failed launches:** the d11 retry launches S14 a2 and S15 a2 died at policy-server start with FileExistsError,
  because the server needs a fresh action-log path per launch. No episode ran; S16 a2 never started. Supervisor
  ruling d12: a launch that dies before any episode does not consume the arm's retry.
- **Lease sharing:** the retry runner and scheduler v4 used the same lease owner on GPU 4.
  `lease.py` treats a same-owner reservation as idempotent, so VA null shard A03 started on GPU 4 under the retry's
  lease. GPU 4 was unleased for about 5 s while A03 ran. No two servers or episodes ran concurrently.
- **Fix:** driver v5 (a fresh sub-root per attempt), runner v3 (refuses while the scheduler is alive or any lease is
  held), and critic-pre-5's conditions R1-R6. The residue of the failed launches is kept and listed in INC-2.

## 10. Comparison with the paper and candidate explanations (HYPOTHESES, untested)

**What the paper reports versus what we measured:**
- STAR's paired effect on our stack is +0.13 pp pooled on VM (CI [−2.93, +3.19]) and −1.85 pp on VA (descriptive).
- Per task, the paper reports gains on all 5 tasks.
- On VM we see −1.7 / +4.2 / −3.7 / 0 pp; on VA +3.7 / −3.7 / 0 / −6.7 / 0 pp. None is significant.
- The mechanism is demonstrably active, and the overhead control is action-identical.

Candidate reasons. None was tested here.
- **H1 Protocol or wrapper.** On VM our baseline is far from the paper's Native row on Coke and Open Drawer.
  - On VA it matches all 5 tasks, so the paper's Native row is consistent with VA.
  - Even on VA, where the baseline matches, STAR shows no gain: 75 → 71/216, paper-implied +5.33 pp; VA is
    underpowered (~0.34).
  - So a protocol mismatch alone does not explain the missing VM gain. A wrapper or checkpoint difference remains
    possible.
- **H2 Re-derivation choices.** The authors' OpenVLA code is unreleased. U1-U18 are our choices:
  - layer band 22-32;
  - the 7 action positions only;
  - W = 6;
  - the float32 geodesic floor;
  - bf16 EMA;
  - a warm start at layer 22.

  A different choice could be what the paper ran.
- **H3 Selection and noise in the paper.**
  - The Table 7 constants and the reported rows came from the same evaluation episodes.
  - Ablation variants span 28.9-36.0%.
  - At the paper's inferred N (441), its own design has about 0.53 power (`audit/power.md` :49, :120).
  - It may report 8-run means (U16).
- **H4 Inferred denominators.** A wrong inferred N changes the implied +5.57 / +5.33 pp, not our measured effects.

## 11. Cost (`eval/cost/gpu_total.v1.json` `8579c09f…`, evaluator a3d62adf75e63d0fe, from the lease ledger)

| task | GPU-s |
|---|---|
| eval-ready-1 (reset probe) | 1,072.5 |
| eval-pre-2 (calibration) | 2,606.4 |
| eval-pre-3 (preflight B1/N1/N2) | 2,085.5 |
| eval-vm-6 (VM shards 52,523.7 + job2 1,301.0) | 53,824.7 |
| eval-va-7 (VA shards) | 17,893.6 |
| eval-null-8 a1 (null pass) | 69,804.5 |
| eval-null-8 a2 (INC-2 failed launches 17.7 + retries 658.2) | 675.9 |
| **total** | **147,963.1 GPU-s = 41.1 GPU-h** (GPUs 4-7; 119 intervals; 0 overlaps) |

- **Source:** compute is cited only from `gpu_total.v1.json` (critic-final-1 §7, condition 15). That file counts the
  INC-2 shared GPU-4 window once and attributes it to A03; it supersedes the null-pass assessment's provisional
  figure.
- **Paid models:** 0 USD. STAR uses no runtime LLM (0 calls).
- **Not measured:** outer Claude agent usage.

## 12. Fidelity label (SETTING rev 3 §1, verbatim; method card `substitutions[0]`)

adapted (re-derivation: OpenVLA/SimplerEnv STAR code unreleased; STAR choices U1-U18 of audit/paper_audit.md §6:
Llama-2 decoder layers 22-32 with the released EMA warm start, whole-layer residual update, the 7 action-token
positions only, per-forward EMA reset with step-level beta, one action per inference (no state-history ingestion),
agent.eef_pos proprio, released float32 geodesic quaternion d_rot (resolution floor ~1e-3 rad; float64 |drot|
logged), Tw 2.0 s -> W 6, released epsilon/clip/warm-up constants, bf16 EMA, torch implementation (the released
pi0.5 reference path is JAX; parity measured), Table 7 OpenVLA constants, no scheduler reset at the Place-in-Drawer
subtask switch, one deterministic run per case (the paper may report 8-run means); protocol: SimplerEnv visual
matching assumed (paper silent; its numbers look VA-like), Close Drawer excluded from the VM block (no VM 5-task
average), Move Near v0 (official VM script; prepackaged task is v1), final-step success, short decodes scored as
policy failures; policy openvla-7b@31f090d with the PCD-port SimplerEnv wrapper (paper names no checkpoint);
inherited PCD-port substitutions: float32 action wire [B7 rule violated], HarnessStudio transport, env identity
[vla-eval SIMPLER image vs SimplerEnv@eddb569])

- **Proprio:** "agent.eef_pos proprio" is implemented as the same quantity, computed by the benchmark from the robot
  (`agent.robot.pose` + link `link_gripper_tcp`, the fork's eef_pos formula). The pinned image has no
  `agent.eef_pos` (P8; critic-ext-2 ruled it permitted public proprioception).
- **VA:** VA runs the VA base scenes with Move Near `MoveNearGoogleInScene-v0` (SETTING :35).

## 13. Disclosures (`control/report_notes/REPORT_ITEMS.md` `db242771…`, all 21 items, plus run records)

1. **J1 reset probe.** It ran against the prototype `simpler_ext` at its earlier hash ff0b9b22, now superseded and
   frozen at 1d77eca0 (`ext/PROTOTYPE_FROZEN_OUT.md`). The scored runs used the branch `benchmarks/simpler_vm.py`
   @ 6d618f8.
2. **Stale SETTING rev 3 citations.**
   - :97 (record fields) maps to the branch `simpler_vm.py` docstring :24-25, `get_step_result` :297-300 and
     `get_metric_keys` :302-303.
   - :165 maps to `REFUSED` :85-86 and `enumerate_grid` :113-114.
3. **SETTING header.** Rev 3 line 6, "Status: DRAFT", is stale text, superseded by the FROZEN header
   (`control/setting/FROZEN.json`).
4. **Amended commit.** 10c6f7c failed one regress test (transforms3d absent in the shared venv). It is preserved in
   `ext/amended_10c6f7c/`. The first-run regress files were overwritten (`reports/conv-fix-4.report.json`).
5. **Parity JAX rule.** It was changed twice after the first run. r1 and r2 are reported as failed; r3 is fp32
   rounding (1.8e-15 in float64).
6. **Layer-22 edit ratio.** It is O(eps), not exactly 0 (bf16 up to 3.1e-3; VM max 3.2e-3).
7. **Host overload.** eval-pre-2 calibration (1 server x 4 workers on three GPUs) contributed to the host overload at
   ~05:50Z. The CPU rule (authority-amendment-2) applied afterwards.
8. **Smoke outcome line.** During calibration the evaluator saw one native baseline outcome line for the first Coke
   smoke case. Smoke STAR outcomes were not scored.
9. **Knowledge-card registry.** `star_residual_ema` was refreshed (sha 7b831c76). Registry re-registration is still
   pending an authorized applier.
10. **Prior exposure before the scored VM block.**
    - The 9 smoke cases ran in preflight, unscored, and were re-run in the scored blocks.
    - critic-pre-1b read the pooled counts of the 4 VM smoke pairs; it did not use them.
    - critic-res-1 read outcomes only after the VM block was complete.
11. **Preflight tool edited in place.** `preflight_check.py` was edited after job1 recorded hash 1ff5d082. The final
    0bcfa616 reproduced job1's N1/N2 and failed 5 critic sabotages (G8).
12. **Profile override.** The bound STAR policy JSON names main's simpler-logits profile. The scored driver used an
    explicit `--profile openvla-simpler-star` override (G9).
13. **Latency.** Preflight latency ratios are noise. The scored-record medians (+2.5% STAR, +1.7% shadow; §8) are
    reported as noisy descriptive numbers, not a controlled overhead measurement. The paper's +3.3% (π0.5) is never
    claimed (G7).
14. **Re-registration timing.** Knowledge-card re-registration (critic-setting-2 §5) was deferred until the
    discovery run's RUN CLOSED, so that the shared retrieval index would not change while discovery could still
    ideate. The card carries only audit facts.
15. **CHANGELOG v32.** The porting skill's append-only CHANGELOG v32 entry, written by the discovery run, carries an
    aggregate of discovery validation outcomes. STAR agents and the supervisor did not read or copy it.
16. **Discovery PROGRESS tail (supervisor).** At ~11:2xZ the supervisor read discovery's PROGRESS.md tail, which
    included a validation aggregate. No STAR decision depends on it.
17. **critic-pre-2 visibility.**
    - It saw one line of discovery's validation summary (not used).
    - Its stats dry run on S01+S02 read only completeness, infra and orphan counts.
18. **Ledger frozen at 15:41:44Z.** `reports/eval-vm-6.ledger-report.json` bound the append-only lease log, so the
    advisory ledger now refuses every command. Bookkeeping continued with task files and
    `control/research/bindings.jsonl` (`control/research/LEDGER_FROZEN.md`).
19. **VM block disclosures (critic-res-1 §7, 13 items).** All are covered in §1, §6, §8, §11, §12 and items 10 and 20
    here. C1 is resolved by §8.
20. **Null-pass infra incident INC-1** and its bounded retries (d11): §9.
21. **Lease incident and failed first retry launches (INC-2):** §9. Lesson candidate: lease.py's same-owner
    idempotence lets two launchers of one run share a GPU.

Also recorded during the run:
- **P6 main-commit provenance.**
  - harnessstudio main moved f7dcd0f → fc8a3af (11:18:42Z) and later to 29e705c (11:29:26Z); both are PCD discovery
    docs-only commits.
  - Decision d08 introduced per-shard before==after P6. The S04 P6 halt was investigated and recorded valid in
    `eval/vm/p6_investigations.jsonl`.
  - The package tree was unchanged throughout (d3bc20ba; critic-res-1 §6).
- **Manifest versions.** VM v1 → v7 and VA v1 → v3 (d08, d09, N1, d11, d12).
  - v1 → v3: the arms, logs and cases are unchanged (critic-res-1 §7 item 10).
  - v3 → v5 (N1): adds the shadow arm (and the derived-null variants).
  - v5 → v6 and v6 → v7: change only the a2 directories, logs and provenance. No changed line names a baseline or
    STAR arm (`eval/null/retry/manifest_v5_to_v6.diff`, `manifest_v6_to_v7.diff`; checked by grep in
    conv-docs-11).
- **Short decodes.** One STAR short decode (VM vm211) and one baseline short decode (VA va018), both scored as
  completed failures and never rerolled.
- **`regress.py --inflight`** does not see run-local drivers (found in conv-merge-plan-10). The merge plan adds a
  process check.

## 14. Reproduce

1. Install the merged code: harnessstudio main contains EXT openvla-residual-hook (merge A `d9e2e8b` of 6d618f8)
   and `methods/star` (B `34c17c8`). Run `uv pip install --python .venv/bin/python -e methods/star`, then
   `hs verify star` / `star-shadow` (tier 0/1).
2. Run the scored drivers: `eval/vm/run_vm_shard.v2.sh`, the VA shard driver, and the null-pass drivers
   `eval/null/run_null_shard.v4.sh` / `.v5.sh`. Their pins are in `eval/null/pins.v4.json` and `pins.v5.json`
   (image `ghcr.io/allenai/vla-evaluation-harness/simpler@sha256:f7744967…`, startup probe 8cd3b024770ffa2e).
3. Run the analysis on `stats/manifests/manifest_vm.v7.json` / `manifest_va.v3.json`, in two steps:
   - `stats/star_stats.v2.py analyze --manifest M --activity-only --out A.json`;
   - `... --activity-file A.json --out FULL.json`.

Porting-skill version: this run's retrospective part 1 committed v30 and its corrections v31. The skill is at v34
now (`skill_evolve.py status`); v32-v34 came from other runs, and I did not read them.

## 15. critic-final-1 §9 conditions: where each is disclosed

| # | condition | where |
|---|---|---|
| 1 | Labels and scope: VM "not reproduced" + scope string + full fidelity label; VA "descriptive"; T5 not run | §1, §12 |
| 2 | VM 189 → 190/756, W70/L69, p 1.0, [−2.93, +3.19] pp; +5.57 pp outside the CI; planned power 0.89 / 0.70 | §1, §6 |
| 3 | VA 75 → 71/216, W23/L27, p 0.672, [−8.22, +4.54] pp, descriptive only | §1, §7 |
| 4 | Per-task rows with p for both blocks; no per-task claims | §6, §7 |
| 5 | Any-step (VM 218 → 217; VA 88 → 88) and the vm211 bounds | §6, §7 |
| 6 | RC4 like-for-like averages per block; only the labelled mixed line 21.81 → 20.83 ("not comparable to Table 1"); no 5-task VM average | §6, §7 |
| 7 | Baseline consistency: VM not matched on Coke (lower) and Open Drawer (higher); VA matched 5/5. The paper's Native row looks VA-like, so the VM absolute comparison is not like for like | §6, §7, §10 |
| 8 | Mechanism active: 97.9% / 100% divergence, step-0 change (533/756, 204/216), β range and non-constancy, edit-ratio medians, heavy-tailed `r_t_tilde` under W = 6; A8 did not fire; "not an unfired port" | §1, §6, §7 |
| 9 | Controls: shadow M 756 / 216, R 0, X 0; derived null M 756 / 216 (not a label input); latency medians +2.5% / +1.7%, noisy | §8 |
| 10 | Short decodes: VM STAR vm211 (decision 52); VA baseline va018 (decision 75), repeated in shadow/null; W/L unchanged | §6, §7 |
| 11 | C4: 0 excluded in both blocks, cap 37, worst case unchanged | §1, §6 |
| 12 | INC-1 (11 arms, one bounded retry each, all M, a1 records kept, partial prefixes = baseline); INC-2 (failed launches, d12, GPU 4 ~5 s unleased, same-owner lease); 4 declared orphan groups; the a2 "infra/retry" gate artifact | §8, §9 |
| 13 | Infrastructure: shard gates (VM 39, VA 13, null pass 52 shards + 11 retries); startup probe constant; P6 events S01-S05 explained, null pass all EQUAL; CPU rule kept (load1 ≤ 59.2 < 64, 2 workers/GPU); manifest versions VM v1 → v7, VA v1 → v3 (d08-d12; critic-pre-1b..5) | §6, §8, §13 |
| 14 | Prior exposure: 9 smoke cases re-run in the blocks; critic-pre-1b read the pooled n = 4 smoke counts; the evaluator saw one Coke vm000 native summary line; reviewers read outcomes only after block completion | §13 items 8, 10 |
| 15 | Compute: 41.1 GPU-h from `gpu_total.v1.json` by task; 0 paid calls; the superseded assessment figure is not used | §11 |
| 16 | Tool identity: stats v2 `51570371`. Independent re-runs by critic-res-1 (VM) and critic-final-1 (VM v7, VA v3 and both derived-null variants) are numerically identical; only path fields differ | §1, this row |
| 17 | Discovery isolation: no seeded-protocol case run (T5 not run); VM Close Drawer refused; VA V2 uses `frl_apartment_stage_simple` base scenes, disjoint from discovery's held-out pool; discovery closed 11:34Z, and before that only gate counts existed for S01-S03. The STAR-vs-PCD comparison (T5) was not done | §5, this row |
