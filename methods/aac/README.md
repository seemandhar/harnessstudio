# hs-aac: Adaptive Action Chunking (`aac`)

Liang et al., *Adaptive Action Chunking at Inference-time for Vision-Language-Action Models*,
arXiv:2604.04161v2 (CVPR 2026). Author code: `Adaptive-Action-Chunking/robocasa@fed3e6b5`
(`robocasa/demos/action_optimization/action_entropy_v2.py`, `select_chunk_size(method="gaussian_bernoulli",
move_th=3.0)`, called with `--chunk_id_selector 0`) and the sampling server
`Adaptive-Action-Chunking/gr00t-multi-sample@11e926b0` (`n_samples=20`). Ported from HarnessStudio v1
(branch `ext/aac` @ `1422be0`, `harnessstudio/methods/aac.py`).

| | |
|---|---|
| Entry point | `aac` → `hs_aac:config` |
| Slots | `proposer` (`AACPool`, N=20) + `executor` (`AACEntropyExecutor`, `sees_proposals = True`). No selector. |
| Fidelity | `faithful`, `substitutions: []`, named differences D1–D7 under `disclosures:` in `hs_aac/paper.yaml` |
| Requires | `act, chunk, seeded_sampling, isolated_streams` |
| Categories | primary `uncertainty_compute`, secondary `execution_monitoring` |
| Null | `derived` (the null pays N=20 samples per decision and runs sample 0 for its native length) |
| Artifact | GR00T N1.5 RoboCasa365 checkpoint-120000 `experiment_cfg/metadata.json` (sha256 `8be0fc60…`): source of the min_max constants |

## What is reproduced

For every decision the policy is sampled N=20 times at the same observation (the DecisionLoop calls
`ctx.policy.sample(20)` because the proposer has `num_samples=20`; sample 0 is the native stream). The
executor sees all 20 chunks (`extras["proposals"]`, seed order) and computes:

- **Entropy elbow.** For each step t: `H_t = H_gauss(normalized position of the 20 samples at t) +
  H_gauss(normalized rotation at t) + H_bernoulli(gripper at t)`, weights 1/1/1,
  `H_gauss = 0.5 (d log 2πe + logdet(cov + 1e-6 I))` (`np.cov`, ddof=1). `chunk_mean[k] = mean(H_1..H_k)`,
  `k_entropy = max(argmax(diff(chunk_mean)) + 1, 2)`.
- **Motion floor.** On sample 0 (unnormalized): `m_k = ||Σ_{t<k} pos_t|| + ||compose(rot_1..k)|| + 0.2 · toggle_k`
  for k = 2..16; `k_motion` = first k with `m_k > 3.0`, else 16.
- **Execute** sample 0 for `k = max(k_motion, k_entropy)` steps (2..16), then re-plan.

All constants are v1's: N=20, `move_th=3.0`, `cov_eps=1e-6`, magnitude weights 1/1/0.2, `min_k=2`,
`per_step=False`, execute sample 0. The numerics are v1's numpy port, which matches the author functions
bitwise (checked again here, see *Verify*).

### Why these categories

AAC estimates the policy's uncertainty from the spread of N samples and uses it to decide how long to
commit before the policy is queried again. That is uncertainty-driven inference allocation
(`uncertainty_compute`: "Uncertainty and inference allocation"), so it is the primary category. It is
realized in the `executor` slot and sets how much of each chunk runs, which is the
`execution_monitoring` slot family ("Execution monitoring and recovery"), so that is secondary. It is
not `proposal_selection`: it never chooses among the samples (sample 0 always runs).

## Disclosed differences from the paper and author code (D1–D7, unchanged from v1)

- **D1 noise.** Per-sample Protocol-N1 seeded streams (sample 0 = the native stream) instead of one batched
  `torch.randn(N, 16, D)`; iid Gaussian either way, not bitwise.
- **D2 compute.** v1 drew K sequential full `get_action` calls (VLM backbone per sample) instead of the
  author server's shared-backbone batch. In v2 the 20 seeds go out in one `sample(20)` request and the
  policy server's `sample_chunks` decides whether to share the backbone. Cost only.
- **D3 normalization.** The author code reads the model's `normalized_action`; HarnessStudio chunks are
  unnormalized, so position and rotation are re-normalized with the checkpoint's min_max statistics (the
  exact affine inverse of GR00T's Normalizer, up to float rounding). For RoboCasa365 multitask
  checkpoint-120000 both keys have min −1 / max 1, so the map is the identity.
- **D4 action groups by key/name**, not by the author's 7-D `single_panda_gripper` column indices
  0:3 / 3:6 / 6:7. In v2 the names are vla-eval's `position`, `rotation` and `gripper` from
  `Embodiment.action_layout`. GR00T's `action.end_effector_position`, `action.end_effector_rotation` and
  `action.gripper_close` (v1's keys) are accepted as aliases. You can also pass an explicit
  `dims={"position": [s, e], "rotation": [s, e], "gripper": [s, e]}`.
- **D5 gripper array shape.** GR00T returns `gripper_close` as (T, 1). The author's `action_magnitude` takes
  `np.diff` over the last axis of a (k, 1) array. That axis has length 1, so the diff is empty and the
  0.2 · toggle term is always 0. The port keeps this behaviour (the term is inert).
- **D6 protocol.** Episodes run under HarnessStudio's seeded RoboCasa365 protocol (official horizons,
  native base_motion/control_mode, native success). In v2 this means vla-eval's RoboCasa365Benchmark with
  the `robocasa365` contract. The author used an old RoboCasa client (max_steps 600, base zeroed, 3-step
  success hold).
- **D7 noise bank across decisions.** On every call the author server reseeds random, numpy, torch and cuda
  to 42 and sets cudnn to deterministic (`gr00t/model/policy.py` L198-208, `seed_per_step=True`), so every
  decision reuses the same (20, 16, D) noise bank, sample 0 included. HarnessStudio's seeding belongs to
  the core (`core.noise.crn_seed`, fresh per decision) and keeps sample 0 equal to the baseline/null
  native stream. Not reproduced.

## v1 → v2 port notes (not paper deviations)

- **P1 chunk length.** The author code hardcodes the motion range k = 2..16 and the fallback 16. v2 uses
  the decision's chunk length T instead (range 2..T, fallback T), which is the author code when T = 16
  (GR00T). v1 refused T ≠ 16; `strict_horizon=True` restores that. With T < 2 there is no elbow (the diff
  of one value is empty) and no motion range, so the whole chunk runs.
- **P2 layout.** v1 read `extras["policy"]["action_layout"]`. v2 gets the layout from `bind(embodiment)` or
  `dims`. AAC uses no proprioception, so `embodiment.eef_position` is not needed.
- **P3 diagnostics.** `AACEntropyExecutor.diagnostics()` returns the same evidence as v1 (k, k_entropy,
  k_motion, per-step entropies, chunk means, motion, `proposal0_equals_chunk`). The v2 DecisionLoop writes
  them to the trace after every decision, in every arm (`decision` events, `diagnostics.executor`;
  v1 traced them as `executor_info`).

### On the fake stack (tier 1: 7-D position/rotation/gripper, chunk length 6)

The elbow is not degenerate there, only short. With T=6, `diff(chunk_mean)` has 5 entries, so
`k_entropy` ∈ [2, 5], `k_motion` ∈ [2, 6] and k ∈ [2, 6]. The fake policy's actions are uniform in
[−1, 1]: the gripper is continuous, not binary, so the Bernoulli mean is clipped to 1e-9 when negative.
Its per-step entropies are nearly flat, so the elbow follows sampling noise, and the motion floor (>3.0)
is usually crossed within 2–4 steps. In the check run the candidate planned k ∈ {2, 3, 4, 5}. The null
stays bit-identical to the baseline, and the candidate uses 2× the null's samples because it re-plans
about every 3 steps.

## Install and verify

From the repository root:

```bash
~/.local/bin/uv pip install --python .venv/bin/python -e methods/aac
.venv/bin/hs list
.venv/bin/hs verify aac --tier 0       # registration, artifact hash, buildable components
.venv/bin/hs verify aac                # tier 1 on the fake stack (fake-t1 profile)
```

Behavioural checks ported from v1 `tests/test_aac.py`:

```bash
.venv/bin/python methods/aac/check_aac.py                              # executor + config + real fake-stack arms (T=6 and T=16)
/home/sejain/miniconda3/envs/gr00t/bin/python -I methods/aac/check_aac.py  # + bitwise equality with the pinned author code (needs scipy)
```

The upstream section loads only the named functions from the hash-pinned author files
(`HARNESSSTUDIO_AAC_UPSTREAM`, default: the RoboMetaHarness-claude paper-repro checkout) and compares
k, per-step entropies, chunk means and motion over 600 random cases. It also checks that the executor
gives the same k in both the v2 layout and the v1 GR00T-key layout.

## Run on GR00T N1.5 / RoboCasa365

Start a GR00T N1.5 RoboCasa365 policy server that speaks the HarnessStudio seeded-sampling extension
(profile `gr00t-n15-robocasa365`: 16-step chunks; `seeded_sampling` and `isolated_streams` are required).
Then run the three paired arms against a vla-eval RoboCasa365 config, for example vla-eval v0.8.0's
`configs/benchmarks/robocasa365/smoke.yaml`:

```bash
.venv/bin/hs eval aac \
  --policy-url ws://localhost:8000 \
  --profile gr00t-n15-robocasa365 --contract robocasa365 \
  --config /home/sejain/research/cvpr27/vla-eval-v0.8.0/configs/benchmarks/robocasa365/smoke.yaml \
  --arms baseline,null,candidate --out runs/aac-rc365-smoke
```

Cost: 20 policy samples per decision in both the null and the candidate arm. The candidate re-plans
every k ∈ [2, 16] steps, so it makes more decisions than the null, which runs all 16 steps.
`budgets.policy_calls` is `null` (bounded by the horizon), as in v1. Start with a small, predeclared,
paired subset before running full suites.
