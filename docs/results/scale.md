# SCALE (arXiv:2602.04208) on OpenVLA / LIBERO through HarnessStudio v2

Status: **in progress** (started 2026-10-31 02:40 PT). This file is updated as results land.

## Core requests

1. `harnessstudio/profiles/fake-t2.yaml` should list `decoding_hooks`. The fake policy ignores samplers, but
   without the capability `hs verify scale` (tier 1, default profile) fails with "fake-t2 lacks decoding_hooks".
   Workaround used here: `hs verify scale --profile methods/scale/profiles/fake-t2-decoding.yaml` (a copy of
   fake-t2 plus `decoding_hooks`, inside the method folder).
2. (Nice to have) `hs eval` cannot start at an episode offset or split one arm across two policy servers, so
   the subset below runs some arms as separate `hs eval --arms ...` invocations and pairs them with
   `harnessstudio.eval.pairing.summarize` (script `hs2-gpu/scale/pair_runs.py`). A `--cases`/episode-offset
   option and an `hs pair DIR...` command would make this a single command.

## What was built

| File | What |
|---|---|
| `harnessstudio/policies/openvla_hs.py` (new core file) | `HarnessOpenVLAServer` = vla-eval 0.8.0 `OpenVLAModelServer` + `HarnessPolicyMixin`. Native requests: vla-eval's greedy `predict_action` unchanged. Seeded requests: `sampler=None` -> the same native greedy prediction (baseline, null); `sampler={factory, params}` -> imports the factory (module prefix `hs_`) and runs the official SCALE KV-cached token loop with the decoder's hooks, one `torch.Generator(cuda).manual_seed(seed)` per seed, vision-attention temperature on both ViTs with the stock attention restored afterwards, per-(session, episode, sampler) decoder state reset at EPISODE_START. Capabilities `act, chunk, seeded_sampling, isolated_streams, decoding_hooks`. |
| `harnessstudio/profiles/openvla-libero-hs.yaml` (new core file) | those capabilities, `chunking: harness`, `execute_horizon: 1` (OpenVLA returns one action per step). |
| `methods/scale/` (`hs-scale`, entry point `scale`) | `hs_scale.decoder:ScaleDecoder` (the algorithm: Eq. 2-8, official arithmetic op for op), `hs_scale:SingleSample` proposer (K = 1), `paper.yaml` (fidelity `faithful`, requires `decoding_hooks`), tier-1 profile copy. |
| `hs2-gpu/configs/openvla_libero10_hs.yaml`, `hs2-gpu/configs/libero10_scale.yaml`, `hs2-gpu/start_openvla.sh` | server launch config (same args as vla-eval's `openvla/libero_10.yaml`), LIBERO-10 benchmark overlay (Docker, GPU 5), launcher. |
| `hs2-gpu/scale/{obs_recorder.py, check_server.py, agree_official.py}` | observation recorder relay, wire-level server checks, token-level agreement with the official code. |

SCALE hyperparameters (official `configs/scale.yaml`, "tuned on LIBERO-Long and applied unchanged"):
`T0 = 1.0, epsilon = 1e-12, num_logits = 256, kappa = 2.0, alpha = 0.8, attn_sensitivity = 0.3`.

## Predeclared subset (written before any arm was run)

- Benchmark: vla-eval 0.8.0 LIBERO, suite **libero_10 (Long)**, all **10 tasks x episodes 0, 1, 2 = 30 cases**,
  benchmark seed 7, `num_steps_wait = 10`, official horizon `max_steps = 520`, Docker image
  `ghcr.io/allenai/vla-evaluation-harness/libero:latest`, rendering on GPU 5. The OpenVLA server's HELLO sets
  the benchmark `seed: 0` observation param exactly as in vla-eval's reproduction (env seed 0).
- Policy: `openvla/openvla-7b-finetuned-libero-10` (vla-eval reproduction: 52.8% on LIBERO-Long, paper 53.7%;
  SCALE paper's greedy: 52.7%), `unnorm_key: libero_10`, `jpeg_roundtrip: true`, `center_crop: true`, bf16.
- Arms: **baseline** (native greedy), **null** (derived: same K = 1 request, sampler off = greedy), **candidate**
  (SCALE, paper defaults). Common random numbers: candidate seed per decision = HarnessStudio's
  `sha256(namespace, benchmark, task, episode, decision, 0)`.
- Outcome: successes per arm; paired wins/losses of candidate vs baseline and vs null with the exact two-sided
  sign (McNemar) p from `harnessstudio.eval.pairing`; null vs baseline M/R/X; wall time.
- Decision rule: extend (more libero_10 episodes, then other suites) **only if the complete 30-case subset
  shows candidate > baseline and candidate > null**. With n = 30, no claim of significance is made unless p < 0.05.

## Verification

**Tiers 0/1** (CPU): `hs verify scale --tier 0` PASS; `hs verify scale --profile methods/scale/profiles/fake-t2-decoding.yaml`
PASS (all three arms valid, derived null reproduces baseline actions, candidate deterministic, case-order
independent, clean-interpreter import). Without `--profile` tier 1 fails only on the missing capability (Core request 1).

**Server = stock vla-eval, natively.** One LIBERO-10 episode (task 0, episode 0; 520 steps, a failure) run directly
(no harness) against vla-eval's stock `openvla.py` server and against `openvla_hs.py`: every recorded action and
state identical (sha256 `910cd4f417578a4f` for both, first differing step: none). `hs2-gpu/compare_runs.py`.

**Wire-level checks** (`hs2-gpu/scale/check_server.py`, first 40 recorded observations of that episode,
`results/scale/check_server_t0e0.json`):

| Check | Result |
|---|---|
| a. native request to the HS server == stock server's recorded reply; == live stock server | 40/40; 40/40 |
| b. seeded request without sampler (baseline/null arms), seeds 11 and 12, == native | 40/40 |
| c. SCALE over the 40-step sequence, two fresh episodes, same seeds: bitwise identical actions and tokens | 40/40 |
| d. isolated streams: seed 7 inside [5, 7, 9] == [7] alone (same state) | yes |
| e. native replies after SCALE requests unchanged (vision-attention patch restored) | 40/40 |
| f. SCALE action differs from greedy | 32/40 steps |
| latency (A6000, GPU shared with a second model) | native 0.31 s/step, SCALE 0.30 s/step |

gamma over those steps ranged 0.69-1.68 (first two steps 1.0 by construction); mean per-step tau 0.004-0.54.

**Agreement with the official SCALE code** (`hs2-gpu/scale/agree_official.py`): same process, same loaded
checkpoint, the 120 first recorded observations of the episode above, seed 7 + t at step t. Ours =
`HarnessOpenVLAServer.sample_chunks(..., sampler=ScaleDecoder(paper defaults))`; official = the official
`OpenVLAForActionPrediction.predict_action(decoding_mode="scale", **configs/scale.yaml)` called on the same model
object after `torch.manual_seed(seed)`, `uncertainty_history` threaded across steps as in `run_libero_eval.py`.

| Quantity (120 steps) | Exact matches |
|---|---|
| executed action (7-D, after gripper mapping) | 120/120 |
| per-token self-uncertainty u (7 per step) | 120/120 steps |
| vision-attention temperature gamma | 120/120 |
| episode state (EMA u, last u) after the step | 120/120 |
| official greedy `predict_action` vs vla-eval native greedy | 120/120 |

**Token ids** (rerun capturing the official ids directly, 150 steps, seed 7 + t,
`results/scale/agree_official_t0e0_tokens.json`): **1050/1050 action tokens identical**, and again 150/150 for
actions, u, gamma, state and greedy. (A first pass inverted the de-tokenization instead and reported 756/840:
tokens 31744 and 31745 both clip to bin 254, the same action, so the inversion cannot tell them apart; the
actions were identical in that pass too.) The official `modeling_prismatic.py` differs from the hub's
`openvla/openvla-7b` file only in the docstring, the SCALE helpers and `predict_action`; the model forward is
identical, so the decoder is checked on exactly the computation the official code runs (with the server's
attention kernel). Fidelity stays `faithful`.

**Tier 2** (2 real LIBERO-10 cases: tasks 0 and 1, episode 0; all three arms on server 8115):

```bash
cd /home/sejain/research/cvpr27/vla-eval-v0.8.0
hs verify scale --tier 2 --config /home/sejain/research/cvpr27/hs2-gpu/configs/libero10_scale.yaml \
  --policy-url ws://127.0.0.1:8115 --contract libero --profile openvla-libero-hs \
  --out /home/sejain/research/cvpr27/hs2-gpu/results/scale/tier2 --benchmark-overrides '{"max_tasks": 2, "episodes_per_task": 1}'
```

`tier 2: scale: PASS`: all episodes valid; null vs baseline **2 M / 0 R / 0 X**; baseline 1/2, null 1/2, candidate 2/2
(SCALE solved task 0 episode 0, which greedy fails at the 520-step horizon). Wall 1304 s for 6 episodes (the GPU
was shared with a second OpenVLA server). Not part of the subset numbers below.

## Results

Subset runs (started 03:05 PT): candidate on server 8116; baseline then null (tasks 0-2) on server 8115; null
(tasks 3-9) on 8116 after the candidate. Both servers are the same file, config and checkpoint on GPU 5, so the
null-vs-baseline attribution also checks cross-process determinism for tasks 3-9.

(pending)

## Deviations and disclosures

- Random stream: one torch generator per (decision, seed) from HarnessStudio's common random numbers, instead of
  the official script's single global stream seeded once per run (`set_seed_everywhere(7)`). The per-token
  arithmetic is unchanged; only which uniform draws feed `torch.multinomial` differs.
- Image preprocessing and LLM attention kernel are vla-eval's OpenVLA server's for every arm: PIL JPEG
  round-trip, Lanczos resize, PIL center crop (official: TF JPEG round-trip + TF `crop_and_resize`), and the
  transformers default attention for the LLM (official `get_vla` asks for flash-attention-2). These are policy-
  server settings shared by baseline, null and candidate, not part of SCALE.
- Prompt: vla-eval's `task_description` as sent (the official script lower-cases it; LIBERO-10 instructions
  are already lower case).
