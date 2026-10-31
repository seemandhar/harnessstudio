# SCALE (arXiv:2602.04208) on OpenVLA / LIBERO through HarnessStudio v2

Status (2026-10-31 09:45 PT): port done and verified; libero_10 subset + extension and a libero_goal subset complete;
official-pipeline reference runs in progress.

## Summary

- `scale` (faithful) runs inside a new OpenVLA policy server (`decoding_hooks`); its decoder matches the official
  SCALE `predict_action` token for token (1050/1050 tokens, 150/150 actions, u, gamma and state on the same
  checkpoint and seeds). The server's native path is bitwise the stock vla-eval server (a 520-step episode).
- Tier 0/1 PASS; tier 2 PASS (2 M / 0 R / 0 X).
- LIBERO-10, 60 paired cases (predeclared 30 + 30 extension): baseline 29/60, null 29/60 (60 M / 0 R / 0 X),
  **SCALE 38/60, +16 / -7, sign test p = 0.093** (paper: 52.7 -> 63.3). Same direction and size as the paper,
  not significant at this n.
- LIBERO-Goal, 30 paired cases: baseline 23/30, null 23/30 (30 M), SCALE 24/30, +6 / -5 (p = 1.0): no effect seen.

## Core requests

Both were implemented by the coordinator in `e68c01b` (2026-10-31): `fake-t2.yaml` now lists `decoding_hooks`
(plain `hs verify scale` passes; the method-local profile copy was removed) and `hs pair --arm ARM=DIR ...`
pairs separately run arms. Still open (minor): `hs pair` keeps only the last `--arm` per arm name, so an arm
split across two servers must be concatenated into one `records.jsonl` first (done here); vla-eval cannot start
at an episode offset (worked around below with a benchmark subclass, `hs2-gpu/scale/bench/libero_offset.py`).

## What was built

| File | What |
|---|---|
| `harnessstudio/policies/openvla_hs.py` (new core file) | `HarnessOpenVLAServer` = vla-eval 0.8.0 `OpenVLAModelServer` + `HarnessPolicyMixin`. Native requests: vla-eval's greedy `predict_action` unchanged. Seeded requests: `sampler=None` -> the same native greedy prediction (baseline, null); `sampler={factory, params}` -> imports the factory (module prefix `hs_`) and runs the official SCALE KV-cached token loop with the decoder's hooks, one `torch.Generator(cuda).manual_seed(seed)` per seed, vision-attention temperature on both ViTs with the stock attention restored afterwards, per-(session, episode, sampler) decoder state reset at EPISODE_START. Capabilities `act, chunk, seeded_sampling, isolated_streams, decoding_hooks`. |
| `harnessstudio/profiles/openvla-libero-hs.yaml` (new core file) | those capabilities, `chunking: harness`, `execute_horizon: 1` (OpenVLA returns one action per step). |
| `methods/scale/` (`hs-scale`, entry point `scale`) | `hs_scale.decoder:ScaleDecoder` (the algorithm: Eq. 2-8, official arithmetic op for op), `hs_scale:SingleSample` proposer (K = 1), `paper.yaml` (fidelity `faithful`, requires `decoding_hooks`). |
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

**Tiers 0/1** (CPU): `hs verify scale --tier 0` PASS; `hs verify scale` (fake-t2) PASS: all three arms valid,
derived null reproduces baseline actions, candidate deterministic, case-order independent, clean-interpreter
import. (Before `e68c01b` this needed a profile copy with `decoding_hooks`.)

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

### Predeclared subset: libero_10, 10 tasks x episodes 0-2 (complete)

| Arm | Successes | vs baseline (wins / losses, sign-test p) | Policy samples | Arm wall time |
|---|---|---|---|---|
| baseline (greedy) | **16/30** | - | 11,837 | 6,090 s |
| null (derived, greedy) | **16/30** | +0 / -0 (p = 1.0) | 11,837 | 1,922 s + 4,203 s (two servers, in parallel with other arms) |
| candidate (SCALE) | **20/30** | **+10 / -6 (p = 0.45)** | 10,958 | 5,846 s |

- Candidate vs null: +10 / -6 (p = 0.45). All 90 episodes valid.
- Null vs baseline: **30 M / 0 R / 0 X** (every observation and action hash equal at every step). For tasks 3-9 the
  null ran on the second server process (8116) and the baseline on 8115, so this is also a cross-process bitwise
  determinism check of the greedy path.
- Baseline 16/30 = 53%, in line with vla-eval's reproduction (52.8%) and the SCALE paper's greedy (52.7%).
  Candidate 20/30 = 67% (paper: 63.3%). With n = 30 and 16 discordant pairs the difference is **not significant**.
- Per task (baseline -> candidate, of 3): alphabet soup + tomato sauce 1 -> 3; cream cheese + butter 3 -> 3;
  stove + moka pot 1 -> 3; black bowl in drawer 2 -> 2; white mug left plate + yellow-white mug right plate
  3 -> **0**; book in caddy 3 -> 2; white mug on plate + pudding 0 -> 2; alphabet soup + cream cheese 0 -> 3;
  both moka pots 1 -> 1; yellow-white mug in microwave 2 -> 1. The gain is concentrated in four tasks and one
  task regresses completely.
- Steps executed: baseline 11,837, candidate 10,958 (SCALE costs the same one forward pass per step; it is not
  slower per step, 0.30 vs 0.31 s on the check above).
- Decision rule (candidate > baseline and > null on the complete subset): **met**, so the extension runs.

Commands (from `/home/sejain/research/cvpr27/vla-eval-v0.8.0`; `hs2-gpu/scale/run_arm.sh` wraps them):

```bash
hs eval scale --config /home/sejain/research/cvpr27/hs2-gpu/configs/libero10_scale.yaml --policy-url ws://127.0.0.1:8116 \
  --contract libero --profile openvla-libero-hs --arms candidate --benchmark-overrides '{"episodes_per_task": 3}' \
  --out /home/sejain/research/cvpr27/hs2-gpu/results/scale/subset_l10_cand
hs eval scale ... --policy-url ws://127.0.0.1:8115 --arms baseline --out .../subset_l10_base
hs eval scale ... --policy-url ws://127.0.0.1:8115 --arms null \
  --benchmark-overrides '{"episodes_per_task": 3, "tasks": [<tasks 0-2>]}' --out .../subset_l10_null_a
hs eval scale ... --policy-url ws://127.0.0.1:8116 --arms null \
  --benchmark-overrides '{"episodes_per_task": 3, "tasks": [<tasks 3-9>]}' --out .../subset_l10_null_b
cat .../subset_l10_null_{a,b}/null/records.jsonl > .../subset_l10_null/null/records.jsonl
hs pair --arm baseline=.../subset_l10_base/baseline --arm null=.../subset_l10_null/null \
  --arm candidate=.../subset_l10_cand/candidate --out .../subset_l10_paired
```

(Task lists: `hs2-gpu/scale/ovr_null_{a,b}.json`. Per-task and per-case table: `results/scale/subset_l10_paired/extra.json`
from `hs2-gpu/scale/pair_runs.py`.)

### Extension: libero_10 episodes 3-5 (complete)

Same setup, LIBERO initial states 3, 4, 5 for every task (30 new cases). vla-eval always runs episodes
0..N-1, so the benchmark is `hs2-gpu/scale/bench/libero_offset.py:LIBEROEpisodeOffset` (vla-eval's
LIBEROBenchmark; `episode_offset = 3` is added to the task's `episode_idx` in place before the runner builds the
EPISODE_START payload, so case keys and seeds carry the true episode index), mounted into the container
(`hs2-gpu/configs/libero10_scale_offset.yaml`). Arms: baseline + null (tasks 0-6) on 8115, candidate + null
(tasks 7-9) on 8116. The extension baseline started at 05:35 PT, before the subset's second null part finished
(05:53); the rule was met, so it counts. It would have been reported but not counted otherwise.

| Arm | Episodes 3-5 (30) | vs baseline | **All libero_10, episodes 0-5 (60)** | vs baseline |
|---|---|---|---|---|
| baseline (greedy) | 13/30 | - | **29/60 (48%)** | - |
| null (derived) | 13/30 | +0 / -0 | **29/60** | +0 / -0 (p = 1.0) |
| candidate (SCALE) | 18/30 | +6 / -1 (p = 0.125) | **38/60 (63%)** | **+16 / -7 (p = 0.093)** |

- Null vs baseline: 30 M / 0 R / 0 X on the extension, **60 M / 0 R / 0 X** over all 60 cases. Candidate vs null is
  identical to candidate vs baseline (+16 / -7).
- Policy samples (= steps): baseline 24,161, candidate 22,133 over 60 cases. Arm wall times on the extension:
  baseline 6,191 s, candidate 5,811 s (two servers sharing GPU 5).
- Per task over 60 (baseline -> candidate, of 6): alphabet soup + cream cheese 0 -> 4, white mug + pudding 2 -> 5,
  alphabet soup + tomato sauce 3 -> 5, cream cheese + butter 5 -> 6, stove + moka 4 -> 5, black bowl drawer 4 -> 4,
  book caddy 5 -> 5, both moka pots 1 -> 1, microwave 2 -> 2, **white mug left / yellow-white mug right 3 -> 1**.
- Reading: the direction matches the paper (SCALE paper, OpenVLA LIBERO-Long: 52.7 -> 63.3; here 48 -> 63 on 60
  paired cases), but the paired sign test gives p = 0.093: **not significant at 0.05** with n = 60. Combined
  tables merge `hs pair` inputs (`results/scale/l10_all/<arm>/records.jsonl` = subset + extension records);
  summaries in `results/scale/{ext_l10,l10_all}_paired/`.

### Second suite (predeclared 05:54 PT, before any libero_goal arm ran): libero_goal

- **libero_goal**, 10 tasks x episodes 0-2 = 30 cases, seed 7, `num_steps_wait = 10`, official horizon 300,
  checkpoint `openvla/openvla-7b-finetuned-libero-goal` (`unnorm_key: libero_goal`; vla-eval reproduction 79.0%,
  SCALE paper greedy 77.7 -> SCALE 82.3), same server file and SCALE defaults (tuned on Long, applied unchanged).
- Arms and statistics as for libero_10. Runs after the libero_10 extension: candidate + null (tasks 4-9) on 8116,
  baseline + null (tasks 0-3) on 8115, each server restarted with the goal checkpoint.

Result (complete, 90 episodes all valid):

| Arm | Successes | vs baseline | Policy samples | Arm wall time |
|---|---|---|---|---|
| baseline (greedy) | **23/30 (77%)** | - | 4,830 | 2,446 s |
| null (derived) | **23/30** | +0 / -0 (p = 1.0) | 4,830 | 1,155 s + 1,304 s (two servers) |
| candidate (SCALE) | **24/30 (80%)** | **+6 / -5 (p = 1.0)** | 4,810 | 2,536 s |

- Null vs baseline **30 M / 0 R / 0 X**; candidate vs null +6 / -5.
- No measurable effect on libero_goal at n = 30: one net success, 11 discordant pairs that nearly cancel
  (paper: 77.7 -> 82.3). Per task, baseline -> candidate (of 3): open top drawer + bowl 1 -> 3, open middle drawer
  1 -> 2, wine bottle on cabinet 2 -> 3; bowl on cabinet 3 -> 2, wine bottle on rack 2 -> 1, turn on stove 3 -> 2;
  the other four unchanged. Not extended (time).

### Official SCALE pipeline, for reference (running)

The official repo's own `run_libero_eval.py` (its LIBERO install, TF preprocessing, flash-attention-2, global
seed 7, `env.seed(0)`), libero_10, `--num_trials_per_task 2` (initial states 0 and 1 of each task), `greedy` and
`scale` concurrently on GPU 5 (`hs2-gpu/scale/run_official.sh`, in the LIBERO container with only GPU 5 visible).
Different random streams and preprocessing from the HarnessStudio runs, so only rates are comparable, not pairs.
(Run outside a container first, robosuite's EGL device index 5 rendered on physical GPU 6; those runs were stopped
after one episode and discarded. TensorFlow's preallocation was turned off with `TF_FORCE_GPU_ALLOW_GROWTH`.)

## Deviations and disclosures

- Random stream: one torch generator per (decision, seed) from HarnessStudio's common random numbers, instead of
  the official script's single global stream seeded once per run (`set_seed_everywhere(7)`). The per-token
  arithmetic is unchanged; only which uniform draws feed `torch.multinomial` differs.
- Image preprocessing and LLM attention kernel are vla-eval's OpenVLA server's for every arm: PIL JPEG
  round-trip, Lanczos resize, PIL center crop (official: TF JPEG round-trip + TF `crop_and_resize`), and the
  the LLM's attention kernel transformers 4.40.1 picks by default for this checkpoint, SDPA (official `get_vla` asks
  for flash-attention-2). These are policy-
  server settings shared by baseline, null and candidate, not part of SCALE.
- Prompt: vla-eval's `task_description` as sent (the official script lower-cases it; LIBERO-10 instructions
  are already lower case).
