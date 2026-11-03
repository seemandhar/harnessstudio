<h1 align="center">HarnessStudio</h1>

<p align="center">
  <b>Training-free harnesses around frozen robot policies, as installable methods.</b><br>
  One paper = one package. Verify it in seconds, then run it against matched controls on real benchmarks.
</p>

A *harness* is everything around a frozen robot policy that is not its weights: what it observes, which
instruction it gets, how many samples are drawn and which one runs, how much of a chunk executes, when
execution is interrupted, which scripted skills run instead of the policy, and which LLM agent plans the
episode. HarnessStudio turns each published harness into a **method package** that plugs into typed
slots around a frozen policy, the way nerfstudio turns NeRF variants into method configs.

v2 runs on [allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)
(`vla-eval==0.8.0`, a pip dependency, not a fork), so every method runs on its benchmarks (LIBERO,
LIBERO-Pro, RoboCasa365, ...) and with its policy servers (π0.5, GR00T, ...).

```
vla-eval benchmark (Docker, unchanged)  ◄─ws─►  HarnessStudio harness server  ◄─ws─►  policy server
 thinks it talks to a policy                    method + components + tools          frozen π0.5 / GR00T / ...
                                                budgets, traces, paired arms
```

## Install

```bash
git clone -b v2 https://github.com/seemandhar/harnessstudio && cd harnessstudio
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e '.[test]'
scripts/install_methods.sh        # every package under methods/ (editable), then `hs list`
```

## Quick start

```bash
hs list                                   # installed methods, categories, required capabilities
hs verify identity --tier 0               # static checks
hs verify medoid-bon                      # tier 1: baseline/null/candidate through real vla-eval on a fake stack (CPU, seconds)

# real benchmark: run one arm's harness server in front of any vla-eval policy server
hs serve identity --policy-url ws://localhost:8101 --contract libero --profile pi05-libero-stock --port 8201
vla-eval run --config my_libero.yaml      # with server.url: ws://localhost:8201

# tier 2: all three arms on a few real cases (gate: every episode valid, null vs baseline has 0 X)
hs verify medoid-bon --tier 2 --config my_libero.yaml --policy-url ws://localhost:8106 --contract libero \
   --profile pi05-libero-hs --out runs/t2 --benchmark-overrides '{"max_tasks": 1, "episodes_per_task": 2}'

# or let hs run all three arms and pair them; hs report prints the paired summary
hs report runs/medoid-bon
hs eval medoid-bon --config my_libero.yaml --policy-url ws://localhost:8101 --contract libero \
   --profile pi05-libero-hs --out runs/medoid-bon
```

## Methods are packages

```
methods/<slug>/
  pyproject.toml          # [project.entry-points."harnessstudio.methods"] <name> = "hs_<slug>:config"
  hs_<slug>/paper.yaml    # paper card: categories, fidelity, substitutions, capabilities, artifacts, LLM operators
  hs_<slug>/__init__.py   # components + config()
```

`hs new <slug> --category <id> [--kind controller --llm]` scaffolds one that already passes tier 1;
`pip install -e methods/<slug>` and it shows up in `hs list`. Full guide: [docs/writing-a-method.md](docs/writing-a-method.md).

**Slot methods** fill the default decision loop:
`observation_op → instruction_op → gate → proposer (+ sampler in the policy server) → selector → action_op → executor (+ monitor)`.
**Controller methods** own the episode through `ctx` (`observe`, `policy.act/sample`, `execute`, `decide`,
`tools["llm"]`), for agent harnesses such as LLM planners.

Slots map onto the 12 harness categories shared with RoboMetaHarness:

| Category | Component |
|---|---|
| observation_grounding | `observation_op` |
| language_alignment | `instruction_op` |
| proposal_selection | `proposer` + `selector` |
| sampler_guidance | `sampler` (in the policy server) |
| uncertainty_compute | `gate` |
| execution_monitoring, contact_feedback | `executor` + `monitor` |
| memory_retrieval | `action_op` (+ experience store) |
| agentic_orchestration | controller |
| skill_composition | skills (planned) |
| world_model_verification | `selector` over predicted futures (planned) |
| experience_evolution | `prepare` stage (planned) |

## Fair by construction

Every comparison runs three arms on the same seeded episodes:

| Arm | What runs |
|---|---|
| baseline | the native policy (sample 0 noise stream, full native chunk, original instruction) |
| null | pays the candidate's compute but executes natively (derived automatically for slot methods; *shadow* mode for controllers) |
| candidate | the method |

Methods only ever see the public observation (the contract's allowlist); the policy still gets the raw
observation, so the baseline is exactly the native vla-eval policy. Per-step observation and action
hashes give each null/baseline pair an **M** (matched), **R** (environment branched, e.g. renderer) or
**X** (harness leak) label. Budgets cap policy samples, LLM calls and tokens; a spent budget ends method
control and the episode finishes natively. Runtime LLMs are pinned (one exact model, `/models` check, no
fallback).

## Policy capability tiers

| Tier | Server | Unlocks |
|---|---|---|
| T0 | any vla-eval policy server, unchanged (`allenai-passthrough`, `pi05-libero-stock`) | agent, instruction and monitoring methods |
| T1 | HarnessStudio subclass via `HarnessPolicyMixin`: seeded, isolated full chunks | best-of-N, executors, exact derived nulls |
| T2/T3 | denoising hooks, velocity / attention / logits / predicted futures | steering and introspection methods |

## Methods

| Method | Paper | Fidelity | Category | Needs |
|---|---|---|---|---|
| `identity` | none (native policy) | faithful | - | T0 |
| `medoid-bon` | CycleVLA, arXiv:2601.02295v2 (MBR selection only) | component | proposal_selection | T1 |
| `aac` | AAC, arXiv:2604.04161v2 (disclosed D1-D7) | faithful | uncertainty_compute | T1 |
| `chunk-interrupt` | VLA-Corrector, arXiv:2607.01804v1 (truncation only) | tf_variant | execution_monitoring | T0 + chunk |
| `subtask-sequencer` | Inner Monologue, arXiv:2207.05608v1 (rule-based, no LLM) | tf_variant | agentic_orchestration | T0 + chunk |
| `llm-subtask-planner` | Inner Monologue, arXiv:2207.05608v1 (NRP qwen3-vl) | tf_variant | agentic_orchestration | T0 + chunk, LLM |
| `chunktrust-ahs` | ChunkTrust, arXiv:2609.39754v2 (adaptive horizon from flow velocity) | faithful | execution_monitoring | T1 + velocity readout |
| `harness-vla` | Harness VLA, arXiv:2607.08448 (zero-shot; NRP qwen3-vl planner) | adapted | agentic_orchestration | T0 + chunk, LLM, SAM3 |
| `harness-vla-sol` | same, gpt-6.1-sol planner | adapted | agentic_orchestration | T0 + chunk, LLM, SAM3 |
| `harness-vla-sol-nocue` | same, planner never told that the goal is not reached (fair ablation) | adapted | agentic_orchestration | T0 + chunk, LLM, SAM3 |
| `harness-vla-sol-fewshot` | same, few-shot task memory built by `hs prepare` | adapted | agentic_orchestration | T0 + chunk, LLM, SAM3 |
| `scale` | SCALE, arXiv:2602.04208 (OpenVLA uncertainty-aware decoding) | faithful | uncertainty_compute | T1 + decoding_hooks (`openvla-libero-hs`) |
| `fast-harness` | FastHarness, arXiv:2610.31427 (gpt-6.1-sol reviewer) | adapted | uncertainty_compute | T0 + chunk, LLM (`robodojo-pi05-hs`) |
| `rtcf` | RTCF, arXiv:2608.04527v2 (withdrawn v3): PMA retrieval over a frozen success memory + low-frequency DCT residual; PCA omitted | tf_variant | memory_retrieval | T1 + chunk, `hs prepare` memory, SigLIP service (pi0-FAST z-score, exec 5) |
| `first-sample` | example plugin (`examples/plugin_method`) | faithful | proposal_selection | T1 |
| `u5z-t` | none (RTCF++ discovery, `runs/rtcfpp-20261101`): success-tube orientation steering (T, memory from reserved init 10-29) + zero-chunk substitution before T's first arming (Z); admitted under a user waiver, docs/results/u5z.md | tf_variant | memory_retrieval | T1 + chunk (pi0-FAST z-score, exec 5) |
| `tube-steer`, `tube-steer-ortho` | T alone (U5+Z component); **negative as a unit** (Campaign-30 net 0) | tf_variant | memory_retrieval | T1 + chunk |
| `tilt-guard-v12-*` | tilt envelope B without G; source of U5+Z's Z; **negative** (B hard48 0/0) | tf_variant | execution_monitoring | T1 + chunk |
| `band-guard`, `band-guard-signctrl` | rx saturation band guard; **negative** (hard48 0/0) | faithful | execution_monitoring | T1 + chunk |
| `tilt-guard-*`, `tilt-guard-v11-*` | G/B tilt guards (v1.1 adds Bsat); **negative** (G+B hard48 0/0; Bsat not screened) | tf_variant | execution_monitoring | T1 + chunk |
| `redecode`, `redecode-b` | re-query zero chunks with a +-1-bin state shift; **negative** (hard48 +1/-0, -1 vs its z control) | component | uncertainty_compute | T1 + chunk |
| `tube-steer-vfb`, `tube-steer-vfb-ortho` | T with a SigLIP visual-retrieval fallback for task 8; **negative** (t8 0/8); needs the uncommitted `visual_memory.npz` | tf_variant | memory_retrieval | T1 + chunk, SigLIP service |
| `prr-*`, `prr-v11-*` | pose restore and re-decide controller; **negative / blocked** (restoration 0/8; 1.1.0 re-run blocked) | tf_variant | execution_monitoring | T1 + chunk |
| `bcp`, `bcp-switch`, `bcp-switch-dot` | boundary clause prompt (shadow kill test K1); **blocked, not scored** | tf_variant | language_alignment | T1 + chunk |
| `vlm-rehome-*` | VLM-flagged re-home (NRP qwen3-vl); **negative** (VLM qualification failed), partial controller, not installed | faithful | execution_monitoring | T0 + chunk, LLM |

All pass tier 0 and tier 1. Ported v1 methods agree with v1 decision for decision on randomized cases
(medoid-bon, chunk-interrupt: 300 each; AAC: bitwise against the authors' code on 600 cases;
chunktrust-ahs: 100% over ~17.8k decisions).

## Policy servers

| Profile | Server | Tier |
|---|---|---|
| `allenai-passthrough`, `pi05-libero-stock` | any unchanged vla-eval policy server | T0 (server chunking) |
| `pi05-libero-hs` | `harnessstudio/policies/pi05_openpi.py` (openpi pi0.5, seeded) | T1 |
| (run profile) | `harnessstudio/policies/pi05_openpi.py --config harnessstudio/policies/configs/pi0_fast_libero_zscore.yaml` (openpi pi0-FAST LIBERO, greedy; `norm_mode: zscore`, the checkpoint's training normalization, EXT openpi-norm) | T1 |
| `gr00t-n15-robocasa365` | `harnessstudio/policies/gr00t_n15_robocasa365.py` (GR00T N1.5, seeded, velocity readout) | T1 (+ velocity) |
| `methods/rtcf/configs/pi0fast-libero-hs.yaml` (path profile, 5 executed actions/call) | `harnessstudio/policies/pi05_openpi.py` with `config_name=pi0_fast_libero`, **served with z-score normalization** (the checkpoint's training norm; openpi 981483d defaults to quantile and breaks it: `--args.norm_mode` after EXT openpi-norm, docs/results/rtcf.md) | T1 (greedy FAST decoding) |
| `fake-t0`, `fake-t1` | `harnessstudio/policy_ext/fake.py` (tier 1) | T0 / T1 |

`pi05_openpi.py` takes `--args.norm_mode=config|zscore|quantile` (default `config`: the pinned openpi TrainConfig,
unchanged). openpi >= a3d8ea3 (2025-07-05) serves every non-PI0 config with quantile norm. Verified for one checkpoint
only: `pi0_fast_libero` (created 2025-05-28, before that switch) needs `zscore` (`docs/extensions/openpi-norm.md`).
Other checkpoints may need it: verify each one (config history at its training date, plus a FAST token-grid or
demo-frame check). A forced mode is
announced in the HELLO action-spec description and so recorded in every run manifest's `policy_hello`.

## Building blocks for agent harnesses

- `harnessstudio.skills.eef_delta`: analytic `move_to`, `move_by`, `set_gripper`, `release`, and
  `vla_act` (the frozen policy as a skill).
- `harnessstudio.tools.llm`: pinned, budgeted runtime LLM/VLM operators (NRP by default; Astra refused).
- `hs prepare` + `harnessstudio.components.memory`: development runs -> sha256-pinned artifacts
  (e.g. trajectory memory); evaluation refuses development cases.
- `harnessstudio.benchmarks.libero_rgbd`: LIBERO / LIBERO-Pro with metric depth and camera calibration
  (sensor data only), plus `harnessstudio.tools.geometry` pixel <-> world helpers (pixel convention
  validated on real LIBERO frames).

## Results so far

- **Transparency (M1):** on real LIBERO, `identity` through the harness is bitwise identical to the
  direct vla-eval run, 5/5 episodes, every action and state ([docs/results/m1_libero_parity.md](docs/results/m1_libero_parity.md)).
- **Paired pipeline:** `hs eval` with three arms on real LIBERO and the seeded pi0.5 server: all episodes
  valid, null vs baseline 5 M / 0 R / 0 X ([docs/results/m3_libero_eval_pipeline.md](docs/results/m3_libero_eval_pipeline.md)).
- **Seeded servers:** pi0.5 and GR00T N1.5 return bitwise-identical chunks for the same seed, in one
  process and after a restart; samples are isolated per seed.

## Paper reproductions (2026-10-31, predeclared subsets, paired arms)

| Method | Benchmark | Frozen policy | Harness | Paper |
|---|---|---|---|---|
| Harness VLA zero-shot, gpt-6.1-sol planner, no success cue (fair) | LIBERO-Pro, all 8 cells, 160 cases | 39/160 (24%) | **75/160 (47%)**, p = 1.4e-5 | Goal-T 79.0, Goal-S 31.0 (Claude Code) |
| same, planner told "goal not reached" (oracle cue) | LIBERO-Pro, all 8 cells, 160 cases | 39/160 (24%) | 84/160 (52.5%), p = 2.4e-7 | |
| same, with the cue | LIBERO-Pro Goal-T / Goal-S, 50 + 50 | 8/50, 15/50 | 27/50, 28/50 | 79.0, 31.0 |
| Harness VLA few-shot, with the cue | LIBERO-Pro Goal held-out, 60 exact cases | 15/60 | 38/60 (zero-shot 33/60) | 87.0 / 87.0 |
| SCALE | LIBERO-10, 60 cases (OpenVLA) | 29/60 (48%) | **38/60 (63%)**, p = 0.093 | 52.7 -> 63.3 |
| FastHarness | RoboDojo build_tower / stack_bowls | - | token savings not reproduced on held-out layouts | -57% reviewer tokens |
| RTCF (withdrawn paper), disjoint dev memory (init 10-29), cadence 5, z-score serving | LIBERO-Long, complete seeds 101, 102, 103, 104, 105, 500 paired cases (pi0-FAST) | 319/500 (63.8%) | 311/500 (62.2%), 49/57, p = 0.5, delta -1.6 pp [-7.27, 4.07] | 61.6 -> 68.6 (withdrawn v2, n = 500) |
| PCD (`pcd`, arXiv:2505.13255; adapted: float32 action wire, HS transport, env identity, SAM2 _C unresolved; [docs/results/pcd.md](docs/results/pcd.md)) | SIMPLER Google robot, authors' seeded protocol, Pick Coke Can + Close Drawer, 64 paired cases (subset only: 2 of 9 tasks, chosen for the largest reported gains) (OpenVLA-7B) | 22/64 (34.4%) | **36/64 (56.3%)**, 17/3, p = 0.00258, CI [+0.090, +0.335]: gain reproduced in direction, magnitude compatible; Move Near 19 -> 21/32 (p = 0.69, descriptive); libero_goal transfer 23 -> 23/30 | per-task 18.0 -> 45.3, 47.3 -> 73.3 (n = 300 each); 9-task 16.8 -> 25.3 not claimed |
| U5+Z (`u5z-t`, new harness, RTCF++ run; admitted under a post-result user waiver of 3 A7 rules) | LIBERO-Long final test init 0-9 x seeds 101-105, 500 paired cases (pi0-FAST; prior exposure via the RTCF run) | fresh 320/500 (64.0%); RTCF recorded 311/500 | **355/500 (71.0%)**, vs baseline 44/9 p = 1.2e-6, vs RTCF 78/34 p = 3.9e-5; validation 81/120 vs 72/120, p = 0.0117; run criterion >= 359 NOT met | - (no paper) |

All with null vs baseline 0 X. Caveat for Harness VLA: the original prompt tells the planner the goal is not yet
reached (LIBERO ends episodes on success, an oracle a real robot lacks); the no-cue row is the fair number. Details, disclosures and an independent compliance audit: `docs/results/`
(`OVERNIGHT-2026-10-31.md` is the summary). Adapted = the paper's LLM was substituted (Astra is not used).

## Status

v2.0.0.dev0. See [docs/superpowers/specs/2026-10-31-harnessstudio-v2-design.md](docs/superpowers/specs/2026-10-31-harnessstudio-v2-design.md)
for the design and roadmap, and `docs/results/` for gate results.

Since 2026-10-31 v2 is `main`. HarnessStudio v1 (the `rh` CLI, RoboCasa365 GR00T/Pi0.5 slots and the
RoboMetaHarness Campaign integration) is preserved unchanged at tag `v1-final` and branch `v1` (local checkout:
`RoboMetaHarness-claude/harnessstudio_v1_old`). Runtime model APIs and local services: `docs/ENDPOINTS.md`;
framework changes go through `docs/extensions/` and `scripts/regress.py`.
