# Harness VLA (arXiv:2607.08448), zero-shot mode, on LIBERO-Pro Goal (2026-10-31, overnight)

Status: **in progress** (this file is updated during the night; the last section has the latest state).

## Core requests

None so far (everything needed lives in `methods/harness_vla/`).

## Predeclared subset (written before any scored episode)

| | |
|---|---|
| Benchmark | vla-eval 0.8.0 LIBERO-Pro, base suite `libero_goal`, two cells: `perturbation=task` (Goal-T) and `perturbation=swap` (Goal-S) |
| Cases | all 10 tasks of each cell x episode_idx 0 (1 episode per task) = **20 cases**; seed 7, `num_steps_wait` 10 (vla-eval defaults) |
| Horizon | official: `libero_goal` max_steps **300** (vla-eval `MAX_STEP_MAPPING`, contract `suite_horizons`) |
| Benchmark class | `hs_harness_vla_bench:LIBEROProRGBDInstruction` = HarnessStudio's RGB-D mixin (metric depth + camera calibration for agentview and wrist, sensor data only) over HarnessStudio's `LIBEROProBDDLInstruction` (sends the BDDL instruction for `task`/`lan`) |
| Contract | `methods/harness_vla/hs_harness_vla/contracts/libero_pro_rgbd.yaml` (libero_pro keys + `depth`, `camera`) |
| Policy | seeded openpi π0.5 `pi05_libero` (HarnessStudio T1 server, profile `pi05-libero-hs`, ws://127.0.0.1:8106) |
| Arms | `hs eval harness-vla`: baseline (native π0.5), null (shadow: the controller runs and pays its LLM/SAM3/policy calls, actions native), candidate |
| Reported | successes per arm and cell, paired wins/losses (candidate vs baseline, candidate vs null), null attribution M/R/X, LLM calls/tokens, SAM3 calls, wall time |
| Extension (only if time remains) | episode_idx 1 for the same 20 tasks, reported separately and only if complete |

**Amendment (03:05 PT, before any scored episode was run):** the tier-2 smoke showed about one minute per episode
per arm, so the subset is **10 tasks x 5 episodes (episode_idx 0-4) per cell = 100 cases** (50 Goal-T + 50 Goal-S),
all three arms. Everything else above is unchanged. The ep-0 rows are reported separately too (the original
declaration). No further extension.

No full-benchmark run follows from this subset unless the subset shows an improvement over both baseline and null.
Paper reference (not comparable: Claude Code planner, RLinf `pi05_libero130` policy, 10 seeds/task, 10000-step
env budget): zero-shot Goal-T 79.0%, Goal-S 31.0% (Table 5).

## Setup

| Part | What |
|---|---|
| Method package | `methods/harness_vla/` (entry point `harness-vla`, fidelity `adapted`, null `shadow`); `hs verify harness-vla --tier 0` PASS, tier 1 PASS |
| Controller | `hs_harness_vla/controller.py:HarnessVLA`: planner REPL, one JSON reply per turn (`thought` + 1-3 commands), max 40 turns, last 8 turns kept in context with images only in the newest message, arm holds its pose after `finish` / turn budget |
| Primitives | `hs_harness_vla/primitives.py`: `move_to` (reuses `harnessstudio.skills.eef_delta.move_to`, + RPent's yaw variant), `move_pose`, `rotate_wrist`, `rotate_pitch` (new, axis-angle from `states[3:6]`), `set_gripper`/`release` (`eef_delta`), `vla_act` (frozen policy via `ctx.decide(instruction=prompt)`, chunk by chunk, RPent `pi0_pick` lift predicate or `pi0_doubled` no-stop) |
| Perception | `hs_harness_vla/perception.py`: calibration-frame views (published image mirrored back), `back_project` (pixel / region), `segment` = SAM3 (RPent server, `rpent.robots.components.sam3_server`, GPU 7, :8117) + metric depth -> median world xyz, extents, overlay image |
| Benchmark | `bench_ext/hs_harness_vla_bench.py:LIBEROProRGBDInstruction` (container side, Python 3.8): `harnessstudio.benchmarks.libero_rgbd._RGBDMixin` over `harnessstudio.benchmarks.libero_pro_instructions.LIBEROProBDDLInstruction`, depth + calibration for agentview and wrist |
| Contract | `hs_harness_vla/contracts/libero_pro_rgbd.yaml` (passed to `--contract` by path) |
| Configs | `methods/harness_vla/configs/libero_pro_goal_{task,swap}.yaml`: LIBERO-Pro image, GPU 7 EGL rendering, HF LIBERO-Pro data mounts, repo mounted at `/hs`, `server.timeout: 900` |
| Planner | NRP `qwen3-vl` (served `Qwen/Qwen3.8-Flash-Next-FP8`), `enable_thinking: false`, temperature 0, max 800 output tokens per call; picked after an image probe on a LIBERO frame (kimi: reasoning ate the budget, 80 s, empty content; gemma-vl: 150 s timeout; gemma4-12b: empty content; glm-5: HTTP 400 on images) |
| Budgets / episode | policy 400 samples, LLM 45 calls, 800 tokens/call, SAM3 + back-projection 60 calls |

## Results

(pending)

## What is substituted

Fidelity `adapted` (also in `hs_harness_vla/paper.yaml`):

| Paper / RPent | Here | Why |
|---|---|---|
| Claude Code (claude-opus-4-8) or Codex planner with structured MCP tool calls | NRP `qwen3-vl` (served `Qwen/Qwen3.8-Flash-Next-FP8`), one JSON reply per turn with 1-3 commands | workspace rule: NRP open-weight runtime LLMs only, never Astra; `original_model` kept in the card |
| RLinf `pi05_libero130` fullshot SFT (LIBERO-130) | openpi `pi05_libero` (seeded HarnessStudio T1 server, :8106) | the seeded server gives exactly paired arms (M/R/X); the RLinf checkpoint is on disk (`/bigdata/sejain/frozen-harness-bench/models/rpent-pi05-libero-130-fullshot-sft`) but needs RPent's/RLinf's PyTorch server |
| RPent env server, 10000-step episode budget | vla-eval LIBERO-Pro, official horizon 300 (libero_goal) | task rule: official horizons |
| 1024x1024 re-renders + world maps (`agentview_high.png`, `*_world_high.npz`) | the benchmark's 256x256 agentview + wrist RGB-D, shown upsampled 2x | only what the benchmark publishes |
| `robot0_eef_quat` from the raw sim observation | orientation from `states[3:6]` (axis-angle, public proprio) | same rotation, public key |
| `object_names` in the state view (sim body names) | dropped | not sensor data |
| `pi0_pick` (max_chunks 24) / `pi0_doubled` (20) on RLinf chunks | `vla_act` with stop `object_lifted` / `none`, 10 actions per chunk, default max_chunks 12 | horizon 300 |
| `move_to` / `move_pose` run to max_steps | same control law, but return early when the arm moved < 4 mm in 10 steps (`stalled`) | horizon 300 (first smoke lost 160 of 300 steps to a blocked move) |
| RPent evaluate prompt + guides + memory (MEMORY.md, task-family, seed-0 recipes, PROVEN_LEVERS) | `hs_harness_vla/prompts.py`: role, rules, localization and primitives from the prompt/guides, condensed; every memory-derived part removed | zero-shot mode |
| `finish` ends the episode | the arm holds its pose until the horizon | HarnessStudio would otherwise hand the rest of the episode to the native policy |
| planner keeps its own localization table in its reasoning (RPent PERCEPTION_ALGORITHM step 6) | the controller keeps the table from the planner's own `segment` results and reports the gripper's offset to each entity every turn | the open-weight planner misread positions from the wrist image (dev case: held the bottle right above the cabinet top and moved away) |
| no loop guard | an identical motion command right after it failed/stalled is refused without env steps | dev case: 20 identical stalled `move_to` calls |
| no gripper geometry in the prompt | one sentence on the hand's size (~20 cm across the fingers, ~12 cm tall) and "travel high, descend vertically" | dev cases: the open hand caught the cabinet top edge |

Privileged-state audit of RPent's LIBERO env server (`robots/libero/env_server.py`, `tools.py`): it exposes
`robot0_eef_quat` / `robot0_eef_pos` / `robot0_gripper_qpos` (proprio, public here via `states`), RGB-D and camera
calibration (public here via the RGB-D subclass), and `object_names` (derived from simulator body keys; dropped
here). No object poses reach the planner in RPent's perception-isolated mode either.

## Known gaps

- Planner strength: the paper's planners are frontier coding agents with long tool-use loops; qwen3-vl in
  non-thinking mode is much weaker at 3-D reasoning (e.g. it drives straight into fixtures). Results here measure
  this port with this operator, not the paper's system.
- Step budget: RPent ran with 10000 env steps; the official 300-step horizon leaves room for roughly one grasp
  attempt plus transport.
- Policy: `pi05_libero` is trained on the four LIBERO suites only (not LIBERO-90/130), so `vla_act` is a weaker
  contact primitive for non-standard objects (e.g. it does not grasp the plate).
- Perception resolution: 256x256 instead of 1024x1024 hi-res.
- Scale: 10 tasks x 5 episodes per cell, one seed; the paper uses 10 seeds per task. Not comparable to Table 5.

## Log

- 02:40 PT: read HarnessStudio docs/code, RPent (`RPent/robots/libero/*`, prompts, guides) and the paper text
  (`Awomo-0.5/research/papers/HarnessVLA.txt`). SAM3 server (RPent `rpent.robots.components.sam3_server`, checkpoint
  `/bigdata/sejain/frozen-harness-bench/models/rpent-sam3/sam3.pt`, venv `rpent-libero-pro-20261020`) up on GPU 7, :8117;
  text-prompt segmentation on a 256 px LIBERO frame takes ~0.1 s after warm-up.
- 02:45-02:55 PT: package scaffolded and written; tier 0 PASS, tier 1 PASS (fake stack: scripted replies; perception
  returns "unavailable" there).
- 02:54 PT: first tier-2 smoke (`runs/t2_dbg1`, Goal-T tasks 0-1, episode 0; these two cases are also in the scored
  subset, disclosed here). Baseline 0/2. Null arm ran the planner end to end (5 LLM calls, 2-7 s each, SAM3
  `cabinet handle` -> plausible world xyz). Seen: the planner called `finish` claiming success while the goal had
  not fired. Generic fix before any scored run: the prompt and every turn now state that the goal has not been
  reached while the planner is still being asked (RPent shows `terminated` to its planner the same way).
  Further development uses only unscored cases (`configs/dev_libero_goal_original.yaml`, original libero_goal).
- 03:00 PT: tier 2 (`hs verify harness-vla --tier 2`, Goal-T tasks 0-1, ep 0, first code version): **PASS**, all
  episodes valid, null vs baseline 2 M / 0 R / 0 X; baseline 0/2, null 0/2, candidate 0/2.
- 03:00-03:25 PT, development on original libero_goal (unscored): perception checked on a real frame (the
  end-effector projects onto the gripper in the calibration-frame image; +x = image down, +y = image right; table
  z 0.902 from depth). Failure modes seen and fixed generically: move_to into the cabinet stalled for 80+80 steps
  (now returns `stalled` after 10 steps without 4 mm progress); the planner repeated a stalled command 20 times
  (loop guard); the open hand caught the cabinet top (hand-size sentence, travel-high rule); the planner held the
  bottle exactly above the cabinet top but misread the wrist image and moved away (controller-kept localization
  table with gripper offsets); release height rule generalized to the held object's hang.
