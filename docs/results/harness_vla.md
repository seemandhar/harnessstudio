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
| Planner | NRP `qwen3-vl` (served `Qwen/Qwen3.8-Flash-Next-FP8`), `enable_thinking: false`, v2 sampling temperature 0.7 / top_p 0.8 / top_k 20 / seed 7 (v1: temperature 0), max 800 output tokens per call; picked after an image probe on a LIBERO frame (kimi: reasoning ate the budget, 80 s, empty content; gemma-vl: 150 s timeout; gemma4-12b: empty content; glm-5: HTTP 400 on images) |
| Budgets / episode | policy 400 samples, LLM 45 calls, 800 tokens/call, SAM3 + back-projection 60 calls |
| Frozen code for the scored run | v2: method fingerprint `90318ab47e08ad601c2569df3f209016d6c9523cb7ba10362c805fb365850515`, file hashes and a copy of `hs_harness_vla/`, `bench_ext/`, `configs/` in `hs2-gpu/harness_vla/runs/subset_v2/code_{sha256.txt,snapshot/}` (v1 `25a3b6ee...` in `runs/subset_v1/`, stopped) |

Commands (all from `/home/sejain/research/cvpr27/harnessstudio-v2`; run dir `R=/home/sejain/research/cvpr27/hs2-gpu/harness_vla`):

```bash
# SAM3 server (RPent's), GPU 7
cd /home/sejain/research/cvpr27/RPent && CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 \
  SAM3_CHECKPOINT_PATH=/bigdata/sejain/frozen-harness-bench/models/rpent-sam3/sam3.pt \
  /bigdata/sejain/frozen-harness-bench/venvs/rpent-libero-pro-20261020/bin/python \
  -m rpent.robots.components.sam3_server --transport http --host 127.0.0.1 --port 8117
# tiers
.venv/bin/hs verify harness-vla --tier 0 && .venv/bin/hs verify harness-vla
.venv/bin/hs verify harness-vla --tier 2 --config methods/harness_vla/configs/libero_pro_goal_task.yaml \
  --policy-url ws://127.0.0.1:8106 --contract methods/harness_vla/hs_harness_vla/contracts/libero_pro_rgbd.yaml \
  --profile pi05-libero-hs --out $R/runs/t2_final --benchmark-overrides '{"max_tasks": 2, "episodes_per_task": 1}'
# scored subset: one process per (cell, arm), e.g. Goal-S candidate
.venv/bin/hs eval harness-vla --arms candidate --config methods/harness_vla/configs/libero_pro_goal_swap.yaml \
  --policy-url ws://127.0.0.1:8106 --contract methods/harness_vla/hs_harness_vla/contracts/libero_pro_rgbd.yaml \
  --profile pi05-libero-hs --out $R/runs/subset_v2/goal_swap_candidate --port 8253 \
  --benchmark-overrides '{"episodes_per_task": 5}'
# pairing (harnessstudio.eval.pairing.summarize over the six record files) and report
.venv/bin/python methods/harness_vla/scripts/merge_arms.py $R/runs/subset_v2/merged \
  baseline=$R/runs/subset_v1/goal_task_baseline null=$R/runs/subset_v2/goal_task_null candidate=$R/runs/subset_v2/goal_task_candidate \
  baseline=$R/runs/subset_v1/goal_swap_baseline null=$R/runs/subset_v2/goal_swap_null candidate=$R/runs/subset_v2/goal_swap_candidate
.venv/bin/hs report $R/runs/subset_v2/merged
```

Arms run as separate processes (all six at once) so the night fits: the policy server is seeded (T1), so a case's
native stream does not depend on which process or order it runs in, and pairing is by case key exactly as in
`hs eval`.

## Results

(pending)

## RPent reference run (adapted, unmodified authors' code)

Launcher: `hs2-gpu/harness_vla/rpent_ref/launch_rpent_ref.sh` (RPent @ 945682a run unmodified; outputs in
`rpent_ref/episodes/<suite>-t<task>-s0/`, `result.json` per case scored from `states.json`, i.e. any
`terminated`). Cases: the 20 episode-0 cases of the subset, interleaved T/S, one at a time.

| | |
|---|---|
| Planner | `--planner api --model openai-chat:qwen3-vl --base-url https://ellm.nrp-nautilus.io/v1 --reasoning-effort none`, plus a request shim (`rpent_ref/rpent_no_think.py`) adding `enable_thinking: false` and temperature 0, so the operator matches the port; `--max-turns 40` |
| Policy | RPent's own VLA server with the paper's RLinf `pi05_libero130` fullshot checkpoint, GPU 7 :8118 |
| Perception | the same SAM3 server (:8117); RPent's own env server with 1024 px re-renders |
| Horizon | `--max-episode-steps 300` (official); RPent settles 15 steps before the episode, vla-eval 10 |
| Init states | RPent `--seed 0` = init state 0 = vla-eval episode_idx 0; task order and BDDL/init files are byte-identical to the HF LIBERO-Pro copy used by vla-eval (checked by sha256) |
| Zero-shot? | memory dirs empty (`--memory-profile local` with a stub empty `MEMORY.md`), no exploration, no reset. **Not fully zero-shot:** RPent has no profile without `PROVEN_LEVERS` (seed-0 lessons, mostly libero_10), KEY HYPERPARAMETERS and the 80-task perception sweep text, so those memory-derived prompt sections remain; the three guides are read as in stock RPent |

Comparison with the port is by case outcome only (different policy checkpoint and env server, so not exactly
paired).

Behaviour at the official horizon: when the 300 steps are used up, robosuite raises "executing action in terminated
episode" inside RPent's env server, the RPC pipe breaks and the RPent planner reports an "infra crash". This is
the horizon, not an infrastructure fault; the case is scored from `states.json` (no `terminated` before the
horizon = failure), which is correct. RPent was built for a 10000-step budget and never hits this normally.
Per case it used ~1.2-1.5 M input tokens (the RPent system prompt is ~33k characters plus guides and three
images per state) and ~5-6 min.

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
- 03:13 PT, last development check before freezing (original libero_goal, 10 tasks x ep 0, unscored, final code):
  candidate 5/10, baseline 10/10 (pi0.5 is trained on exactly these tasks; the planner's staging can only get in
  its way there). Failures: drawers (planner drives into the cabinet), plate push, bowl->plate, bottle->rack.
- 03:14 PT: code frozen (hashes above); scored subset launched: 6 processes (Goal-T/Goal-S x baseline/null/candidate).
- 03:15 PT: RPent adapted reference run launched (see "RPent reference run" below).
- 03:23 PT: **scored v1 null/candidate arms stopped** after 2-8 episodes each (records kept in `runs/subset_v1/`):
  with greedy decoding (temperature 0) the planner fell into deterministic loops, e.g. the same `back_project`
  29 times in a row (Goal-S t0 ep0) or the same refused `move_to` 17 times (Goal-T t0 ep1), spending its whole
  turn budget at step 0/90. Qwen's model card warns against greedy decoding for exactly this. v2 (method version 2,
  fingerprint `90318ab4...`): Qwen's recommended non-thinking sampling (temperature 0.7, top_p 0.8, top_k 20, request
  seed 7) and an identical-reply guard (a reply repeating the previous commands is not executed and is kept out of
  the context, with a note). Checked on 2 unscored dev cases, then relaunched at 03:27 PT into `runs/subset_v2/`.
  The baseline arms were not stopped: they never run the controller, so their records are the same for v1 and v2.
  The RPent reference run was restarted with the same sampling (its first case had not finished).
- 03:33 PT: tier 2 re-run with the frozen v2 code (`runs/t2_v2`, Goal-T tasks 0-1, ep 0): **PASS**, all episodes
  valid, null vs baseline 2 M / 0 R / 0 X (baseline 0/2, null 0/2, candidate 0/2). (v1 code: `runs/t2_final`, also
  PASS, 2 M.) Baseline arms finished: Goal-T 8/50, Goal-S 15/50.
