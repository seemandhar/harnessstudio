# Harness VLA (arXiv:2607.08448), zero-shot mode, on LIBERO-Pro Goal (2026-10-31, overnight)

Status: **complete** (05:45 PT): package, tiers 0/1/2, predeclared 100-case subset (3 arms), RPent reference run on the 20 ep-0 cases.

## Core requests

1. **Case keys collide inside LIBERO-Pro `libero_goal_task`.** Two tasks there carry the same BDDL instruction
   ("put the wine bottle in the bowl": files `put_the_cream_cheese_in_the_bowl.bddl` and
   `put_the_wine_bottle_on_top_of_the_cabinet.bddl`). `harnessstudio/server/harness_server.py:_case` keys a case by
   `(contract, suite/name, episode_idx)`, so both tasks map to the same keys: `traces/<key>.json` of the first task
   is overwritten by the second, and `harnessstudio/eval/pairing.py:summarize` raises "duplicate case keys" (so a
   one-process `hs eval` over all of Goal-T would crash at the pairing step, after writing `records.jsonl`).
   Change: include a per-task identifier in the key when the EPISODE_START task payload has one (vla-eval's sync
   runner forwards every str/int/float/bool/list field of the task dict: e.g. `task_id`/`task_idx` if present, or
   `filename_instruction`, which `LIBEROProBDDLInstruction` adds), e.g. `name = f"{suite}/{name}#{task_id}"`, and
   use the same key for the trace file name. Workaround here: records are paired by occurrence order
   (`methods/harness_vla/scripts/merge_arms.py:disambiguate`), valid because every arm runs tasks in the same
   order; the first task's 5 per-episode trace files per arm are overwritten on disk, its records are not.
2. **On LIBERO / LIBERO-Pro, `episode_idx` does not fix the whole scene** (found 2026-10-31 in the few-shot run, see
   "Few-shot mode"). The init state fixes the objects (qpos), but the jointless fixtures (stove, cabinet, rack) are
   placed at every `env.reset()` from the env's seeded RNG, so episode k's fixtures depend on how many resets the
   process did on that task's env before it (measured: identical qpos, 1-2 cm fixture shift, 18% of agentview pixels
   differ). A sequential run gives episode k the (k+1)-th placement; a sharded run or any run that starts at k > 0
   gives a different one under the same case key. `docs/writing-a-method.md` ("the scene is fixed by
   `episode_idx`") should say this, and pairing should either require arms to run the same episode sequence per
   task or the benchmark wrapper should replay the skipped resets. Package-side fix used here:
   `bench_ext` `match_sequential_resets: true` (verified: first observation hash equal to the sequential baseline
   record's on Goal-S episodes 1 and 3).

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

## Results (v2, complete: 100/100 cases valid in every arm)

Paired summary (`hs report hs2-gpu/harness_vla/runs/subset_v2/merged`; baseline records from the v1 launch, see Log):

| Cell | Cases | Baseline | Null | Candidate | Cand vs base (wins / losses) |
|---|---|---|---|---|---|
| Goal-T (`libero_goal`, `task`) | 50 | 8 | 8 | 9 | +6 / -5 |
| Goal-S (`libero_goal`, `swap`) | 50 | 15 | 15 | 21 | +16 / -10 |
| Both | 100 | 23 | 23 | **30** | +22 / -15 (sign test p = 0.32) |
| episode_idx 0 only (original declaration) | 20 | 5 | 5 | 5 | +2 / -2 |

- Candidate vs null: +22 / -15 (the null reproduces the baseline outcome on every case).
- Null attribution (null vs baseline): **90 M / 10 R / 0 X**. All 10 R are Goal-S cases where the observation hashes
  diverged before any action did (steps 40-256), i.e. the simulator/renderer branched with identical actions
  (the arms ran in different containers next to each other on GPU 7; the T1 policy cannot cause an observation
  change). Outcomes still agree on all 10.
- Not an improvement claim: n = 50 per cell, one seed, p = 0.32; no full-benchmark run follows from this.

Per task (successes out of 5 episodes; baseline = null on every task):

| Goal-S task | Base | Cand | | Goal-T task (instruction sent) | Base | Cand |
|---|---|---|---|---|---|---|
| open the middle drawer of the cabinet | 0 | 0 | | open the bottom drawer of the cabinet | 0 | 0 |
| open the top drawer and put the bowl inside | 5 | **0** | | open the top layer of the drawer and put the cream cheese inside | 0 | 1 |
| push the plate to the front of the stove | 0 | 0 | | push the cream cheese to the front of the stove | 0 | 0 |
| put the bowl on the plate | 3 | 4 | | put the cream cheese on the rack | 0 | **3** |
| put the bowl on the stove | 4 | **0** | | put the plate on the stove | 0 | 0 |
| put the bowl on top of the cabinet | 0 | **5** | | put the plate on the top of the drawer | 1 | 0 |
| put the cream cheese in the bowl | 0 | **3** | | put the wine bottle in the bowl (t2, file put_the_wine_bottle_on_top_of_the_cabinet) | 0 | 1 |
| put the wine bottle on the rack | 0 | **3** | | put the wine bottle in the bowl (t6, file put_the_cream_cheese_in_the_bowl) | 0 | 0 |
| put the wine bottle on top of the cabinet | 0 | 1 | | put the wine bottle on the plate | 3 | 1 |
| turn on the stove | 3 | 5 | | turn off the stove | 4 | 3 |

Reading: the harness does what the paper says it should on swapped scenes where the policy goes to the old
location (SAM3 + depth find the moved object/fixture; VLA grasps; scripted transport/release): bowl to cabinet top
0 -> 5, cream cheese to bowl 0 -> 3, bottle to rack 0 -> 3. It loses tasks the frozen policy already solves end to
end (open top drawer + bowl 5 -> 0, bowl to stove 4 -> 0): from the planner's staged poses the VLA grasp often
misses, and the scripted placement (a descent that stalls on the stove, releases that do not satisfy the predicate)
is worse than the policy's own pick-and-place. On Goal-T the net effect is about zero: most redirected goals need a grasp the policy cannot
do from a staged pose (the plate, pushing) or drawer approaches the planner drives into the cabinet.

Cost and behaviour (from the traces, `scripts/trace_stats.py`):

| Arm | LLM calls / ep | LLM tokens / ep | SAM3 + back-proj. / ep | wall s / ep | Policy samples (total) |
|---|---|---|---|---|---|
| baseline | 0 | 0 | 0 | 25 | 2653 |
| null | 7.3 | 42.3k | 2.9 | 61 | 2653 (native) |
| candidate | 10.2 | 64.7k | 3.5 | 70 | 1343 (method) |

Candidate commands over 100 episodes: move_to 497, segment 326, vla_act 129, back_project 25, set_gripper 23,
release 14, rotate 2. vla_act with `object_lifted` detected a lift in 41/129 calls (Goal-S 31/54, Goal-T 10/75).
58 + 60 move_to calls returned `stalled`; the identical-reply guard rejected 116 replies (91 on Goal-S). Median
steps at success ~130-140 of 300. Planner stops: 6 episodes ended by turn budget (5) or `finish` (1); the rest ran
to the horizon or to success.

## RPent reference run (adapted, unmodified authors' code)

Launcher: `hs2-gpu/harness_vla/rpent_ref/launch_rpent_ref.sh` (RPent @ 945682a run unmodified; outputs in
`rpent_ref/episodes/<suite>-t<task>-s0/`, `result.json` per case scored from `states.json`, i.e. any
`terminated`). Cases: the 20 episode-0 cases of the subset, interleaved T/S, one at a time.

| | |
|---|---|
| Planner | `--planner api --model openai-chat:qwen3-vl --base-url https://ellm.nrp-nautilus.io/v1 --reasoning-effort none`, plus a request shim (`rpent_ref/rpent_no_think.py`) adding `enable_thinking: false`, temperature 0.7, top_p 0.8, top_k 20 and seed 7, so the operator matches the port's v2 operator; `--max-turns 40` (40 model requests) |
| Policy | RPent's own VLA server with the paper's RLinf `pi05_libero130` fullshot checkpoint, GPU 7 :8118 |
| Perception | the same SAM3 server (:8117); RPent's own env server with 1024 px re-renders |
| Horizon | `--max-episode-steps 300` (official); RPent settles 15 steps before the episode, vla-eval 10 |
| Init states | RPent `--seed 0` = init state 0 = vla-eval episode_idx 0; task order and BDDL/init files are byte-identical to the HF LIBERO-Pro copy used by vla-eval (checked by sha256) |
| Zero-shot? | memory dirs empty (`--memory-profile local` with a stub empty `MEMORY.md`), no exploration, no reset. **Not fully zero-shot:** RPent has no profile without `PROVEN_LEVERS` (seed-0 lessons, mostly libero_10), KEY HYPERPARAMETERS and the 80-task perception sweep text, so those memory-derived prompt sections remain; the three guides are read as in stock RPent |

Comparison with the port is by case outcome only (different policy checkpoint and env server, so not exactly
paired).

Result (20/20 cases finished, `rpent_ref/comparison.txt`, `scripts/compare_rpent.py`):

| Cases (episode_idx 0) | RPent reference | Port: baseline (openpi pi05_libero) | Port: candidate |
|---|---|---|---|
| Goal-T, 10 | 1 (t2 "put the wine bottle in the bowl") | 2 (t7, t8) | 1 (t7) |
| Goal-S, 10 | 1 (t6 "put the cream cheese on the bowl") | 3 (t3, t7, t8) | 4 (t4, t7, t8, t9) |
| Both, 20 | **2** | 5 | 5 |

No case is solved by both the RPent reference and the port's candidate. RPent per case: ~1.42 M input tokens,
~6.4k output tokens, ~6 min; one case (Goal-T t2, a success) logged an NRP connection error late in the episode.
With the same NRP operator, the open-weight planner is the limiting factor in both implementations; n = 20 says
nothing finer than that.

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
| no loop guard | an identical motion command right after it failed/stalled is refused without env steps; a reply repeating the previous reply's commands is not executed and kept out of the context | dev case: 20 identical stalled `move_to` calls; scored v1: 29 identical `back_project` calls |
| Claude / Codex default sampling | Qwen's recommended non-thinking sampling: temperature 0.7, top_p 0.8, top_k 20, request seed 7 (v1 used temperature 0 and looped) | model card; greedy decoding repeats |
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
- The prompt rules added during development (hand size, travel high, localization table, loop guards) were
  developed on unscored original-libero_goal cases, but two Goal-T cases (t0/t1 ep 0) were seen in the first
  tier-2 smoke before the `finish` rule was added (disclosed in the Log).
- 10 of 100 null/baseline pairs are R (environment branched with identical actions); the null still matches
  the baseline outcome on all 100.
- The RPent reference run is not strictly zero-shot (PROVEN_LEVERS and other memory-derived prompt text cannot
  be switched off in RPent) and uses a different policy checkpoint and env server.

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
- 04:30 PT: all six scored processes done (v2 null/candidate ~65-70 min per arm, run in parallel). Paired
  (`runs/subset_v2/merged/summary.json`): baseline 23/100, null 23/100, candidate 30/100; null attribution
  90 M / 10 R / 0 X; all 100 cases valid in every arm. Each single-arm `hs eval` process ended with the pairing
  error from Core request 1 after writing its `records.jsonl` (Goal-T only; expected, records complete: 50 each).
- 04:35 PT: `pytest` 37 passed; tier 0 / tier 1 PASS on the final code. RPent reference run at case 11/20.
- 05:40 PT: RPent reference run finished: 2/20 (port on the same 20 cases: baseline 5, candidate 5). RPent's VLA
  server stopped by its launcher; SAM3 server (:8117, GPU 7) stopped by me afterwards (restart command above).

## Variant: gpt-6.1-sol planner (`harness-vla-sol`), added by the coordinator 2026-10-31 ~05:55 PT

Question: how much of the gap to the paper is the planner? Same controller, prompts, primitives, perception and
policy as `harness-vla` v2; only the planner operator changes to OpenAI `gpt-6.1-sol` (default temperature,
reasoning model, completion cap 6000; fidelity `adapted`). Cost probe (2 Goal-S cases, not scored): about 40k
prompt + 0.8k completion tokens and ~50 s per episode.

**Predeclared before any scored run:** candidate arm on the same 100 cases as the subset above (Goal-T and
Goal-S, 10 tasks x 5 episodes, 300-step horizon, seeded pi05_libero on :8106); null arm on the 20 episode-0
cases (leak and compute check); baseline reused from `runs/subset_v2` (the baseline arm never runs the
controller, so it is identical for both planners). Pairing via `scripts/merge_arms.py`. Budget cap: about $30 of
OpenAI usage; stop if the running total projects above it.

### Results (`runs/sol/`, finished 2026-10-31 ~06:45 PT)

All runs valid. Candidate on the predeclared 100 cases, paired with the `subset_v1` baseline records
(`runs/sol/merged_100`), and all three arms on the 20 episode-0 cases (`runs/sol/merged_ep0`).

| Arm | Goal-T | Goal-S | Total | vs baseline (wins / losses, sign test) |
|---|---|---|---|---|
| baseline (pi05_libero alone) | 8/50 | 15/50 | 23/100 | |
| harness-vla, NRP qwen3-vl planner (v2) | 9/50 | 21/50 | 30/100 | +22 / -15, p = 0.32 |
| **harness-vla-sol, gpt-6.1-sol planner** | **27/50** | **28/50** | **55/100** | **+41 / -9, p = 5.6e-6** |
| paper, zero-shot (Claude Code, RLinf pi05_libero130) | 79.0% | 31.0% | | |

Episode 0 only (20 cases, all three arms): baseline 5/20, null 5/20 (null vs baseline 17 M / 3 R / 0 X; the 3 R
are Goal-S simulator branches), candidate 12/20 (+9 / -2, p = 0.065).

LLM usage (candidate, 100 episodes): 3.33M prompt + 0.074M completion tokens (634 calls), vs 6.37M + 0.10M for
the qwen planner (the stronger planner finishes in fewer turns). Mean wall time per episode ~36 s (candidate)
vs ~25 s (baseline).

Reading: the planner is the main gap to the paper. With a frontier planner the same zero-shot controller
more than doubles the frozen policy's success on these LIBERO-Pro Goal cells, exceeds the paper's zero-shot
Goal-S number and closes much of the Goal-T gap, with a different base checkpoint (openpi pi05_libero instead of
RLinf pi05_libero130), 256 px images instead of 1024 px, and the official 300-step horizon. Not yet run: the
other LIBERO-Pro suites (spatial, object, 10), more episodes, and the few-shot (memory) mode.

### Extension to the other LIBERO-Pro cells (predeclared 2026-10-31 ~06:55 PT, before any run)

The complete Goal subset improved over baseline (above), so the gpt-6.1-sol variant is extended to the six
remaining cells of the paper's LIBERO-Pro table: libero_spatial / libero_object / libero_10 x task / swap
perturbations, 10 tasks x 2 episodes (episodes 0-1) each = 120 cases, official horizons (220 / 280 / 520),
baseline and candidate arms, same servers and settings (`configs/libero_pro_{spatial,object,10}_{task,swap}.yaml`).
Budget: about $25 more OpenAI usage. Results are reported per cell together with Goal (episodes 0-1 of the
100-case runs) for an 8-cell table.

### Results: all eight LIBERO-Pro cells (episodes 0-1; finished 2026-10-31 ~07:30 PT)

`harness-vla-sol` (gpt-6.1-sol planner) vs the frozen pi05_libero baseline, 10 tasks x 2 episodes per cell,
official horizons, all 320 episodes valid. Goal cells use episodes 0-1 of the 100-case runs above. Pairing per
cell by case (duplicate instructions in goal_task and spatial_task disambiguated by occurrence order,
`scripts/merge_arms.py`). Table: `runs/sol/eight_cells.json`.

| Cell | n | Baseline | Harness | wins / losses | sign-test p |
|---|---|---|---|---|---|
| Goal-T | 20 | 3 | 11 | +8 / -0 | 0.008 |
| Goal-S | 20 | 6 | 13 | +10 / -3 | 0.092 |
| Spatial-T | 20 | 8 | 9 | +7 / -6 | 1.0 |
| Spatial-S | 20 | 7 | 12 | +6 / -1 | 0.13 |
| Object-T | 20 | 2 | 14 | +14 / -2 | 0.004 |
| Object-S | 20 | 7 | 15 | +10 / -2 | 0.039 |
| LIBERO-10-T | 20 | 4 | 5 | +2 / -1 | 1.0 |
| LIBERO-10-S | 20 | 2 | 5 | +4 / -1 | 0.38 |
| **All** | **160** | **39 (24.4%)** | **84 (52.5%)** | **+61 / -16** | **2.4e-7** |

For reference the paper reports zero-shot only for Goal (T 79.0, S 31.0) and an overall LIBERO-PRO 82.4% for
the few-shot (memory) mode with Claude Code; RPent's leaderboard (Astra + memory) is 92.6%. This run is
zero-shot (no task memory), with a different base checkpoint and 256 px images, so it is a
lower bound for the method rather than a replication of those numbers. LIBERO-10 gains are small: long tasks
hit the 520-step horizon (RPent used a 10,000-step budget).

## Few-shot mode

The paper's second mode (arXiv:2607.08448 §2.4, RPent `--explore`): before evaluation, an agent explores ONE
reference seed of each task (with RESET) and writes task-specific memory (audit JSON + replayable recipe JSONL);
evaluation retrieves the memory for the observed task language, re-localizes everything and reuses the technique.
Paper: LIBERO-PRO few-shot 82.4% overall (Claude Code), Goal-T 87.0, Goal-S 87.0.

### Implementation (`harness-vla-sol-fewshot`, `methods/harness_vla` only)

| Part | What |
|---|---|
| Entry point / card | `harness-vla-sol-fewshot` (`hs_harness_vla:config_sol_fewshot`, card `hs_harness_vla/paper_sol_fewshot.yaml`, fidelity `adapted`): the `harness-vla-sol` controller (v2 code, gpt-6.1-sol, same budgets) with `memory: true` and `prepare=hs_harness_vla.memory:Preparer` |
| Development | `Preparer.collect_arm = "candidate"`: development episodes run this controller itself. Every turn is traced (`turn` events: thought, commands with parameters, results); the turn during which the goal fires is now traced too (trace-only change, the actions are unchanged) |
| Reference scene + RESET | `bench_ext` parameter `pin_init_index: 0` starts every development episode from init state 0 (the reference scene, = episode_idx 0) whatever its episode_idx; attempt k runs as episode_idx k-1 (`episode_offset: k-1`) under case namespace `dev-ref0`, so chained rounds never reuse a development case key. Round k is `hs prepare ... --prepared <round k-1>`: it sees the memory written by rounds 1..k-1 (RPent-like iteration) |
| Memory (`Preparer.build`) | deterministic, no LLM: one JSONL line per attempt (`memory.jsonl`, format `harness-vla-task-memory/1`) in RPent's task-memory shape: audit (localization table = first sighting of every `segment` prompt, prompts that found nothing, outcome, failure lessons = errors / stalls / grasps without lift, the planner's per-turn reasoning) + recipe (the executed primitive sequence with its parameters and results; every `move_to` / `move_pose` target also given as an offset from the landmark it was aimed at). Chained builds append to the previous artifact. Only RPent's task-specific layer; no task-family / global layers, and RPent's shipped corpus (PROVEN_LEVERS, MEMORY.md) is not used |
| Evaluation | `ctx.artifacts["memory"]` (sha256-pinned, read-only); all attempts whose task language equals `ctx.instruction` are rendered into the first planner message (latest successful attempt in full, the others as outcome + what failed); the system prompt gets one paragraph on how to use it (RPent PROVEN_LEVERS framing: reuse technique, re-localize, never copy coordinates). Two Goal-T tasks share the instruction "put the wine bottle in the bowl": both tasks' attempts are shown for it |
| Leakage | evaluation runs `episode_offset: 1, episodes_per_task: 4`, i.e. episode_idx 1-4 only: **the reference scene (episode 0) is never run in evaluation, by construction**. (The development keys are namespaced, so the framework's automatic refusal would not catch an ep-0 evaluation case; it cannot arise here.) Checked after the run: no evaluation record has episode_idx 0 (holds), and every evaluation case's first observation hash equals that of the zero-shot record of the same case and differs from the reference scene's (failed for the predeclared run, see Results: fixture placement; holds for the exact re-run) |

Development of this mode (before the declaration below) used only unscored original `libero_goal` task 0
(smoke: 2 development attempts on its init state 0 under namespace `dev-smoke`, 1 evaluation episode at
episode_idx 1; all 3 succeeded) to check the pipeline: pinned attempts reproduce the reference scene (identical
first-observation hash), chained rounds see the earlier memory, the offset evaluation keys the case as episode 1.

### Predeclared plan (written 2026-10-31 07:24 PT, before any development or scored episode on Goal-T / Goal-S)

| | |
|---|---|
| Development budget | up to **3 exploration attempts per task** on the reference scene (init state 0) of all 20 tasks (Goal-T 10 + Goal-S 10): round 1 = all 20 tasks (memory empty, i.e. the zero-shot controller); round 2 = tasks whose instruction has no successful attempt yet; round 3 = the same rule. Chained `hs prepare`, namespace `dev-ref0`, attempt k = episode_idx k-1. The final round's `prepared.json` is frozen for evaluation |
| Evaluation | Goal-T and Goal-S, 10 tasks x **episodes 1-4** = **80 cases**, official horizon 300, seeded pi05_libero (:8106), SAM3 (:8117), `docker.gpus: "7"`; candidate arm only (run as 4 processes: per cell episodes 1-2 and 3-4) |
| Candidate | `harness-vla-sol-fewshot --prepared <final round>` |
| Comparisons (pair by pair, same 80 cases) | (a) frozen baseline (`runs/subset_v1/goal_{task,swap}_baseline`, episodes 1-4) and (b) zero-shot `harness-vla-sol` (`runs/sol/cand_goal_{task,swap}`, episodes 1-4); both already recorded. Duplicate-instruction pairing by occurrence order as before (`scripts/merge_arms.py:disambiguate`) |
| Reported | successes per cell and total, wins / losses and two-sided sign-test p per comparison, development successes per round, LLM tokens (prompt / completion) for development and evaluation separately, wall time, OpenAI cost estimate |
| Budget / stop rules | **$40** OpenAI total for this mode (estimate $5/M prompt, $25/M completion); stop if the running total projects above it. Hard time box: results written by 13:00 PT; an incomplete run is reported as incomplete with its n, never as a percentage of the full plan |
| Paper reference | few-shot Goal-T 87.0 / Goal-S 87.0 (Claude Code, RLinf pi05_libero130, 10 seeds/task, 10000-step budget, 1024 px renders): not comparable, context only |

### Results (development 07:24-07:37 PT, scored runs 07:37-08:15 PT; every episode valid)

Run dir `R=hs2-gpu/harness_vla/runs/fewshot` (drivers `hs_run.sh`, `dev_round.sh`, `eval_launch.sh`, `eval_exact*.sh`;
summaries `summary*.json` from `methods/harness_vla/scripts/fewshot_compare.py`; frozen code `code_snapshot/` for
development and the predeclared run, `code_snapshot_v2/` for the exact re-run, which differs only in `bench_ext`).

**Development** (32 episodes on the reference scenes; first-observation hash identical to the zero-shot episode-0
records, so every attempt saw exactly the episode-0 scene):

| Cell | attempt 1 (memory empty) | attempt 2 | attempt 3 | tasks with a successful attempt |
|---|---|---|---|---|
| Goal-T | 6/10 | 1/4 | 1/3 | 8/10 |
| Goal-S | 7/10 | 1/3 | 0/2 | 8/10 |

Never solved: Goal-T "open the bottom drawer of the cabinet", "put the plate on the top of the drawer"; Goal-S "open
the middle drawer of the cabinet", "push the plate to the front of the stove". Frozen memory: `dev/goal_task_r3`
(17 attempts, sha256 `3ed29dd2...`), `dev/goal_swap_r3` (15 attempts, `a835e369...`).

**Predeclared evaluation** (80 cases, episodes 1-4, `eval/`; no record has episode_idx 0):

| Cell | n | Baseline | Zero-shot sol | **Few-shot** | few vs base (sign p) | few vs zero-shot (sign p) |
|---|---|---|---|---|---|---|
| Goal-T | 40 | 6 | 21 | **26** | +21 / -1 (1.1e-5) | +8 / -3 (0.23) |
| Goal-S | 40 | 12 | 22 | **27** | +17 / -2 (7.3e-4) | +9 / -4 (0.27) |
| Both | 80 | 18 | 43 | **53** | +38 / -3 (1e-8) | +17 / -7 (0.064) |

**Protocol flaw found in the post-run scene check, and what it changes.** The evaluation ran episodes 1-2 and 3-4
in fresh processes (`episode_offset`), the baseline and zero-shot records ran episodes 0-4 sequentially. The objects
are identical (simulator qpos equal for the same init state, checked on 3 tasks x 2 cells), but LIBERO places the
jointless fixtures (stove, cabinet, rack) at every `env.reset()` from the env's seeded RNG (Core request 2): here
episodes 1 and 3 got the first reset's placement (= the reference scene's fixtures) and episodes 2 and 4 the
second's, against resets 2-5 in the sequential runs (mean depth change 1-2 cm over the fixture region, 18% of
agentview pixels differ; the first-observation hashes matched 0/80). So the table above does not pair identical scenes. The same protocol with
the frozen policy alone (`control/`, no LLM) gives 23/80 instead of 18/80 (+7 / -2): placement alone moves outcomes.

**Exact-protocol re-run** (`eval_exact/`, `bench_ext` `match_sequential_resets: true`: the skipped resets are replayed,
so episode k gets reset k+1 as in a sequential run; first-observation hash equal to the baseline and zero-shot
records on 60/60 cases). The remaining budget covered 60 of the 80 cases: Goal-T episodes 1 and 3, Goal-S episodes 1-4.

| Cases (exactly paired) | n | Baseline | Zero-shot sol | **Few-shot** | few vs base (sign p) | few vs zero-shot (sign p) |
|---|---|---|---|---|---|---|
| Goal-T, episodes 1, 3 | 20 | 3 | 11 | **13** | +10 / -0 (0.002) | +4 / -2 (0.69) |
| Goal-S, episodes 1-4 | 40 | 12 | 22 | **25** | +17 / -4 (0.007) | +7 / -4 (0.55) |
| All exact | 60 | 15 | 33 | **38** | +27 / -4 (3.4e-5) | +11 / -6 (0.33) |

Cross-check: on episodes 1 and 3, whose fixtures matched the reference scene in the predeclared run, few-shot
scored 26/40 in both protocols (+2 / -2), so the shared placement did not inflate them. On Goal-S episodes 2 and 4
the predeclared run's 15/20 drops to 12/20 with exact scenes (zero-shot there: 8/20).

**Cost** (OpenAI gpt-6.1-sol, estimate $5/M prompt + $25/M completion; cached tokens negligible):

| Stage | Episodes | LLM calls | Prompt tokens | Completion tokens | Est. USD | Episode time |
|---|---|---|---|---|---|---|
| Development (+ smoke: 3 episodes on original libero_goal) | 32 (+3) | 209 (+10) | 1.262M (+0.043M) | 24.9k (+0.7k) | 6.93 (+0.23) | 1244 s (+57 s) |
| Evaluation, predeclared | 80 | 496 | 3.318M | 53.4k | 17.92 | 2620 s (33 s/ep) |
| Evaluation, exact re-run | 60 | 386 | 2.631M | 42.3k | 14.21 | 2219 s (37 s/ep) |
| **Total, this mode** | | | | | **39.30** (cap 40) | |
| for reference: zero-shot sol on the same 80 cases | 80 | 515 | 2.718M | 60.6k | 15.10 | 2911 s (36 s/ep) |

The memory adds about 7k prompt tokens per episode (41.5k vs 34.0k; it stays in the first message on every turn).
Elapsed: development 13 min (2 processes), predeclared evaluation 13 min (4 processes), exact re-run 14 min.

**Reading.**
- Few-shot vs the frozen policy: large and consistent (exactly paired 38 vs 15 of 60, +27 / -4).
- Few-shot vs zero-shot: no detectable gain. Exactly paired: 38 vs 33 of 60, +11 / -6, p = 0.33. The predeclared
  run's +17 / -7 (p = 0.064) compares different fixture placements and is not a valid pairing; it is not promoted.
- Where the memory changed outcomes (exact cases): it helped multi-stage drawer tasks that development had solved
  ("open the top drawer and put the bowl inside" 0 -> 4/4, Goal-T "open the top layer of the drawer and put the
  cream cheese inside" 0 -> 2/2, "turn on the stove" 2 -> 4/4) and hurt pick-and-place tasks that zero-shot already
  solved ("put the cream cheese in the bowl" 3 -> 1/4, "put the bowl on the stove" 2 -> 1/4, Goal-T wine bottle in
  bowl 2 -> 1/2 for each of the two tasks). Tasks never solved in development: 1 success in 12 exact cases (Goal-T
  "put the plate on the top of the drawer" 1/2); "put the plate on the stove", solved once in development, was 0/2
  (exact) and 0/4 (predeclared) in evaluation.
- Paper (Claude Code, RLinf pi05_libero130, 1024 px, 10000 steps, agent-written memory): Goal-T 87.0, Goal-S 87.0.
  Here Goal-S 25/40 (exact) and Goal-T 26/40 (predeclared protocol only; 13/20 exact). Not comparable; the port's
  memory is the task-specific layer only, distilled without an LLM, from 1-3 attempts per task.

**Known gaps (few-shot).** One seed, small n; the planner is not repeatable (development attempt 1 on episode 0 =
the zero-shot controller: 13/20, vs 12/20 in the zero-shot run on the same scenes). Exact pairing covers 60 of the 80
predeclared cases (Goal-T episodes 2 and 4 exist only in the shifted-fixture protocol). The two Goal-T tasks that
share an instruction get both tasks' memory. No task-family / global memory layer.

### Log (few-shot)

- 07:14-07:21 PT: read docs, package, RPent memory format (task-specific `<cell>_t<i>_s0.json` + `_recipe.jsonl`,
  PROVEN_LEVERS framing). Wrote `memory.py` (Preparer, render), controller `memory` option, card, entry point,
  `bench_ext` `pin_init_index` / `episode_offset`; tier 0 / tier 1 PASS.
- 07:21-07:23 PT: smoke on unscored original libero_goal task 0 (2 pinned attempts + 1 offset evaluation): pinned
  attempts reproduce the reference scene (hash), chained memory is injected, the offset case is keyed as episode 1.
- 07:24 PT: plan predeclared (above); code frozen (`code_snapshot/`); development round 1 launched.
- 07:31 / 07:35 / 07:37 PT: rounds 1 / 2 / 3 done (13/20, 2/7, 1/5). 07:37 PT: predeclared evaluation launched.
- 07:50 PT: evaluation done, 53/80. Scene check failed (first-observation hashes 0/80 equal to the zero-shot
  records'). 07:53-07:58 PT: diagnosed in the container (`scene_check/`): equal qpos, fixtures placed per reset by
  the env RNG. Launched the same-protocol baseline control (no LLM).
- 07:58 PT: added `match_sequential_resets` to `bench_ext` (off by default), verified on 2 Goal-S cases with the
  baseline arm (first-observation hash equal to the original baseline records'), froze `code_snapshot_v2/`, re-ran
  episodes 1 and 3 of both cells (08:06 PT, 26/40), then Goal-S episodes 2 and 4 under a budget guard (08:15 PT).
  Stopped spending at an estimated $39.30. `pytest` 37 passed; `hs verify harness-vla-sol-fewshot` tier 0 / 1 PASS.

## Audit (2026-10-31, ~10:50 PT)

A read-only compliance audit of the package found no critical violations: no simulator state or evaluator
labels in method code, sensor-only benchmark additions (RGB-D + calibration), nothing trained, memory built only
from development records and pinned by sha256, null arms reproducing the baseline (90 M / 10 R / 0 X); the
8-cell table and the few-shot rows recompute from `records.jsonl`. Disclosures it required, now listed in the
three paper cards: the planner is told the goal is not yet reached on every turn, which uses LIBERO's
termination-on-success as an implicit verifier (the baseline and a real robot have no such check, so the
harness's advantage partly reflects this); controller-side aids not in the paper (stall early-return,
identical-reply guard, failed-command refusal, a localization table); SAM3 calls bypass `ctx.tools`; gpt-6.1-sol
is unseeded; few-shot development uses a namespace, so the framework's dev-case refusal cannot fire and
held-out evaluation relies on `episode_offset` (the exactly-paired 60 cases are the valid comparison).
