# harness-vla: Harness VLA (arXiv:2607.08448), zero-shot mode

Port of the **zero-shot** Harness VLA agent (paper §2-3, Table 5; authors' code RPent, Apache-2.0,
`/home/sejain/research/cvpr27/RPent` @ 945682a) as a HarnessStudio controller. Fidelity **adapted**: the paper's
planner (Claude Code / Codex) is replaced by a pinned NRP open-weight VLM, and the policy is openpi `pi05_libero`
instead of RLinf `pi05_libero130`. Full list: `hs_harness_vla/paper.yaml` (substitutions).
Results: `docs/results/harness_vla.md`.

## What runs

A turn-based planner REPL (paper §2.1): each turn the planner gets the task language, the coordinate-free robot
state (eef xyz, yaw/pitch, gripper opening), the env steps left, the results of its last commands and the current
agentview + wrist images (plus SAM3 overlays), and replies with JSON commands:

| Command | Kind | Ported from (RPent `robots/libero/tools.py`) |
|---|---|---|
| `segment` | perception: SAM3 mask on the current image -> median world xyz via metric depth + calibration | `segment` + `_mask_to_world` |
| `back_project` | perception: pixel or region -> world xyz | `back_project` |
| `move_to` | analytic, composite (reuses `harnessstudio.skills.eef_delta.move_to`; optional yaw) | `move_to` |
| `move_pose` | analytic, composite (xyz + pitch + yaw) | `move_pose` |
| `rotate_wrist`, `rotate_pitch` | analytic, atomic (position hold) | same names |
| `set_gripper`, `release` | analytic, atomic (`eef_delta.set_gripper` / `release`) | same names |
| `vla_act` | the frozen VLA with the planner's prompt, chunk by chunk, early return on `object_lifted` (pi0_pick's predicate) or `none` (pi0_doubled) | `pi0_pick`, `pi0_doubled` |
| `finish` | stop; the arm holds until the horizon | `finish` |

Zero-shot means: no Task Specific Memory, no Global Memory, no reference-seed exploration, no RESET.
Only public sensor data is used: RGB, metric depth, camera calibration, proprioception. No object poses,
no object-name list, no BDDL.

## Files

- `hs_harness_vla/controller.py`: the REPL (`HarnessVLA`), reply parsing, history, holding after finish
- `hs_harness_vla/primitives.py`: the primitive library on `ctx`
- `hs_harness_vla/perception.py`: calibration-frame views, back-projection, SAM3 client (RPent's HTTP RPC)
- `hs_harness_vla/prompts.py`: the planner prompt (RPent evaluate prompt + guides, memory removed)
- `hs_harness_vla/contracts/libero_pro_rgbd.yaml`: contract with `depth` and `camera` public
- `bench_ext/hs_harness_vla_bench.py`: container-side benchmark = HarnessStudio RGB-D mixin + LIBERO-Pro BDDL
  instruction fix (Python 3.8, mounted with the repo)
- `configs/libero_pro_goal_{task,swap}.yaml`: vla-eval configs (GPU 7 rendering, official horizon, timeout 900 s)

## Run

```bash
# 1. SAM3 server (RPent's), GPU 7, port 8117 (override with HS_HARNESS_VLA_SAM3_URL)
cd /home/sejain/research/cvpr27/RPent && CUDA_VISIBLE_DEVICES=7 \
  SAM3_CHECKPOINT_PATH=/bigdata/sejain/frozen-harness-bench/models/rpent-sam3/sam3.pt HF_HUB_OFFLINE=1 \
  /bigdata/sejain/frozen-harness-bench/venvs/rpent-libero-pro-20261020/bin/python \
  -m rpent.robots.components.sam3_server --transport http --host 127.0.0.1 --port 8117
# 2. a seeded pi0.5 policy server (profile pi05-libero-hs) on ws://127.0.0.1:8106
# 3. checks and evaluation (NRP_API_KEY in the environment)
hs verify harness-vla --tier 0 && hs verify harness-vla
hs eval harness-vla --config methods/harness_vla/configs/libero_pro_goal_task.yaml \
  --contract methods/harness_vla/hs_harness_vla/contracts/libero_pro_rgbd.yaml \
  --policy-url ws://127.0.0.1:8106 --profile pi05-libero-hs --out runs/harness-vla-goal-task
```

Tier 1 (fake stack, no network) runs the scripted replies in `TIER1_SCRIPT`; there is no depth or calibration
there, so `segment` / `back_project` return `{"error": ...}` and the planner continues.

## Scripts

- `scripts/merge_arms.py OUT baseline=DIR null=DIR candidate=DIR ...`: pair arms that ran as separate
  `hs eval --arms` processes (handles the duplicate Goal-T instruction, see the results doc's Core requests);
  `hs report OUT` then prints the paired summary.
- `scripts/trace_stats.py DIR ...`: primitive usage, stalls, rejected repeats, LLM cost per arm.
- `scripts/compare_rpent.py RPENT_EPISODES baseline=DIR candidate=DIR ...`: case-by-case table against the RPent
  reference run (`hs2-gpu/harness_vla/rpent_ref/launch_rpent_ref.sh`).
- `scripts/check_geometry.py OBS.npz`: projection / back-projection sanity check on an observation dumped with
  `HS_HARNESS_VLA_DEBUG_DIR=<dir>` (development aid; off by default).

Version 2 (the scored one): Qwen's recommended non-thinking sampling (temperature 0.7, top_p 0.8, top_k 20,
seed 7) and an identical-reply guard; v1 used greedy decoding and looped (see the results doc).

## Few-shot (memory) mode: `harness-vla-sol-fewshot`

Same controller and planner as `harness-vla-sol` with `memory: true` and a development stage
(`hs_harness_vla/memory.py:Preparer`, card `hs_harness_vla/paper_sol_fewshot.yaml`). Development explores the
reference scene (init state 0) of each task: `bench_ext` parameter `pin_init_index: 0` replays that scene for every
attempt (RPent's RESET), attempt k runs as episode_idx k-1 (`episode_offset`) under `--case-namespace dev-ref0`,
and each round is a chained `hs prepare --prepared <previous round>`. `Preparer.build` turns the traced turns into
`memory.jsonl` (per attempt: localization table, outcome, failure lessons, reasoning, and the executed primitive
recipe with landmark-relative targets). Evaluation must exclude the reference scene: run it with
`"params": {"episode_offset": 1, "match_sequential_resets": true}` so only episodes >= 1 exist and each episode gets
the same fixture placement as in a sequential run (LIBERO places fixtures per `env.reset()` from the env RNG; without
the flag, episode k of an offset run is not the scene of episode k in other arms' sequential records).

```bash
OV='{"episodes_per_task": 1, "params": {"pin_init_index": 0, "episode_offset": 0}}'   # attempt 1
hs prepare harness-vla-sol-fewshot --config configs/libero_pro_goal_swap.yaml --case-namespace dev-ref0 \
  --benchmark-overrides "$OV" --out prep_r1 --policy-url ws://127.0.0.1:8106 \
  --contract hs_harness_vla/contracts/libero_pro_rgbd.yaml --profile pi05-libero-hs --action-timeout 1800
# attempt 2 (unsolved tasks only, via "tasks": [...]): episode_offset 1, --prepared prep_r1 --out prep_r2
hs eval harness-vla-sol-fewshot --arms candidate --prepared prep_r2 --config configs/libero_pro_goal_swap.yaml \
  --benchmark-overrides '{"episodes_per_task": 4, "params": {"episode_offset": 1, "match_sequential_resets": true}}' ...
python scripts/fewshot_compare.py --fewshot EVAL_DIRS --baseline BASE_DIRS --zeroshot ZERO_DIRS --dev PREP_DIRS
```
