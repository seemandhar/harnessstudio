# RoboHarness (arXiv:2603.24060v3) in HarnessStudio v2

Status: LIBERO-Pro subset running (2026-10-31). Package: `methods/roboharness` (entry point `roboharness`).

## What was ported

The authors' evaluation code (github.com/LZY-1021/RoboHarness, MIT, commit 716defc), not a re-derivation from the
paper: the `soma_eval.py` rollout step for step (perception every 10 steps, LeRobot-style 10-action queue,
Encore rollback 40 + 20, key-step timeout rollback 50 + 25, subtask queue), `PerceptionModule.process_frame`
(VLM plan -> SAM3 tools -> VLM verification -> re-plan, 40-step visual quota), the VLM prompts, the SAM3 tool
server's pixel operations, the CLIP + MiniLM + hash encoder and the dual memory bank. The full list of
substitutions is in `methods/roboharness/hs_roboharness/paper.yaml`.

## Findings about the released artifact

1. **The released evaluation never intervenes.** `soma_perception.py` sets `MAX_RETRIES = 0`, so the plan/verify
   loop in `process_frame` never runs: every call returns the frame and instruction unchanged. Only the memory
   logging after each episode is active. The port uses 3 attempts (the documented retry behaviour).
2. **Task chaining was not VLM-driven.** `chain_step_eval.py` disables the agent (`soma_agent = None`) and runs a
   hard-coded three-subtask chain with fixed step budgets (280/280/300) and a full reverse-action rollback
   between subtasks.
3. **Paper components missing from the code**: blank-inference attention validation (attention maps only appear in
   visualization scripts), Stage-2 offline comparative memory consolidation, and the texture assets that
   `replace_texture` / `replace_background` need (the logger never stores them, so those tools never run).
4. **The paper's latency table implies one orchestration per episode** (VLM 7.2 ms/step amortized over 1000
   steps), while the code orchestrates every 10 steps. The port follows the code.
5. **LIBERO-RoboHarness cannot be rebuilt from the release as published**: 3 of 5 init-state files do not match
   their BDDL files (state length 188 vs 123, 175 vs 110, 110 vs 97; LIBERO silently mis-places objects), the
   `red_bowl` object class is not registered in stock LIBERO and its stock texture is grey (the paper's bowls are
   red), and the Clutter Removal BDDL has its distractor placements commented out (one bowl on the table).
   `bench_ext/hs_roboharness_bench.py` registers the suite; it is not used for results until these are resolved.
6. **Protocol differences of the authors' LeRobot evaluation**: it sends `Task.language` (derived from the BDDL file
   name), so LIBERO-Pro `task`/`lan` episodes receive the unperturbed instruction (consistent with the paper's
   0.86% pi0.5 Task baseline); libero_spatial runs 280 steps (official: 220). Here both arms get the BDDL
   instruction and the official horizon.

## Validation

| Check | Result |
|---|---|
| Parity vs the authors' modules (`scripts/check_parity.py`, 300 randomized episodes) | 0 differences: 24,916 control-flow steps, 1,798 `process_frame` calls (frames, refined task, flags, counters, every prompt message), 600 memory retrievals, 3,000 parser cases |
| Live tool chain on a LIBERO frame | plan JSON valid; detector box on target; SAM3 overlay (score 0.55) and inpainting (0.87) correct; 1168-d embedding; verifier passes |
| Controller with interventions off vs native pi0.5 (real LIBERO-Spatial, 50 cases) | 50/50 identical actions and outcomes (4,979 steps) |
| Tier 1 (fake stack, incl. prepare stage and dev-case refusal) | PASS |

## Development memory

`hs prepare` on standard LIBERO-Spatial + LIBERO-Object, 10 tasks x 10 episodes each (the released code logs
10 episodes per task), perception off as released, 4 shards merged by `scripts/merge_dev.py`:
200 episodes, 199 successes -> memory of 199 success and 1 failure entries (VLM: 200 calls, 81k prompt / 19k
completion tokens). Because the prompt only carries failure diagnoses, the frozen memory contributes one
diagnosis ("Control Error: ... gripper opens prematurely ...") when it is within cosine 0.4; the orchestrator is
otherwise zero-shot. The authors' memory additionally grows online from evaluated LIBERO-Pro episodes (mostly
failures, labelled by the environment), which HarnessStudio does not allow.

## Results

Predeclared subset: LIBERO-Pro {spatial, object} x {task, swap}, all 10 tasks x episodes 0-1 (80 paired cases),
official horizons, pi0.5 (openpi pi05_libero, seeded), orchestrator NRP qwen3-vl. Table III rows are reported as
a sub-subset.

(pending)
