# Ported v1 methods on real LIBERO (first LIBERO runs; v1 only ran RoboCasa365)

**Predeclared 2026-10-31 ~08:30 PT, before any run.**

| | |
|---|---|
| Benchmark | vla-eval 0.8.0 LIBERO, `libero_10` (Long), seed 7, official horizon 520, Docker (rendering on GPU 7) |
| Policy | seeded pi0.5 (`harnessstudio/policies/pi05_openpi.py`, pi05_libero), :8106 on GPU 6, profile `pi05-libero-hs` |
| Cases | `medoid-bon`, `chunk-interrupt`: 10 tasks x episodes 0-2 (30 cases); `aac`: 10 tasks x episode 0 (10 cases, K=20 is 2.5x the cost of K=8) |
| Arms | baseline / null / candidate via `hs eval` (one process per arm, same case list and order) |
| Report | successes per arm, paired wins/losses and sign-test p, null attribution M/R/X, policy samples, wall time |

These are screens of training-free methods ported from v1 on a new benchmark/policy pair, not paper
reproductions (medoid-bon and chunk-interrupt are components of larger papers; AAC's paper used GR00T on RoboCasa).

## Results

### AAC, LIBERO-10 episode 0 (10 cases)

All three arms 10/10: pi0.5 is at ceiling on standard LIBERO-10 episode 0, so an adaptive-horizon executor has
no headroom here. Null vs baseline 9 M / 1 R / 0 X (the R: observations diverged at step 36 before any action
did). Policy samples: baseline 246, null 4,940, candidate 7,060 (K = 20); wall 232 s / 1,262 s / 1,996 s.

### medoid-bon and chunk-interrupt, LIBERO-10 episodes 0-2 (30 cases each)

| Method | Baseline | Null | Candidate | Null vs baseline | Policy samples (b / n / c) |
|---|---|---|---|---|---|
| medoid-bon (K=8) | 29/30 | 29/30 | 29/30 | 29 M / 1 R / 0 X | 768 / 6,160 / 6,112 |
| chunk-interrupt | 29/30 | 29/30 | 29/30 | 30 M / 0 R / 0 X | 768 / 768 / 778 |

Same picture as AAC: pi0.5 is at ceiling on standard LIBERO-10 (97%), so these screens only confirm that the
methods run cleanly on a new benchmark/policy pair (all episodes valid, 0 X). The single failing case is the
same for every arm.

## Second screen: LIBERO-Pro (predeclared 2026-10-31 ~09:40 PT, before any run)

Because pi0.5 is near ceiling on standard LIBERO, the same methods are screened where it is not: LIBERO-Pro
`libero_object` and `libero_goal` with the `task` perturbation (instruction fix:
`harnessstudio.benchmarks.libero_pro_instructions`), 10 tasks x episodes 0-1 = 20 cases per cell,
`medoid-bon` and `chunk-interrupt`, three arms each, same server, config `hs2-gpu/configs/libero_pro_hs.yaml`.

### LIBERO-Pro results (finished ~11:40 PT)

All 240 episodes valid. Goal-task runs started before the case-key fix was merged, so they were paired with
`methods/harness_vla/scripts/merge_arms.py` (occurrence order; two goal_task tasks share an instruction).

| Method | Cell (20 cases) | Baseline | Null | Candidate | vs baseline | Null vs baseline |
|---|---|---|---|---|---|---|
| medoid-bon (K=8) | libero_object task | 2 | 2 | 2 | +0 / -0 | 20 M / 0 R / 0 X |
| medoid-bon (K=8) | libero_goal task | 3 | 3 | 4 | +2 / -1 | 19 M / 1 R / 0 X |
| chunk-interrupt | libero_object task | 2 | 2 | 2 | +0 / -0 | 20 M / 0 R / 0 X |
| chunk-interrupt | libero_goal task | 3 | 3 | 3 | +0 / -0 | 19 M / 1 R / 0 X |

Reading: action-level methods (consensus selection over 8 seeded samples, stall-based chunk cutting) do not
change outcomes under LIBERO-Pro's task perturbation: when the policy executes the wrong task, every sample
executes the same wrong task. On the same cells the agentic Harness VLA moves 2 -> 14 (object) and 3 -> 11 (goal)
with the success cue, 3 -> 6 (goal) without it.
