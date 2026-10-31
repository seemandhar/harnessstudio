# Overnight summary, 2026-10-31 (HarnessStudio v2)

Status: draft, updated as runs finish.

## Framework

HarnessStudio v2 runs as a harness server on allenai `vla-eval==0.8.0` (pip dependency, no fork). Papers are
method packages (`hs new`, `hs verify` tiers 0/1/2, `hs eval`, `hs prepare`, `hs pair`, `hs report`).

| Check | Result |
|---|---|
| LIBERO, direct vs through the harness (`identity`) | 5/5 episodes bitwise identical |
| LIBERO-Pro, direct vs through the harness | 3/3 episodes bitwise identical |
| LIBERO, three arms with the seeded pi0.5 server | null vs baseline 5 M / 0 R / 0 X |
| RoboCasa365 + GR00T N1.5, CPU rendering | gate 2 M / 0 R / 0 X; `hs eval medoid-bon` 4 cases 3 M / 1 R / 0 X |
| Tier 2, `medoid-bon` on real LIBERO | PASS (2 M / 0 X) |
| Core code review | 3 critical/high bugs fixed with regression tests (3 fail on the old code) |

## Papers ported tonight

| Paper | Package | Fidelity | Predeclared subset | Result |
|---|---|---|---|---|
| Harness VLA (zero-shot) | `harness-vla` (NRP qwen3-vl planner) | adapted | LIBERO-Pro Goal-T + Goal-S, 100 cases | 30/100 vs baseline 23/100 (p = 0.32) |
| Harness VLA (zero-shot) | `harness-vla-sol` (gpt-6.1-sol planner) | adapted | same 100 cases | **55/100 vs 23/100 (+41 / -9, p = 5.6e-6)**; Goal-T 27/50, Goal-S 28/50 (paper zero-shot: 79.0 / 31.0) |
| Harness VLA (zero-shot) | `harness-vla-sol` | adapted | all 8 LIBERO-Pro cells, 10 tasks x 2 eps (160) | **84/160 (52.5%) vs 39/160 (24.4%), +61 / -16, p = 2.4e-7**; better in every cell |
| Harness VLA (few-shot) | `harness-vla-sol-fewshot` (prepare-stage task memory) | adapted | Goal-T + Goal-S held-out episodes, 60 exactly matched cases | 38/60 vs zero-shot 33/60 (+11 / -6, p = 0.33) vs baseline 15/60 (+27 / -4) |
| Harness VLA, **no success cue** (ablation) | `harness-vla-sol-nocue` | adapted | Goal-T + Goal-S episodes 0-1 (40, same scenes) | **17/40 vs cue version 24/40 vs baseline 9/40**: about half of the gain depends on telling the planner the goal is not yet reached (an oracle from success-termination) |
| SCALE (OpenVLA decoding) | `scale` | see results doc | LIBERO-10, 10 tasks x 6 episodes (60) | **38/60 (63%) vs greedy 29/60 (48%), +16 / -7, p = 0.093**; paper 52.7 -> 63.3; null vs baseline 60 M / 0 X |
| SCALE (OpenVLA decoding) | `scale` | see results doc | LIBERO-Goal, 10 tasks x 3 episodes (30) | 24/30 vs 23/30 (p = 1.0); paper 77.7 -> 82.3 |
| FastHarness | `fast-harness` (gpt-6.1-sol reviewer; experience frozen by `hs prepare`) | adapted | RoboDojo build_tower (+ stack_bowls), held-out layouts, cold vs warm | tier 2 PASS; port matches the original code on 198,542 randomized checks; **token savings not reproduced**: warm vs cold prompt tokens -11% / -16% per step (build_tower), +26% (stack_bowls) vs the paper's -57%; experience features rarely match across layouts; no success claim (n = 1-2) |

Details: `harness_vla.md`, `scale.md`, `fast_harness.md` in this folder.

## Audit of the new packages

A read-only compliance audit (10:50 PT) found no critical violations in Harness VLA, SCALE or FastHarness: no
simulator state or evaluator labels in method code, sensor-only benchmark additions, nothing trained, memory
built only from development data, null arms reproducing baselines, and headline tables that recompute from the
records. Required disclosures are now in the paper cards and results docs. The most important one: **Harness
VLA's planner is told every turn that the goal is not yet reached**, which relies on LIBERO ending episodes on
success (an implicit verifier the baseline and a real robot lack). The follow-up ablation measured it: without
the cue the harness solves 17/40 instead of 24/40 on the same scenes (baseline 9/40), so about half of the gain
depends on it.
SCALE's "episodes 3-5" share fixture placements with episodes 0-2 (valid pairing inside the run, different
scenes from sequential episodes 3-5). Framework gaps it found are in `docs/core-todo.md` (source fingerprints and
pairing checks, charged external tools, namespace-aware dev-case refusal).

## Incidents

- 01:02 PT: GPU 0 (another user's job) hit Xid 62 -> 154; the NVIDIA driver deadlocked machine-wide. A reset
  attempt could not unload the driver; it recovered on its own after all GPU holders were killed, so no reboot
  was needed. GPU 0 still reports "Recovery Action: Reset" (clears at the next reboot). GSP firmware was already
  off: if this recurs, suspect GPU 0 hardware/power.

## Lesson

LIBERO scenes depend on run history (one env per task, fixtures re-placed every reset): paired arms are exact
only for identical case lists in identical order. `hs eval` satisfies this; the guide now says so.

## Open

- Deferred core changes: `docs/core-todo.md` (task_id in case keys, contract action bounds, RoboDojo profile).
