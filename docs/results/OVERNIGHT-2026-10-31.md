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
| SCALE (OpenVLA decoding) | `scale` | see results doc | LIBERO-10, 10 tasks x 3 episodes | 20/30 vs greedy 16/30 (p = 0.45); paper greedy 52.7 -> SCALE 63.3 |
| FastHarness | `fast-harness` (gpt-6.1-sol reviewer) | adapted | RoboDojo build_tower | in progress |

Details: `harness_vla.md`, `scale.md`, `fast_harness.md` in this folder.

## Incidents

- 01:02 PT: GPU 0 (another user's job) hit Xid 62 -> 154; the NVIDIA driver deadlocked machine-wide. A reset
  attempt could not unload the driver; it recovered on its own after all GPU holders were killed, so no reboot
  was needed. GPU 0 still reports "Recovery Action: Reset" (clears at the next reboot). GSP firmware was already
  off: if this recurs, suspect GPU 0 hardware/power.

## Open

- Deferred core changes: `docs/core-todo.md` (task_id in case keys, contract action bounds, RoboDojo profile).
