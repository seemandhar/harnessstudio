# medoid-bon (v3)

CycleVLA's Minimum Bayes Risk (MBR) consensus selection over K seeded samples, as a HarnessStudio v2
method package. Paper: *CycleVLA: Proactive Self-Correcting Vision-Language-Action Models via Subtask
Backtracking and Minimum Bayes Risk Decoding*, arXiv:2601.02295v2 (Sec. IV-B, V-A, Table V). Source
notes: `hs_medoid_bon/CYCLEVLA_SOURCE.json`.

| | |
|---|---|
| Entry point | `medoid-bon` (`hs_medoid_bon:config`) |
| Slots | proposer `SeededPool` (K = 8) + selector `MBRSelect` |
| Category | `proposal_selection` |
| Fidelity | `component` (MBR selection only) |
| Requires | `act`, `chunk`, `seeded_sampling`, `isolated_streams` |
| Null | derived (the null draws the same 8 samples and executes sample 0) |

## What is reproduced

The MBR rule as published:

- each of the N = 8 samples becomes a trajectory by accumulating its end-effector translation and
  rotation deltas over the first 16 steps (a feature in R^(6H));
- the gripper is excluded (so are `base_motion` and `control_mode`, which are not in the paper's action
  space); no normalization;
- d = L2; the executed sample is argmin_i of the mean distance to the other samples (ties go to the
  lower index). It selects an existing sample and never averages.

The end-effector dims are found by name in the policy's public action layout (`bind(embodiment)`,
vla-eval names `position` and `rotation`). `MBRSelect(dims=[...])` overrides the layout.

## What is not reproduced

CycleVLA's progress-aware VLA (fine-tuned), the VLM failure predictor and planner, and subtask
backtracking. Hence `component`: no CycleVLA number transfers to this method.

## Changes from v1/v2 (HarnessStudio v1)

- v3: ported to HarnessStudio v2. Chunks arrive as arrays `[H, A]`; dims come from
  `Embodiment.action_layout` (`position`, `rotation`) instead of `extras['policy']['action_layout']`
  (`action.end_effector_position/rotation`). Algorithm, constants and arithmetic are unchanged: on 300
  random cases the v3 selector picks the same index as v2.
- A policy whose action spec has no `position`/`rotation` blocks (for example vla-eval's pi0 server,
  which advertises a single `actions: RAW` block with 0 dims, so the layout is empty) makes `select`
  raise, and the candidate episode is recorded invalid. Pass `dims=` to run there.

## How to run

```bash
uv pip install --python .venv/bin/python -e methods/medoid_bon
.venv/bin/hs list
.venv/bin/hs verify medoid-bon --tier 0      # static checks
.venv/bin/hs verify medoid-bon               # tier 1 on the fake stack (fake-t1)
```

Paired arms on a real benchmark need a seeded policy server (one with `seeded_sampling` and
`isolated_streams`, for example GR00T N1.5 on RoboCasa365 behind the HarnessStudio policy mixin):

```bash
.venv/bin/hs eval medoid-bon \
  --config ../vla-eval-v0.8.0/configs/benchmarks/robocasa365/smoke.yaml \
  --policy-url ws://127.0.0.1:8101 --contract robocasa365 --profile gr00t-n15-robocasa365 \
  --arms baseline,null,candidate --out runs/medoid-bon-smoke
```

Cost: 8 policy samples per decision in the null and candidate arms (1 in the baseline).
