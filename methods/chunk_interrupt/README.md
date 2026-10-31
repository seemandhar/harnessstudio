# chunk-interrupt (v3)

A per-step executor that cuts the current action chunk, and lets the policy re-plan, when the arm stops
following its commands. A training-free variant of the truncation step of *VLA-Corrector: Lightweight
Detect-and-Correct Inference for Adaptive Action Horizon*, arXiv:2607.01804v1.

| | |
|---|---|
| Entry point | `chunk-interrupt` (`hs_chunk_interrupt:config`) |
| Slots | executor `StallCut` (`min_k=4`, `per_step=True`, `window=4`, `rho=0.05`) |
| Category | `execution_monitoring` |
| Fidelity | `tf_variant` |
| Substitution | VLA-Corrector trained latent-dynamics corrector -> public commanded-vs-achieved end-effector displacement check; truncation only, no gradient recovery |
| Requires | `act`, `chunk` (no seeded sampling: one request per decision, as natively) |
| Null | derived (the executor runs, but cuts are ignored and the native length executes) |

## What is reproduced

Only the truncate-and-replan behaviour. After each executed step t >= `window`:

- commanded = (sum of the `position` action block over the last `window` steps) x 0.05 m per unit
  (robosuite OSC_POSE `output_max/input_max` of the pinned `default_pandaomron.json`,
  sha256 `4f91b12b…`; check with `controller_config_path()` when robosuite is installed);
- achieved = end-effector position now minus `window` steps ago, from the public observation
  (`embodiment.eef_position(obs)`: RoboCasa365 `state.end_effector_position_relative`, LIBERO
  `states[0:3]`, fake `state[0:2]`);
- a step is "low" if |achieved| / |commanded| < `rho` (ratio 1 when |commanded| < 1e-6); after
  `window` consecutive low steps the chunk is cut. A cut is honoured only from step `min_k` on.
  With the defaults the earliest cut is step 7.

## What is not reproduced

VLA-Corrector's latent-space vision monitor and its online gradient guidance. No improvement is claimed:
in v1 the ChunkTrust H1 stall detector did not predict failure (P(fail|fired) 0.55 vs base 0.53).

## Changes from v1/v2 (HarnessStudio v1)

- v3: ported to HarnessStudio v2. The commanded block is `position` in `Embodiment.action_layout`
  (was `action.end_effector_position` in `extras['policy']`); the achieved position comes from the
  public observation through the contract's `eef_position` (was `extras['policy']['proprio_eef']`).
  Window, rho, scale and the stall rule are unchanged: on 300 random cases v3 makes the same
  continue/cut decision at every step as v2.
- If the policy's action layout has no `position` block or the benchmark exposes no end-effector
  position, v1 raised an error; v3 never cuts (the chunk runs natively) and records why in
  `StallCut.diagnostics` (the framework does not trace slot diagnostics). vla-eval's pi0 server
  advertises `actions: RAW` (0 dims), so on `pi05-libero-openpi` the layout is empty and this method
  is a no-op unless the executor gets `slice=[0, 3]`.
- When the end-effector position has fewer dims than the commanded block (fake stack: 2 vs 3), both are
  compared on the common leading dims. On LIBERO and RoboCasa365 both have 3, as in v1.
- v2's DecisionLoop does not enforce `min_k`; the executor keeps v1's rule (no cut before `min_k`) itself.

## How to run

```bash
uv pip install --python .venv/bin/python -e methods/chunk_interrupt
.venv/bin/hs list
.venv/bin/hs verify chunk-interrupt --tier 0     # static checks
.venv/bin/hs verify chunk-interrupt              # tier 1 on the fake stack (fake-t0)
```

Paired arms on RoboCasa365 with GR00T N1.5:

```bash
.venv/bin/hs eval chunk-interrupt \
  --config ../vla-eval-v0.8.0/configs/benchmarks/robocasa365/smoke.yaml \
  --policy-url ws://127.0.0.1:8101 --contract robocasa365 --profile gr00t-n15-robocasa365 \
  --arms baseline,null,candidate --out runs/chunk-interrupt-smoke
```

Cost: one policy sample per decision; cuts only shorten decisions, so the candidate may re-plan more
often than the null.
