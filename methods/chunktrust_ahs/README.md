# hs-chunktrust-ahs: ChunkTrust adaptive horizon selection (`chunktrust-ahs`, v2)

*ChunkTrust: Adapting Execution Horizons for Robot Policies with Action-Expert Evidence*,
arXiv:2609.39754v2. Upstream code: `hf618/ChunkTrust@a9fd57412b3a19a0728cd726b0558dd89395fb84`,
`src/chunktrust/gr00t_ahs.py` (MIT). Ported from HarnessStudio v1 (`harnessstudio-ext-g`,
`harnessstudio/methods/chunktrust_ahs/`). Only the training-free part (AHS) is ported; QHA trains an
auxiliary model and is excluded.

| | |
|---|---|
| Entry point | `chunktrust-ahs` → `hs_chunktrust_ahs:config` |
| Slots | `proposer` (`VelocityReadoutProposer`: K=1, `readouts: ["velocity"]`) + `executor` (`AHSExecutor`, `min_k=4`, `per_step=False`) |
| Fidelity | `faithful`, `substitutions: []` (as v1). It is an adapted reproduction only in setting (A1–A6 under `disclosures:` in `hs_chunktrust_ahs/paper.yaml`) |
| Requires | `act, chunk, seeded_sampling, velocity_readout` |
| Categories | primary `execution_monitoring` (as the RoboMetaHarness paper card), secondary `uncertainty_compute` |
| Null | `derived`: the readout is requested and AHS plans every decision, but 16 actions run (v1's shadow AHS) |
| Vendored | `chunktrust_gr00t_ahs.py` (upstream numerics), `horizon.py` (v1 episode state), `CHUNKTRUST_LICENSE`, all byte-identical to v1 and pinned in `PROVENANCE.json` (unchanged from v1) |

## What is reproduced

Each decision draws one native sample (sample 0, the native noise stream) together with the policy's
flow-velocity readout: the action decoder's `pred[:, -H:]` at every denoising step, `[4, 16, 32]` float32
on GR00T N1.5 (32 = the padded model action width). The executor calls the vendored
`select_exec_k_horizon` through v1's `ChunkTrustAHS`:

- **Intra-chunk evidence.** For each candidate k ∈ {4, 8, 12, 16}, the share of high-frequency spectral
  energy (cut 0.25) of the zero-padded velocity prefix at each denoising step. Then the window-RMS
  deviation over the steps (W_τ = 3) is min-max normalized across candidates, and q = exp(−4 z).
- **Inter-chunk evidence.** The speed-uniformity CV of the last 20 − k executed actions followed by the
  first k planned actions, min-max normalized, and p = exp(−4 u). Without enough history, intra only.
- **Selection.** q_mix = 0.5 q + 0.5 p, times a Thompson sample from a per-candidate Beta posterior
  (a0 = b0 = 1, Gaussian-kernel update σ = 10, forgetting ρ = 0.99, η = 1). k = round(Σ prob · k)
  (expected-round, T_sel = 1.0); with probability 0.05, a uniformly random candidate. The posterior
  and the rng (`default_rng(0)`) persist through the episode.
- **Execute** the first k actions, then append them to the history.

All 19 parameters are v1's `AHS_PARAMS` (= `ahs_reference.json` of the original ChunkTrust conversion
run, sha256 `ba8dd5b2…`). `velocity_dims: "all"` uses all 32 padded dims, as the upstream overlay.

## Port notes (v1 → v2)

- **P1 readout.** v1 recorded the velocity with a server-side `Sampler` slot (`VelocityReadout.on_step`)
  and read `extras["sampler"]["velocity"]`. v2 declares `readouts = ("velocity",)` on the proposer, so the
  DecisionLoop requests the policy readout with the sample, and the executor reads `extras["velocity"]`.
  On GR00T N1.5 both are the same tensor. `extras["native_length"]` and the timesteps are not used;
  `used_h = min(16, chunk length)` as in v1.
- **P2 column order.** v1's flat GR00T rows were the sorted GR00T keys (base_motion, control_mode,
  position, rotation, gripper); the v2 wire order is vla-eval's (position, rotation, gripper, base_motion,
  control_mode). The math does not depend on the column order (only row norms of action differences use
  the actions), but float32 summation does. So when the bound layout has the five GR00T groups,
  `column_order: "v1"` permutes the chunk and the executed actions back to v1's order, as C-contiguous
  arrays (a strided history also changes `np.linalg.norm`'s float32 summation order). Other layouts (the
  7-D fake stack) keep their native order.
- **P3 null.** v1's derived null with the velocity sampler on is v2's derived null with the readout on.
- **P4 diagnostics.** v1's executor had no `diagnostics()`. v2 adds a compact one, written to every
  decision row: k, choose mode, candidates, expected stride, z, q, p, q_mix, θ, posterior, column order.

### Parity with v1

`check_parity.py` loads v1's own files (`modules.py`, `horizon.py`, `chunktrust_gr00t_ahs.py`, and
`AHS_PARAMS` parsed from v1's `__init__.py`). It runs both executors side by side on randomized episodes:
GR00T-shaped velocity traces `[4, 16, 32]` (iid, smooth, alternating, step-trend and mixed, with zero,
constant or noisy padding dims) and 16×12 action chunks (v1 order for v1, wire order through the bound
layout for v2). Each episode records the same executed prefixes: candidate k, episode-end truncation, or
null-style 16.

| Run | Episodes | Decisions | k equal | Full info equal | Final state equal |
|---|---|---|---|---|---|
| seed 0 (default) | 400 | 6,632 | 6,632 | 6,632 | 400 |
| seed 1 | 600 | 10,975 | 10,975 | 10,975 | 600 |
| fake shapes (`[4, 6, 7]`, 6×7), seed 0 | 50 | 212 | 212 | 212 | 50 |

The full info dict is compared exactly through JSON. The final state is the history, the Beta posterior
and the rng state. The runs cover expected-round and ε-random choices, and the speed term is active on
94% of decisions. Without the column permutation (`column_order: "native"`), k still matched on every
decision, but the info floats differed on 85% of them. The first version of the port also differed in
63/6,632 info dicts because the permuted arrays were not C-contiguous.

### On the fake stack (tier 1, profile `fake-t2`)

The chunks have 6 steps and the velocity is a synthetic `[4, 6, 7]`, so only the candidate 4 fits
(`strides = [4]`), and every candidate decision plans k = 4 (expected-round or ε-random, both 4). The
null executes the full 6-step chunks and reproduces the baseline actions.

## Install and verify

From the repository root:

```bash
~/.local/bin/uv pip install --python .venv/bin/python -e methods/chunktrust_ahs
.venv/bin/hs list
.venv/bin/hs verify chunktrust-ahs --tier 0     # registration, buildable components, a profile offers velocity_readout
.venv/bin/hs verify chunktrust-ahs              # tier 1 on fake-t2 (seeded chunks + synthetic velocity readout)
.venv/bin/python methods/chunktrust_ahs/check_parity.py   # v1 parity (needs the v1 checkout; --v1 or $HARNESSSTUDIO_V1_CHUNKTRUST)
```

## Run on GR00T N1.5 / RoboCasa365

Use the HarnessStudio GR00T N1.5 server (`harnessstudio/policies/gr00t_n15_robocasa365.py`, launched with
`--chunk_size null`), which serves the `velocity` readout, with profile `gr00t-n15-robocasa365`:

```bash
.venv/bin/hs eval chunktrust-ahs \
  --config /home/sejain/research/cvpr27/vla-eval-v0.8.0/configs/benchmarks/robocasa365/smoke.yaml \
  --policy-url ws://127.0.0.1:8105 --contract robocasa365 --profile gr00t-n15-robocasa365 \
  --arms baseline,null,candidate --out runs/chunktrust-ahs-smoke
```

Cost: one sample plus the velocity readout per decision. The candidate re-plans every 4–16 steps, so it
needs up to 4× the null's policy calls (v1: 45 vs 29 on CloseToasterOvenDoor trial 1, mean executed k
≈ 9.6 in the original conversion run). No paper number transfers (A1). v1's local evidence on this
checkpoint: base16 scored 20/31 and AHS 15/31 on an atomic_seen development subset (not significant,
p = 0.18).
