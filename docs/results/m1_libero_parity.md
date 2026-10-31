# M1 gate: the harness server is transparent on real LIBERO (2026-10-31)

**Result: 5/5 episodes bitwise identical** (every recorded action and state at every step, same outcomes)
between the benchmark connected directly to the policy and the benchmark connected through
HarnessStudio running `identity`.

| | |
|---|---|
| Benchmark | vla-eval 0.8.0 `LIBEROBenchmark`, libero_spatial, task 0, episodes 0-4, seed 7, max_steps 220, Docker image `ghcr.io/allenai/vla-evaluation-harness/libero:latest` |
| Policy | vla-eval 0.8.0 openpi `Pi0ModelServer`, `pi05_libero` (gs://openpi-assets), chunk_size 10, GPU 4, `XLA_FLAGS=--xla_gpu_autotune_level=0` |
| Harness | `hs serve identity --contract libero --profile allenai-passthrough` (server chunking, T0) |
| Protocol | policy server restarted before each run (openpi's RNG advances per inference call) |

| Episode | Direct | Through harness | Step-by-step |
|---|---|---|---|
| 0 | success, 77 steps | success, 77 steps | identical |
| 1 | success, 76 steps | success, 76 steps | identical |
| 2 | success, 83 steps | success, 83 steps | identical |
| 3 | success, 77 steps | success, 77 steps | identical |
| 4 | success, 74 steps | success, 74 steps | identical |

Harness traces: all 5 episodes valid, termination `environment`, one policy request per step (377 total),
about 10 s wall time per episode. Comparison: `scripts/compare_vla_eval_runs.py` on the two vla-eval
SQLite recordings.

Reproduce (commands in `/home/sejain/research/cvpr27/hs2-gpu/m1_gate.sh`):

```bash
# policy server (fresh), then either the benchmark directly or through the harness:
hs serve identity --policy-url ws://127.0.0.1:8101 --contract libero --profile allenai-passthrough \
   --arm candidate --host 127.0.0.1 --port 8201 --out results/m1_harness_pt/hs
vla-eval run --yes --config configs/m1_harness_pt.yaml --benchmark-field max_tasks=1 --benchmark-field episodes_per_task=5
python scripts/compare_vla_eval_runs.py results/m1_direct results/m1_harness_pt
```

On the fake stack (tests/test_stack.py) the policy additionally receives byte-identical observations in
the same order with and without the harness.
