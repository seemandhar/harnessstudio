# `hs eval` end to end on real LIBERO with the seeded pi0.5 server (2026-10-31)

**Result:** all three arms ran on the same 5 seeded LIBERO episodes; every episode valid; the derived
null reproduced the baseline step for step in all 5 pairs (**5 M / 0 R / 0 X**) while paying the
candidate's 4x sampling compute. This checks the pipeline, not a method: `test-pick-last-k4` executes the
last of 4 seeded samples and is not expected to help.

| | |
|---|---|
| Benchmark | vla-eval 0.8.0 LIBERO, libero_spatial task 0, episodes 0-4, Docker |
| Policy | `harnessstudio/policies/pi05_openpi.py` (vla-eval Pi0ModelServer + HarnessPolicyMixin), pi05_libero, GPU 6, profile `pi05-libero-hs` (seeded, isolated 10-step chunks) |
| Method | `harnessstudio.verify.testing_methods:pick_last_k4` (proposer K=4, selector "last") |

Env seed (clarification, env-seed audit
`/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/repro-next-20261105/audit/env_seed_audit.md` (sha256
`1c7dabbe…`)): the config sets `seed: 7` and no `env_seed`, so vla-eval 0.8.0 calls `env.seed(7)`
(`vla_eval/benchmarks/libero/benchmark.py:119,200`); the pi0.5 server sends no seed. This matches openpi's LIBERO
example (`env.seed(7)`). All three arms used it, so pairing is unaffected.

| Arm | Successes | Policy samples | Wall time (s) |
|---|---|---|---|
| baseline | 5/5 | 44 | 40.3 |
| null | 5/5 | 176 | 68.7 |
| candidate | 5/5 | 216 | 84.2 |

Null attribution: 5 M (every observation and action hash equal at every step), 0 R, 0 X.

T1 server checks (`scripts/check_t1_pi05.py`): same seed twice bitwise identical; a sample is independent
of the other seeds in the request; different seeds differ; a restarted server returns bitwise-identical
chunks (max abs diff 0.0); about 0.22 s per sample.

Command:

```bash
hs eval harnessstudio.verify.testing_methods:pick_last_k4 --config configs/libero_spatial_hs.yaml \
  --policy-url ws://127.0.0.1:8106 --contract libero --profile pi05-libero-hs \
  --out results/eval_picklast_k4 --benchmark-overrides '{"max_tasks": 1, "episodes_per_task": 5}'
```

GR00T N1.5 (RoboCasa365) seeded server: `harnessstudio/policies/gr00t_n15_robocasa365.py`, checks in
`scripts/check_t1_determinism.py`: bitwise determinism for K=1/8, isolated streams, identical after a
restart (11/11 chunks), [16, 12] chunks, native replies unchanged by seeded requests, velocity readout
available; v1's Human300 checkpoint-120000 loads on vla-eval's server. K=8 about 0.35 s.

## Tier 2 on real LIBERO (`hs verify medoid-bon --tier 2`)

PASS: 2 paired cases (libero_spatial task 0, episodes 0-1), every episode valid, null vs baseline 2 M / 0 R / 0 X.
baseline 2/2 (22 policy samples, 27 s), null 2/2 (176 samples, 54 s), candidate 2/2 (128 samples, 39 s).
