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

(pending)
