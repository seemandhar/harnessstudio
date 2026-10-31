# M2: RoboCasa365 (GR00T N1.5) and LIBERO-Pro (pi0.5) through HarnessStudio (2026-10-31)

Full notes, commands and scripts: `/home/sejain/research/cvpr27/hs2-gpu/robocasa365/README.md` and
`/home/sejain/research/cvpr27/hs2-gpu/libero_pro/README.md`. Both runs were cut short at about 01:00 PDT when
the host's NVIDIA driver wedged machine-wide (every CUDA/EGL/nvidia-smi client blocked in D state on
`os_acquire_rwlock_write`); everything below ran before that, or on CPU.

## RoboCasa365 + seeded GR00T N1.5

- vla-eval's RoboCasa365 image is not public, so the benchmark runs in-process (`--no-docker`) from a
  dedicated env (`/bigdata/sejain/hs2-cache/envs/robocasa365-bench`, robosuite 1.5.2 + robocasa365 1.0.1 at
  the Dockerfile's pinned commits, existing kitchen assets symlinked).
- v1's Human300 checkpoint-120000 loads in vla-eval's GR00T N1.5 server.
- Null-vs-baseline gate (`identity` baseline vs `medoid-bon` null, K=8, GPU EGL rendering): 2 cases,
  null paid 8x the samples and never changed an action. Old attribution said 1 M / 1 R / 0 X; the R pair
  had all 450 actions identical and only 3 non-decision frames differing (EGL renderer noise), which the
  revised attribution now counts as M with a noise note.
- Re-scored with the revised attribution the GR00T gate is 2 M / 0 R / 0 X.
- vla-eval's CPU render backend is bitwise deterministic across processes; with it the gate is 2 M on a CPU
  stand-in policy. GR00T + CPU rendering and the `hs eval medoid-bon` screen are scripted but not run (driver):
  `hs2-gpu/robocasa365/run_gr00t_cpu_render.sh`.
- Found and fixed: RoboCasa365 resets depend on what the process ran before, so `hs eval --no-docker` now runs
  each arm's benchmark in its own child process (Docker runs already get one container per arm).

## LIBERO-Pro + stock pi0.5

- Image `ghcr.io/allenai/vla-evaluation-harness/libero-pro:latest` (6.3 GB) works; perturbation data comes from
  HF `zhouxueyang/LIBERO-Pro` mounted into the container.
- Smoke baseline, task 0 x 2 episodes per cell, 12 of 21 cells before the driver wedge (spatial: 2/2 on
  original/swap/object/lan, 0/2 on task; object: 2/2 original, 1/2 swap and object, 0/2 task).
- Upstream issue: in vla-eval 0.8.0 the `lan` and `task` perturbations send the *original* instruction to the
  policy; a benchmark subclass fix is in `hs2-gpu/libero_pro/ext/hs2_liberopro_ext.py` (not yet in HarnessStudio).
- Found and fixed in HarnessStudio: perturbed suite names now use their base suite's horizon; case keys
  include the suite (no collisions across perturbations).
- Observation keys match the `libero_pro` contract.
