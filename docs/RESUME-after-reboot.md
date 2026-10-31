# Resume HarnessStudio v2 work after the 2026-10-31 reboot

Why the reboot: at 01:02:01 PDT GPU 0 (PCI 0000:01:00, board serial 1711124001296) hit **Xid 62** (internal
micro-controller halt) under another user's job, then Xid 154 (GPU reset required). The NVIDIA driver
deadlocked (even its `nv_queue` kernel threads wait on `os_acquire_rwlock_write`), so rmmod/FLR was impossible.
GSP firmware was already off (`EnableGpuFirmware: 0`), so per the July notes suspect GPU 0 hardware/power.

After the reboot (in order):

1. Check the GPUs: `nvidia-smi`; per-GPU torch matmul for 4-7.
2. Restart the policy servers (all were killed):
   - stock pi0.5 LIBERO, GPU 4, :8101: `bash /home/sejain/research/cvpr27/hs2-gpu/start_server.sh && bash /home/sejain/research/cvpr27/hs2-gpu/wait_healthy.sh`
   - seeded pi0.5 (HarnessStudio T1), GPU 6, :8106: see `docs/results/m3_libero_eval_pipeline.md` (vla-eval serve with
     `/home/sejain/research/cvpr27/hs2-gpu/configs/pi05_libero_hs.yaml`, `CUDA_VISIBLE_DEVICES=6 XLA_FLAGS=--xla_gpu_autotune_level=0
     OPENPI_DATA_HOME=/bigdata/sejain/hs2-cache/openpi PYTHONSAFEPATH=1 PYTHONPATH=<this repo>`)
   - seeded GR00T N1.5, GPU 5, :8105: `/bigdata/sejain/hs2-cache/envs/launch_gr00t_n15_8105.sh`
3. Re-run what the hang interrupted:
   - tier 2 on real LIBERO: `hs verify medoid-bon --tier 2 --config /home/sejain/research/cvpr27/hs2-gpu/configs/libero_spatial_hs.yaml
     --policy-url ws://127.0.0.1:8106 --contract libero --profile pi05-libero-hs --out /home/sejain/research/cvpr27/hs2-gpu/results/tier2_medoid
     --benchmark-overrides '{"max_tasks": 1, "episodes_per_task": 2}'` (run from /home/sejain/research/cvpr27/vla-eval-v0.8.0)
   - RoboCasa365 GR00T gate with CPU rendering + `hs eval medoid-bon`: `/home/sejain/research/cvpr27/hs2-gpu/robocasa365/run_gr00t_cpu_render.sh`
   - LIBERO-Pro parity: `cd /home/sejain/research/cvpr27/hs2-gpu/libero_pro && SUITE=libero_spatial_with_mug PERT=none EPS=3 TAG=_with_mug ./parity.sh`,
     and the remaining baseline cells with `configs/libero_pro_bddlinstr_8101.yaml` (correct lan/task instructions).
   - Prefer CPU rendering (`render: cpu`) for benchmark containers where speed allows: it is bitwise deterministic.
4. Then the first papers (docs/next-papers.md): RTCF, Harness VLA (zero-shot), FastHarness.
