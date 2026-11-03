# PCD counterfactual-image service

Frozen GroundingDINO-base + SAM2.1-L camera tracker + Big-LaMa, reproducing the PCD-Mask GDINO path of
Koorye/PCD@31b027e. See `runs/pcd-port-20261103/reports/perception_service.md` for the design, the file:line map,
the parity result and the GPU plan; `NOTICE` for licences.

- venv (outside harnessstudio/, never inside it): `runs/pcd-port-20261103/venvs/pcd-perception`
  (`uv venv --python 3.10 VENV && uv pip sync --python VENV/bin/python --index-url https://pypi.org/simple
  --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match --require-hashes
  requirements.lock`).
- start: `CUDA_VISIBLE_DEVICES=G VENV/bin/python -I -B serve.py --port 8141 --device cuda:0`
  (`-B`: no bytecode is written into this tree). Weights come from `readiness/weights_manifest.json`.
- API: GET /health; POST /reset {session, instruction}; POST /step {session, instruction, image_npy|image_png,
  return_masks?} -> counterfactual at input resolution + per-target evidence; POST /close. Client:
  `pcd_perception/client.py`.
- tests (fake models): `VENV/bin/python -I -B tests/test_engine_fake.py -v`.
Keep compiled/binary artifacts (venvs, .so, weights, build dirs, __pycache__) out of harnessstudio/.
