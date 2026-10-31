# rtcf: Retrieve in Time, Correct in Frequency (HarnessStudio v2 method)

Port of RTCF (arXiv:2608.04527 v1/v2; **withdrawn by the authors in v3, 2026-10-28**) around frozen pi0-FAST on
LIBERO. No code was released, so the port implements v2 Eq. 5-11 / Alg. 1. Fidelity `tf_variant`: the fitted
SigLIP-PCA projection is omitted (raw l2-normalized SigLIP features), and every constant the paper does not
report is frozen a priori (see `hs_rtcf/paper.yaml` substitutions and `runs/rtcf-20261031/port/conversion-plan.md`).

## What runs
- `hs_rtcf.action_op:RTCFActionOp` (slot `action_op`, derived null). At every policy invocation:
  SigLIP descriptor of the agentview image (service `scripts/siglip_service.py`) -> PMA frontier update over the
  whole frozen success memory (`hs_rtcf/pma.py`, Eq. 5-9) -> read the aligned record -> clipped low-frequency DCT
  residual on the motion dims in pi0-FAST's coefficient space (`hs_rtcf/spectral.py`, Eq. 10-11) -> magnitude limit.
  No memory artifact = collection mode (native chunk, diagnostics carry descriptor + full chunk + PNG).
- `hs_rtcf.prepare:Preparer` (`collect_arm: candidate` = native in collection mode): successful development
  episodes -> `memory.npz` (descriptors, offsets, quantized coefficients, decode-failure flags) + `memory.json`.
- Ablations (inferred rules): `rtcf-framenn` (single-frame NN retrieval), `rtcf-timedomain` (time-domain residual).

## Pure-function API (parity checks)
```python
from hs_rtcf.spectral import rtcf_correct, coefficients, load_scale, dct_matrix, quantize, dequantize
corrected, info = rtcf_correct(native_chunk[H, A], memory_coef[H, A], scale[A], motion_dims, f_cut, delta_max,
                               lambda0, action_limit=1.0, mode="spectral")   # memory_coef = FAST tokens / 10
from hs_rtcf.pma import local_cost, pma_init, pma_step, pma_select, frame_nn_select, local_index
J = pma_init(local_cost(z_1, Z)); J = pma_step(J, local_cost(z_t, Z), local_index(offsets), v_max, gamma)
i_star, b_star, d = pma_select(J, offsets, t)            # 0-based positions, lowest index on ties
```

## Run (evaluator only; GPU work is never started by the method)
```bash
H=/home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio; M=$H/methods/rtcf
# SigLIP service (own uv venv; weights google/siglip-so400m-patch14-224 @ d04cf29, sha256 30a00c26...)
CUDA_VISIBLE_DEVICES=G HF_HUB_OFFLINE=1 uv run $M/scripts/siglip_service.py --snapshot SNAPSHOT_DIR \
    --revision d04cf29fca7b6374f74d8bea1969314492266b5e --port 8149
# configs (quat_no_antipodal: true; dev init states 10-49 seed 7; eval init states 0-9 seeds 101-105)
python3 $M/scripts/make_configs.py --suite libero_10 --kind dev --seed 7 --out $M/configs/generated
cd /home/sejain/research/cvpr27/vla-eval-v0.8.0
export HS_RTCF_ENCODER_URL=http://127.0.0.1:8149
$H/.venv/bin/hs prepare rtcf --config $M/configs/generated/dev_libero_10_seed7_t0.yaml --contract libero \
    --policy-url ws://127.0.0.1:8140 --profile $M/configs/pi0fast-libero-hs.yaml --port P --out DEV_t0
$H/.venv/bin/python $M/scripts/merge_dev.py PREP DEV_t0 ... DEV_t9
$H/.venv/bin/hs eval rtcf --config $M/configs/generated/eval_libero_10_seed101_t0.yaml --contract libero \
    --case-namespace seed101 --policy-url ws://127.0.0.1:8140 --profile $M/configs/pi0fast-libero-hs.yaml \
    --arms baseline,candidate --prepared PREP --port P --out EVAL
```
Checks: `scripts/check_native_equivalence.py` (collection mode == baseline hashes), `scripts/probe_fast_integrality.py`
(10 s C integral on real chunks), `scripts/check_split.py` (dev init states disjoint from eval), `scripts/summarize.py`
(paired table vs the withdrawn paper rows, exact McNemar).

## Known limitation
HarnessStudio logs diagnostics after a decision's actions ran; an episode ending mid-chunk (every LIBERO success)
cancels that last decision, so each memory misses its final invocation. Proposed framework extension
`decision-pre-diagnostics` (the Preparer already prefers its `decision_pre` events when present).
