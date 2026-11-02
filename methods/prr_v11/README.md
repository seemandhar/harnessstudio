# hs-prr-v11 (= hs-prr 1.1.0): PRR with the C7-revised correction phase

The single permitted W1 technical revision of `hs-prr` 1.0.0 (ruling `runs/rtcfpp-20261101/reviews/prr-c6-c7-ruling.json`).
ONLY the correction phase differs; `planner.py` and `geometry.py` are byte-identical to 1.0.0 and every non-correction
scalar in the cards is identical (test `test_non_correction_params_identical_to_v100`).

| | 1.0.0 | 1.1.0 |
|---|---|---|
| position row | `clip((p_cp-p)/0.0054)` per element | `(1.0 (p_cp-p) - 2.0 v)/0.05`, direction-preserving saturation |
| rotation row | `clip(log(R_cp R^T)/0.0324)` per element | `(1.0 log(R_cp R^T) - 4.0 w)/0.5`, direction-preserving saturation |
| budget | 2 decisions | 10 decisions (C4: abstain if f + L + 55 > 460) |

`v`, `w` = last step's displacement / world-frame rotation from consecutive public states. Kp 1 sets the robosuite OSC_POSE
goal to the checkpoint pose. Entries: `prr-v11`, dev-only `prr-v11-hold`, `prr-v11-ortho`. The evaluator keeps using the
frozen 1.0.0 `witness.py` (sha 44bde2b2; it imports `hs_prr.geometry`, which is identical) with the 1.1.0 table.
Design evidence and pre-check: `runs/rtcfpp-20261101/implementation/prr-v11/calibration/`.
