# tilt-guard v1.1 (HarnessStudio v2 method package, rtcfpp-20261101)

New training-free mechanism; not a reproduction of any cited paper. A copy of the frozen tilt-guard v1.0
(`harnessstudio/methods/tilt_guard`, never edited) with ONE added switchable flag for B, `saturation_arming`
(`off` = v1.0 outputs bitwise, `on` = Bsat, `yawctrl` = dev-only yaw control), per
`runs/rtcfpp-20261101/reviews/r2-residual-review.json` frozen_spec_p1. G is `hs_band_guard` (frozen, read-only).

Entry points: `tilt-guard-v11-{gb,gyaw,gbz,gz,gbzsign,gzsign,z,zsign}` (v1.0 variants, flag off, shadow saturation
diagnostic logged) and `tilt-guard-v11-{gbsatz,gbsat}` plus the dev-only controls `tilt-guard-v11-{gbsatz-yaw,gbsat-yaw}`.

Install: `uv pip install --python harnessstudio/.venv/bin/python --no-deps -e harnessstudio/methods/tilt_guard_v11`.
Tests: `pytest harnessstudio/methods/tilt_guard_v11/tests` (v1.0 suite + S3-S10); replays `scripts/replay_dev.py`
(T1-T10 counts) and `scripts/replay_sat.py` (S1, S2, S3 replay, S7). Evidence: `runs/rtcfpp-20261101/implementation/tilt-guard-v11/`.
