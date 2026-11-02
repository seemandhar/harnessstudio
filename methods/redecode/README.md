# ReDecode (HarnessStudio v2 controller method, rtcfpp-20261101 round 4)

New training-free mechanism; not a reproduction of any cited paper. On an openpi FAST decode failure (the zero chunk) of
the frozen `pi0_fast_libero`, re-query the policy with the public state shifted by one state-token bin (+, then -),
execute the first valid decode re-based to the true state, else fall back to the frozen tilt-guard v1.2 Z rule. The
selected chunk always passes through the frozen `hs_tilt_guard_v12` op (imported read-only).

Entry points: `redecode` (ReDecode -> Z), `redecode-b` (ReDecode -> Z -> B). Controls: `tilt-guard-v12-z`, `tilt-guard-v12-bz`.
Install: `uv pip install --python harnessstudio/.venv/bin/python --no-deps -e harnessstudio/methods/redecode`.
Tests: `pytest harnessstudio/methods/redecode/tests`; replay `scripts/replay_redecode.py`. Spec, review and evidence:
`runs/rtcfpp-20261101/{ideation/r4-control.json,reviews/r4-proposal-review.json,implementation/redecode/}`.
