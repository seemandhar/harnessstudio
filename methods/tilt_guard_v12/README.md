# tilt-guard v1.2 (HarnessStudio v2 method package, rtcfpp-20261101)

New training-free mechanism; not a reproduction of any cited paper. tilt-guard v1.1 with G removed (order Z -> B),
per amendment A2 of `runs/rtcfpp-20261101/campaign-inputs/preregistration-r1.md`. `action_op.py` and `geometry.py` are
byte-identical to v1.1; `composite.py` forces `guard=False`. No dependency on hs-band-guard.

Entry points: `tilt-guard-v12-{b,bz,bsat,bsatz,z}` and the dev-only controls `tilt-guard-v12-{b-yaw,bsat-yaw}`.
Install: `uv pip install --python harnessstudio/.venv/bin/python --no-deps -e harnessstudio/methods/tilt_guard_v12`.
Tests: `pytest harnessstudio/methods/tilt_guard_v12/tests`; replays `scripts/replay_dev.py` (T1-T10),
`scripts/replay_v12.py` (v1.2 == v1.1 with G disabled; activation counts). Evidence:
`runs/rtcfpp-20261101/implementation/tilt-guard-v12/`.
