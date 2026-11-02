# tube-steer (HarnessStudio v2 method package, rtcfpp-20261101 round 5)

New training-free mechanism; not a reproduction of any cited paper. One action_op for openpi `pi0_fast_libero`: when
the gripper approach axis leaves the same-instruction success tube (a frozen memory of decision-time gripper z-axes from
127 S2 successes, init 10-29) by > 15 deg on 2 consecutive decisions with the first arming by step 195, replace the
rotation component along the restoring axis with a command toward the nearest tube orientation. Self-contained (no other
method package imported). Entries: `tube-steer`, dev-only control `tube-steer-ortho` (duration-capped).
Install: `uv pip install --python harnessstudio/.venv/bin/python --no-deps -e harnessstudio/methods/tube_steer`.
Tests: `pytest harnessstudio/methods/tube_steer/tests`; replay `scripts/replay_tube.py`. Evidence:
`runs/rtcfpp-20261101/implementation/tube-steer/`.
