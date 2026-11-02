# band-guard (HarnessStudio v2 method)

Original RoboMetaHarness method from run `rtcfpp-20261101` (round-1 GENERAL proposal `r1g-p1`, approved with
conditions C1-C10 by critic `a8623158c27233d0a`). Not a paper port.

## What it does

`pi0_fast_libero` (openpi 981483d, `extra_delta_transform=True`) predicts `m = a - s` on dims 0-5; openpi's
`AbsoluteActions` adds the decision-time state back. When the gripper's rx state leaves the training band
(`s_rx > 3.2705`), `m_rx` cannot go below its training floor (`-3.306`), so the executed `a_rx = m_rx + s_rx` keeps
rotating outward. One `action_op` re-expresses the saturated delta relative to the nearest in-band state:

```
a'_rx = m_rx + clip(s_rx, state_q01_rx, state_q99_rx)        (every row; candidate executes rows[:5])
```

It fires only when `s_rx` is outside `[1.5004, 3.2705]` AND `min m_rx <= -3.3064 + 0.02` over rows[:5] (above the
band), or `max m_rx >= -1.4915 - 0.02` (below the band). Otherwise the input chunk object is returned unchanged
(bitwise native). Algebraically: a gated `AbsoluteActions(clip(s))` on rx. It only makes sense for checkpoints with
delta-to-state output reconstruction.

| Entry point | Role |
|---|---|
| `band-guard` | candidate |
| `band-guard-signctrl` | development-only norm-matched control: same gate, same magnitude, sign from a public coin (sha256 of the episode's first public state + decision counter) |

All scalars are frozen in the card `params` (`hs_band_guard/paper.yaml`); the op refuses band/saturation edges that
differ from the pinned checkpoint `norm_stats.json` (sha256 `4ece6c43...`, copy in `hs_band_guard/assets/`).
Budgets: native policy calls, 0 LLM/tool calls, derived null, no RNG.

## Checks

```bash
H=/home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio
uv pip install --no-deps --python $H/.venv/bin/python -e $H/methods/band_guard
$H/.venv/bin/hs verify band-guard --tier 0 && $H/.venv/bin/hs verify band-guard
$H/.venv/bin/hs verify band-guard-signctrl --tier 0 && $H/.venv/bin/hs verify band-guard-signctrl
cd $H/methods/band_guard && $H/.venv/bin/python -m pytest -q tests        # unit + active-path fake-stack tests
$H/.venv/bin/python scripts/replay_dev.py --run RUN_DIR --out OUT.json    # CPU replay on dev identity rollouts
```

Tier 1 alone never activates the op (the fake contract exposes `state`, not `states`); `tests/test_fake_stack.py`
exercises the active path through the real in-process harness server with a fake benchmark exposing `states`.

## Development screen (evaluator)

Configs and the `hs eval` argv template: `runs/rtcfpp-20261101/implementation/band-guard/screen-configs/`
(`--profile methods/rtcf/configs/pi0fast-libero-hs.yaml`, cadence 5, libero_10 init 30-37, env seed 7, horizon 520,
`quat_no_antipodal`, `LIBEROSelect` offset + `match_sequential_resets`). The sign control runs on development scenes
only, never as a Campaign arm.
