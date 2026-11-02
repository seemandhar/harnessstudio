# hs-u5z 1.0.0 — U5+Z composition (rtcfpp-20261101, amendment A6.Z)

The single composition the A5 ruling allows: the frozen tube-steer (U5 = T, `hs-tube-steer` 1.0.0) composed with the
frozen tilt-guard v1.2 'z' zero-chunk substitution (`hs-tilt-guard-v12` 1.2.0), with Z acting only before U5's first
arming. It is a composite of two frozen components, not a new mechanism. Only `hs_u5z/glue.py` is new.

| entry | U5 | Z | role |
|---|---|---|---|
| `u5z-t` | tube-steer | tilt-guard v1.2 `z` (`TiltGuardV12ActionOp`, paper_z.yaml) | candidate |
| `u5zsign-t` | tube-steer | tilt-guard v1.1 `zsign` (`TiltGuardActionOp`, paper_zsign.yaml) | development-only control |

Frozen precedence, from `runs/rtcfpp-20261101/reviews/a5-decision.json` (`tz_composite_ruling.frozen_spec`) and
`reviews/u5z-proposal-review.json` (C5, C9). At every decision U5 is called first, on the native chunk. If U5 has
not armed after that call, Z processes the same native chunk exactly as v1.2 does and Z's output is executed. From
U5's first arming decision on, U5's output is executed and Z is never called again, including after a U5 disarm.
A zero chunk at the arming decision belongs to U5.

Byte identity: the components are imported read-only from the installed frozen packages. Before any of their code is
imported, every pinned file is sha256-checked at construction: code, card and assets (`glue.COMPONENTS`, which matches
the three freeze manifests). Each component's params must equal its frozen card's params block, so no new scalar can
enter. The params are copied into this package's cards by `scripts/make_cards.py`. For Zsign the glue also checks
that the v1.1 zsign params equal the v1.2 paper_z params, except `zero` and the inert guard fields. Guard is false, so
`hs_band_guard` is never imported.

Diagnostics per decision (`diagnostics.action_op`): `glue` (q, executed `z`|`u5`, arm_decision, z_called,
z_substituted, native_zero, u5_steered, stats), `u5` (tube-steer diagnostics), `z` (tilt-guard diagnostics, or null
once U5 has armed).

Checks: `tests/` (byte identity, no new scalars, refusals, the A6.Z.5 synthetic cases, mutation checks) and
`scripts/replay_round5.py` (CPU replay over the recorded round-5 T arm, streamed one scene at a time). Results and the
freeze manifest are in `runs/rtcfpp-20261101/implementation/u5z/`.

```bash
H=/home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio
uv pip install --python $H/.venv/bin/python --no-deps -e $H/methods/u5z
$H/.venv/bin/hs verify u5z-t --tier 0 && $H/.venv/bin/hs verify u5z-t
$H/.venv/bin/python -m pytest -p no:cacheprovider $H/methods/u5z/tests
```

## The -tv entries are not in this package

`u5z-tv` and `u5zsign-tv` would be used only if A5.3(5) makes U5 = TV (`hs-tube-steer-vfb`). This package does not
define them, so that its frozen hashes never depend on TV's review outcome. The plan, if U5 = TV:

1. Wait until `review-tv-code` accepts TV 1.0.0 (manifest `runs/rtcfpp-20261101/implementation/tube-steer-vfb/manifest.json`)
   and A5.3(5) selects TV.
2. Create a new distribution (proposed `harnessstudio/methods/u5z_tv`, `hs-u5z-tv` 1.0.0; this needs a new editable-path
   assignment). Its op subclasses the frozen `hs_u5z.glue.U5ZActionOp` and sets only `COMPONENTS` (adding
   `tube-steer-vfb` with the files of TV's freeze manifest) and `VARIANTS` (`u5z-tv`, `u5zsign-tv`). It inherits
   `transform` unchanged, and lists hs-u5z 1.0.0 in its Campaign freeze append list.
3. TV 1.0.0 as frozen exposes the same `mode` / `first_arm` API that the glue reads. It needs a SigLIP service and
   `obs['images']['agentview']` whenever the instruction has no success memory (t8), and it calls that service before
   Z on every t8 decision. Tier 1 and the -tv tests must therefore handle the encoder; see the feasibility probe
   `runs/rtcfpp-20261101/implementation/u5z/tv-feasibility.json`.
4. Recompute S_Z's t8 members with the C6 copy of sz.py (A6.Z.3-TV), and have a critic review it before any -tv t8
   episode.
