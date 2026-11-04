# EXT openvla-residual-hook: decoder-layer residual hooks on the OpenVLA SimplerEnv server + SimplerEnv official grids with proprioception

Status: BUILT on branch `ext2/openvla-residual-hook` (worktree
`/home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio-wt/ext2/openvla-residual-hook`, created from main
`f7dcd0fa0ab20147fa9e96b29e8417d50bbb5c62`), commits listed in section 9. NOT merged: main stays untouched until the
PCD discovery run's PROGRESS.md says RUN CLOSED and `scripts/regress.py --inflight` is empty. Run star-port-20261104,
task conv-ext-2, author rmh_converter (agent ad6f985cab099bc95). Proposal review: critic-ext-1 APPROVE_WITH_CONDITIONS
(`runs/star-port-20261104/reviews/critic-ext-1.md`, `.review.json`); this document carries R1-R13 into the build. The
same text is committed as `docs/extensions/openvla-residual-hook.md` on the branch.

## 1. Paper mechanism and why no existing hook expresses it

STAR (arXiv:2610.41009v1; released code PKU-VaLuE-Lab/STAR@0eaa15f, pi0.5 only) replaces, in the last third of a VLA's
decoder layers, each layer's residual update by an EMA over depth, `m^k = beta_t m^{k-1} + (1-beta_t) Delta h^k`,
`h^k <- h^{k-1} + m^k`, with one beta per control step from a causal proprioception scheduler. It rewrites hidden
states INSIDE the forward pass; only the policy server holds the model. Existing hooks cannot do this:
`decoding_hooks` (`openvla_hs.py`) selects tokens and sets the ViT attention temperature on a template-prompted path;
`logits_readout` only reads; the `simpler_pcd_authors` mode refuses samplers (`openvla_simpler_hs.py:172`, `:221-222`).
A method-side monkeypatch is forbidden. The merged SIMPLER benchmark has no official grid resets, forces
`send_state=False` and refuses Place in Drawer.

## 2. The change (smallest that works)

New files (every merged parent stays byte-identical, R6):

| file | role |
|---|---|
| `harnessstudio/policy_ext/residual_hooks.py` | `residual_hooks(layers, mixer, record)`: request-scoped forward pre/post hooks, removed in `finally` (`:50-92`, `:89`); `ForwardRecord` shape counters (`:28-41`) |
| `harnessstudio/policies/openvla_simpler_star_hs.py` | `HarnessOpenVLASimplerStarServer(HarnessOpenVLASimplerServer)`, capability `residual_hooks` |
| `harnessstudio/benchmarks/simpler_vm.py` | `SimplerVMBenchmark(SimplerSeededBenchmark)`: official VM + VA-base grids, `states`, OpenVLA post-processing, two-stage Place in Drawer |
| `harnessstudio/contracts/simpler_vm.yaml` | contract (public `images`, `task_description`, `states`; official horizons; `default_horizon: 80`) |
| `harnessstudio/profiles/openvla-simpler-star.yaml` | server profile (parent's capabilities + `residual_hooks`) |
| `tests/test_ext_openvla_residual_hook.py`, `tests/toys/hs_ext_residual_toy.py` | CPU tests and a toy hook |
| `docs/extensions/openvla-residual-hook.md` | this document |

Edits of existing files:

| file | edit |
|---|---|
| `harnessstudio/core/capabilities.py` | one entry `residual_hooks` + the T2 protocol in the docstring, distinct from `decoding_hooks` (`:4-14`, `:29`) |
| `harnessstudio/core/method.py` | `validate`: a `sampler` slot is also legal with `residual_hooks` (`:205-208`); found by tier 0 (`RegistrationError: A sampler needs ...`) |
| `harnessstudio/profiles/fake-t3.yaml` | `residual_hooks` added (R5 route a, `:3-7`) |
| `tests/test_openvla_logits.py` | `test_new_profiles` pins `fake-t3 - fake-t2 == {logits_readout}`; amended to exclude only this EXT's `residual_hooks` (section 6, flagged for the diff review) |
| `docs/extensions/README.md`, `docs/writing-a-method.md` | backlog line; a "Decoder-layer residual hooks" paragraph |

### 2.1 Server protocol (capability `residual_hooks`)

Request `sampler: {factory, params}`, factory module under `sampler_prefixes` (default `hs_`). The hook object:
`READS` (observation keys it may read; must be a subset of `HOOK_READABLE = {"states"}`), `bind(total_layers)`,
`init_state()` (per (session, episode, sampler key); dropped at EPISODE_START/END by the inherited `_drop_states`),
`begin(state, obs_view) -> (mixer, info)`, `next_state(state, info)`. Mixer: `reset()` at layer 0 of every forward,
`step(h_in, h_out) -> h` per layer, optional `correction_start_layer`, optional `step_context()`.
Sampler None: the parent's `sample_chunks`/`predict` unchanged (baseline, derived null, any native client).

## 3. Review conditions R1-R7 (in proposal and build)

- **R1 activation evidence.** Extras never reach the trace (`context/policy.py:108-113`), so every hooked request
  writes one record to the server's pre-cast action log, path `residual_hook`
  (`openvla_simpler_star_hs.py:180-183`, writer `:129-134`): `seq`, `session`, `episode`, `tokens`, `action_f64`,
  `action_f64_hex`, `seeds` (the CRN seeds of the request: the join key to the hs trace's `sample` event,
  `context/episode.py:141-142`), `input_sha256` (framework `fingerprint` of the observation as received; advisory join
  key, equal to the harness's `fingerprint(payload)` only if the wire preserves the dict exactly), `hook` (factory +
  params), `info` (the hook's per-request values; STAR: beta_t, r_t, r_t_tilde, c/v/j/w, w_t_f64, request,
  correction_start_layer, shadow) and `shape` (R4 counts). `action_log` is REQUIRED (`:75-78`). Fail-closed rule for
  the evaluator (written here, enforced in the analysis script, not in the server): a candidate decision without a
  joined `residual_hook` record makes the episode invalid (P-20261103-14). Option "declare a readout" was not chosen
  (it would change `context/policy.py`'s readout surface for one method).
  Test: `test_hooked_request_record_and_shape`.
- **R2 structural input boundary.** The server hands `begin` a NEW dict with only `READS` keys (`_view`, `:100-102`),
  refuses a hook whose `READS` leaves `HOOK_READABLE` (`:93-95`). Tests: `test_r2_view_has_only_declared_keys` (the
  toy hook receives exactly `["states"]`; a hook declaring `task_name` is refused); method side
  `method/tests/test_server_hook.py::test_hook_reads_only_states` now audits EVERY use of the name `obs` in
  `hs_star/server_hook.py` (only `obs[STATE_KEY]` / `STATE_KEY in obs`), plus `StarResidualHook.READS == ("states",)`.
- **R3 concurrency.** Factory load, state lookup, `begin`, hook registration, the decode, removal, the record and
  `next_state` all run under `self._infer_lock` (`:163-185`); the parent's native path takes the same RLock. Tests:
  `test_r3_exception_leaves_no_hooks_and_native_thread_stays_native` (a native thread interleaved with hooked requests
  stays bitwise equal to its reference; a `generate` exception leaves no forward hooks or pre-hooks on any layer).
- **R4 runtime shape assertion.** `_check_shape` (`:105-127`): `action_dim` forwards; forward 1 longer than one token,
  forwards 2..N one token; every forward calls every layer; mixed calls = layers from `correction_start_layer`; one
  step context for all forwards; `len(seeds) == 1` (`:161-162`). Violation -> `ResidualHookShapeError`
  (infrastructure) before any action or record. Tests: `test_hooked_request_record_and_shape` (7 forwards, `[T>1, 1 x
  6]`, 32 layer calls, 11 mixed, start 22, through Prismatic-shaped `generate` around a real 32-layer Llama),
  `test_r4_shape_violation_raises` (a decode with 14 forwards is refused, no record), `test_refusals`.
- **R5 minimal edits / tier-1 route.** `policy_ext/fake.py` untouched (pinned). Route (a): `residual_hooks` added to
  `fake-t3.yaml`; `tier1.default_profile` is first-match over fake-t0..t3, so no installed method's tier-1 profile
  changes (`test_tier1_route_a_star_like_method_gets_fake_t3`, and the existing
  `test_tier1_profile_choice_unchanged_for_installed_methods` passes). One existing assertion had to change: section 6.
- **R6 byte-identity and scope.** `test_r6_parents_byte_identical` pins sha256 at f7dcd0f of `openvla_hs.py`,
  `openvla_logits_hs.py`, `openvla_simpler_hs.py`, `benchmarks/simpler_seeded.py`, `contracts/simpler_seeded.yaml`,
  `profiles/openvla-simpler-logits.yaml`, `policy_ext/{fake,protocol,mixin}.py`, `context/*` (and that `context/` has
  no new file). `test_branch_touches_only_its_files` is branch-scoped (P-20261103-11 form: applies only on
  `ext2/openvla-residual-hook` or detached at its commits while HEAD is not in main; diffs against merge-base with
  main; counts untracked files too) and lists exactly this EXT's files.
- **R7 discovery safety.** Worktree created from the explicit commit (`git worktree add ... -b ext2/openvla-residual-hook
  f7dcd0fa0ab2...`); every git command on the main checkout and on vla-eval ran as `git --no-optional-locks`, read-only
  (`ext/snapshot_main.sh`); regress with `PYTHONDONTWRITEBYTECODE=1`; nothing installed into `harnessstudio/.venv`;
  tier 0/1 of `star` / `star-shadow` from the run-local venv `runs/star-port-20261104/venvs/hs-wt-verify` (vla-eval
  installed from a COPY `venvs/src-copies/vla-eval-v0.8.0`, the worktree and `method/` editable); snapshots of main
  `rev-parse HEAD`, `rev-parse HEAD:harnessstudio`, `status --porcelain` (whole tree and `-- harnessstudio tests`) and
  vla-eval `status --porcelain` before/after worktree creation and around each regress (`ext/snapshots/*.txt`, all
  equal; section 8). Nothing was written into the main tree.

## 4. Conditions before GPU checks / scored runs (R8-R13)

- **R8 shadow arm.** `star-shadow` (`hs_star:config_shadow`, sampler params `shadow=True`) computes beta, Delta and m in
  every mixed layer and returns each layer's own output object (`hs_star/star_core.py` [c6]). CPU: bitwise native
  through the server (`test_shadow_toy_is_native_bitwise`) and through real `generate` (parity O5 16/16). Card
  `paper_shadow.yaml`: description "overhead control: STAR computed, nothing written back ...", `reported: "none: control
  arm"`. **N2 plan addition:** on both smoke episodes, star-shadow vs baseline per-step tokens and float64 action-log
  bytes equal; latency ratios STAR/baseline, shadow/baseline, STAR/shadow from per-request server wall time (the
  evaluator adds request timing around `sample_chunks` in its harness log; the server records none).
- **R9 SimplerDecodeShort (supervisor ruling 2026-11-04).** A short decode is policy behaviour in every arm: that
  episode is a completed failure (no reroll), counted and reported per arm; only crashes/transport errors are
  infrastructure. The server keeps the parent's behaviour (raise) and first writes a `decode_short` record (`path`,
  `seeds`, `session`, `episode`, `input_sha256`, `error`) on every path: native `predict` (`:145-150`), seeded native
  (`:154-158`) and hooked (`:173-176`). Test `test_r9_decode_short_record_then_raise`. The evaluator's analysis turns
  the resulting episode error into a failure using these records (by session/episode).
- **R10 rotation distance (supervisor ruling).** The released float32 geodesic `2*acos|<q1,q2>|`
  (`scheduler.py:240-253`) stays in beta. `info.w_t_f64` logs the same angle in float64 as
  `2*atan2(|q2 - <q1,q2> q1|, |<q1,q2>|)` and never enters beta (`hs_star/server_hook.py` `_geodesic_f64`,
  `begin`). Disclosed floor: about 1e-3 rad per step (critic-ext-1: 100% rel. error at 1e-4-3e-4 rad, 2.3% median at
  1e-3-3e-3, 0.2% at 1e-2). Test `test_w_t_f64_logged_and_not_used` (3.5e-4 rad: f64 exact to 1e-6 relative, f32
  off by > 10%).
- **R11 substitutions.** `method/hs_star/paper.yaml` `substitutions` now carries the SETTING.md RC6 VM label verbatim,
  the VA label, and items for: final-step primary metric, VM assumed, Close Drawer excluded (no VM 5-task average),
  Move Near v0 (VA: MoveNearGoogleInScene-v0), one deterministic run per case vs 8-run means (U16), no state-history
  ingestion, U17, JAX (README) vs torch backend, bf16 EMA, the R10 choice, the R9 ruling, the float32 wire.
- **R12 parity reporting.** History of the JAX row of `parity/check_parity.py`: r1 FAILED its predeclared absolute
  bound 1e-6 (2.86e-6, 404/1152 non-bitwise, free-running streams; `parity_result.r1.json`); r2 FAILED a 1e-5 relative
  bound on near-zero entries (`parity_result.r2.json`; its 381/1152 is not a final figure); r3 gates on "authors' torch
  controller == ours bitwise" and reports JAX descriptively: **425/1170 non-bitwise, max |diff| 9.5e-7**
  (`parity_result.json`, sha 42dbd6c5...); its `max_ulps` value is a near-zero artifact and is not reported. critic-ext-1
  check 1 (float64, `jax_enable_x64`): JAX vs ours max 1.8e-15, authors' torch vs ours 0. Wording used everywhere:
  "bitwise equal to the authors' torch controller; equal to their JAX path up to fp32 rounding". The bound files are not
  rewritten; the re-run on the 0.2.0 method files is `parity/parity_result.v2.json` from `parity/check_parity.v2.py`
  (adds J64 = the float64 check: 708 layer steps, JAX vs ours max 1.78e-15, authors' torch vs ours 0 mismatches).
  `port_design.md:128` (and the PROGRESS 04:00 line) quoting 381/1152 are superseded by this paragraph.
- **R13 benchmark and tests.**
  - (a) The FakeEnv models the real advance (`_elapsed_steps >= 100` before the step); the place instruction is first
    observed after step 101 (`seen[100]`), and the gripper state resets once, inside the 101st step
    (`test_place_in_drawer_switch_and_horizon`, VM and VA). The reset probe in the pinned image must record the actual
    switch index (evaluator, N0).
  - (b) `contracts/simpler_vm.yaml` `default_horizon: 80`.
  - (c) Every episode result carries `steps` and `horizon_ok` (steps == official horizon 80/80/113/200); an episode with
    `horizon_ok: false` is invalid (truncated by a config `max_steps`); eval configs must not set `max_steps`
    (`test_horizon_ok_false_when_cut_short`).
  - (d) Reset probe plan (evaluator, before GPU episodes): obs0 RGB + `states` hashes for each of the 756 VM and 216 VA
    cases (distinctness) and for 3 cases in two fresh processes (repeatability); overlay files and `agent.eef_pos`
    present in the pinned image.
  - (e) VA base grids built in this EXT (SETTING.md rev 2 section 3.2): `va_open_drawer` 27, `va_close_drawer` 27
    (allowed; VM Close Drawer stays refused), `va_place_in_drawer` 27, `va_move_near` 60, `va_pick_coke_can` 75, from
    the "# base setup" blocks of `rt1_*_variant_agg.sh` (no overlay, no urdf kwarg; ray tracing for drawers/place;
    `model_ids=apple`; Move Near `MoveNearGoogleInScene-v0`). Tests `test_va_base_values_match_scripts`,
    `test_va_has_no_overlay`.
  - (f) `positions="last"` through real `generate` (critic's F1-F5) = `parity/check_openvla_hooks.v2.py` O6 (8 cases:
    F1/F2/F4/F5 0 bad, F3 rel fp32 6.0e-8, bf16 3.9e-3); the float64 JAX check = J64 above. Both are in the run-local
    parity harness (they need `hs_star` and jax, which framework tests must not import).

## 5. Compatibility claim

- STAR off is the parent server: sampler None calls `super()` (`test_star_off_is_parent_bitwise`); hooks exist only
  inside the `with` block; `openvla_hs.py`, `openvla_logits_hs.py`, `openvla_simpler_hs.py` byte-identical (R6).
- The new benchmark is a new class/file; `simpler_seeded.py` and its contract byte-identical.
- Every installed method's tier 0/1 result is unchanged: regress before (worktree detached at f7dcd0f) vs after (branch)
  differ in no method (section 7); the core edits only ADD a capability name and widen the sampler rule by it.

## 6. The one amended existing test (flag for the diff review)

`tests/test_openvla_logits.py::test_new_profiles` asserted `set(fake-t3) - set(fake-t2) == {"logits_readout"}`. R5 route
(a) adds `residual_hooks` to fake-t3, so the assertion is amended to
`set(fake-t3) - set(fake-t2) - {"residual_hooks"} == {"logits_readout"}` plus `"residual_hooks" in fake-t3`: the original
property (fake-t3 adds only `logits_readout` over fake-t2 for every pre-existing capability) is kept, nothing is
weakened for existing capabilities. Route (b) (a new `fake-t4`) would instead amend `_expected_new_profile` and
`tier1.py`; (a) is the smaller change.

## 7. Regression gate

`python3 harnessstudio/scripts/regress.py --source <worktree> --out ext/regress_report.json` with
`PYTHONDONTWRITEBYTECODE=1` (shared venv; `star`/`star-shadow` are not installed there and are covered by the run-local
tier 0/1, `ext/verify_tier01.json`). Before/after comparison and the final result: `ext/regress_report.json`,
`ext/regress_before_f7dcd0f.json`, `ext/regress_compare.json`.

## 8. Evidence files (run folder)

`ext/snapshots/0*.txt` (R7), `ext/logs/*` (test, verify, regress logs), `ext/verify_tier01.json`, `ext/regress_*.json`,
`parity/parity_result.v2.json`, `parity/openvla_hooks_result.v2.json`.

## 9. Commits on `ext2/openvla-residual-hook`

Listed in `reports/conv-ext-2.report.json` (`git log f7dcd0f..ext2/openvla-residual-hook`).
