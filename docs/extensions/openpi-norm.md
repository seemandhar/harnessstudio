# EXT openpi-norm (2026-11-01): explicit openpi normalization mode on the openpi policy server

Author: agent ac5461833992ed7fd (recovery engineer, run rtcf-20261031, task ext2-openpi-norm). Branch `ext2/openpi-norm`.

**Process disclosure.** The supervisor compressed Loop 2 because the change is small and time-critical. The proposal
and the build were written in one pass. The independent critic reviews proposal, diff and regress report together
**before merge**. The proposal did not get a separate review before the build. Nothing is merged.

## Need
- **Mechanism.** openpi wraps a frozen model with Normalize (state) and Unnormalize (actions). The `DataConfig.use_quantile_norm`
  flag chooses z-score (mean/std) or quantile (q01/q99).
- **What changed upstream.** openpi commits a3d8ea3 and 540ebe8 (2025-07-05) set it to `model_type != PI0` for every config.
  Before that, `LeRobotLiberoDataConfig` never set the flag, so the DataConfig default False applied (z-score).
- **Scope (critic D1).** The claim is verified for **one checkpoint only**, `pi0_fast_libero`, from three pieces of
  evidence: the GCS creation date (2025-05-28, before the switch), the openpi config history, and the demo-frame
  behaviour test. Other openpi checkpoints from before 2025-07-05, on LeRobot*DataConfig or otherwise, *may* need
  `zscore`. Verify each one: config history at its training date, plus a FAST token-grid or demo-frame check.
  `config` stays the default.
- **Why it matters.** `gs://openpi-assets/checkpoints/pi0_fast_libero` was created 2025-05-28 and was trained with z-score.
  Our pinned openpi rev 981483d serves it with quantile norm. pi0-FAST then emits position commands up to ±2.2 and
  fails LIBERO-Long 0/10. Under z-score, its outputs on demo frames track the demo actions: position MAE drops from
  0.58 to 0.06–0.11. Evidence: `runs/rtcf-20261031/repair/diagnosis.md`.
- **Why existing pieces cannot express it.** The server builds the openpi policy inside `Pi0ModelServer.__init__` from
  `config_name`. No server argument, profile or contract reaches `use_quantile_norm`. A method must not monkeypatch the
  framework, and a run-local wrapper (`repair/server/pi0fast_zscore_server.py`) leaves the choice unrecorded in HS
  manifests.

## Change (smallest that does it)
`harnessstudio/policies/pi05_openpi.py`:
- New `HarnessPi0Server(norm_mode="config"|"zscore"|"quantile")`. CLI: `--args.norm_mode=...`.
  - `config` is the default. It is the old code path, byte for byte: it calls the same `super().__init__` and returns
    the same action spec.
  - `zscore` and `quantile` route openpi's `get_config` through `apply_norm_mode` only while the policy is built (a
    context manager that restores the original function, including on error). The config's data factory is wrapped in
    `NormOverrideDataFactory`, whose `create()` returns the inner DataConfig with only `use_quantile_norm` replaced.
  - Weights, `norm_stats.json`, transform order, tokenizers, prompt, image handling and decoding are untouched.
- **Identity.** A forced mode is appended to the HELLO action-spec description (`actions: "...; openpi norm_mode=zscore"`).
  - HarnessStudio's harness server stores the HELLO as `policy_hello` in every run manifest, so traces record the mode.
  - vla-eval's `check_specs` ignores descriptions, so compatibility is unchanged.
  - The server also logs the mode at start. `/health` belongs to vla-eval (`{"status": "ok"}`) and is not changed;
    services.json should record the launch args.
- New `harnessstudio/policies/configs/pi0_fast_libero_zscore.yaml`: vla-eval `libero_fast.yaml` args plus `norm_mode: zscore`.

## Compatibility claim and evidence
- **Default (`config`).** The old path: same constructor call, same action spec.
  `tests/test_openpi_norm.py::test_default_mode_equals_plain_vla_eval_server` runs on CPU with only the 3B model replaced
  by a recording stub; all transforms, tokenizers and norm stats are real. It checks pi05_libero and pi0_fast_libero:
  - flags: Normalize and Unnormalize both quantile;
  - input and output transform lists are identical to a plain vla-eval `Pi0ModelServer`;
  - actions are bitwise equal on a fixed observation.
- **`zscore`.**
  - **Matches an independent re-derivation.** `test_zscore_mode_equals_openpi_release_normalization` checks the server
    output against a re-derivation from `norm_stats.json`: FAST decode × std + mean, plus state on dims 0–5, within
    1e-5. It also checks the tokenized-prompt state tokens bitwise.
  - **Matches the run-local wrapper.** `runs/rtcf-20261031/repair/ext2/wrapper_equivalence.json` compares against the
    reviewed run-local wrapper on 3 fixed observation/chunk pairs: actions and prompt tokens are bitwise equal, the
    maximum difference is 0.0, and z-score differs from the default.
- **Negative controls.**
  - `quantile` equals the default for pi0_fast_libero; `zscore` differs in both actions and state tokens.
  - An unknown mode is refused before loading.
  - The context manager restores `get_config` even on failure.
  - Sabotage check: making the factory ignore the mode fails 4 of the 12 tests
    (`runs/rtcf-20261031/repair/ext2/sabotage_run.txt`).
- **Unchanged framework.** Core, runtime, contracts, profiles and methods are untouched. `scripts/regress.py --source`
  this worktree: see `regress.json` (result recorded in `runs/rtcf-20261031/repair/ext2-report.md`).
- **Not yet done (evaluator, GPU).** A 2-case native-equivalence check on the worktree code:
  - pi05_libero `config` must stay bitwise equal to the live server;
  - pi0_fast_libero `zscore` must be bitwise equal to the run-local wrapper server on a probe observation.
  Commands are in the ext2 report.

## Merge rule
Merge only after critic review, with `scripts/regress.py --inflight` empty. Running servers import the live
`pi05_openpi.py`, so restart them only between runs.
- Existing runs and frozen manifests keep their behaviour, because the default is unchanged.
- pi0-FAST LIBERO runs must relaunch with `norm_mode: zscore` under a new protocol fingerprint. Their old records stay
  as audit evidence.
