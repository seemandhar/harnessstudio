# HarnessStudio v2 core (M0-M3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working HarnessStudio v2: a harness server on `vla-eval==0.8.0` that runs paper methods (installed via entry points) around a frozen policy, with paired arms, tier 0/1 verification, a GPU parity gate on LIBERO, and the v1 methods ported as packages.

**Architecture:** See `docs/superpowers/specs/2026-10-31-harnessstudio-v2-design.md`. The harness server is an allenai `ModelServer`; per benchmark session it builds an `EpisodeContext` and runs the method as an anyio task bridged to the benchmark's observation/action loop. The policy is reached through allenai's `Connection` client. Tier 1 runs the full stack in-process (`vla_eval.api.serve_background` + `vla_eval.api.run(..., docker=False, no_save=True)`) with a fake benchmark and a fake policy.

**Tech Stack:** Python 3.11, `vla-eval==0.8.0` (anyio, websockets, msgpack), numpy, PyYAML, pytest.

**Budget:** this plan is executed in one ~12 h session. Code is written during execution rather than duplicated here; each task lists files, interfaces, the tests that pin behavior, and its gate. Tests stay lean (no CI, no mock infrastructure beyond the fake policy/benchmark that tier 1 needs anyway).

---

## File structure (v2 branch, replaces the v1 package tree; v1 stays on `main` and `ext/*`)

```
pyproject.toml                       name=harnessstudio 2.0.0.dev0; deps vla-eval==0.8.0, numpy, pyyaml; script hs
harnessstudio/__init__.py            __version__
harnessstudio/core/errors.py         HarnessError, RegistrationError, ContractError, BudgetExceeded, PolicyError
harnessstudio/core/categories.py     the 12 category IDs (RoboMetaHarness HARNESS_CATEGORIES.md)
harnessstudio/core/capabilities.py   capability names + validate
harnessstudio/core/method.py         Fidelity, ArtifactPin, ModuleRef, Budgets, MethodConfig, validate()
harnessstudio/core/card.py           PaperCard + load_card(paper.yaml)
harnessstudio/core/registry.py       entry-point discovery (group harnessstudio.methods) + built-ins
harnessstudio/core/noise.py          crn_seed()
harnessstudio/core/arms.py           Arm + derived-null rules (ported from v1)
harnessstudio/contracts/__init__.py  Contract, load_contract, FORBIDDEN_KEYS, filter_public
harnessstudio/contracts/*.yaml       fake, libero, libero_pro, robocasa365
harnessstudio/context/trace.py       fingerprint(), Trace (jsonl)
harnessstudio/context/policy.py      Proposal, PolicyHandle (T0 act, T1 sample) over vla_eval Connection
harnessstudio/context/bridge.py      Bridge: method task <-> observation/action loop (anyio streams)
harnessstudio/context/episode.py     EpisodeContext (method-visible API)
harnessstudio/components/protocols.py ObservationOp, InstructionOp, Gate, Proposer, Selector, ActionOp, Executor, Monitor, Controller
harnessstudio/components/decision_loop.py DecisionLoop (default controller, derived null rules)
harnessstudio/components/library.py  reusable components (TrajectoryMedoid, PublicStasis, InstructionRewrite)
harnessstudio/server/harness_server.py HarnessServer(ModelServer)
harnessstudio/policy_ext/mixin.py    HarnessPolicyMixin for policy servers (T1+ protocol)
harnessstudio/policy_ext/fake.py     FakePolicyServer (deterministic CPU policy, T0+T1)
harnessstudio/methods/identity.py    built-in identity
harnessstudio/verify/fixtures.py     FakeBenchmark (allenai StepBenchmark)
harnessstudio/verify/tier0.py        static checks
harnessstudio/verify/tier1.py        in-process arm runs + checks
harnessstudio/eval/launch.py         start harness servers per arm, run vla-eval per arm, collect
harnessstudio/eval/pairing.py        pair episodes, paired stats
harnessstudio/eval/attribution.py    M/R/X (ported from v1)
harnessstudio/cli.py                 hs list | verify | serve | eval
methods/<slug>/                      one package per paper (ported v1 methods in M3)
examples/plugin_method/              minimal external method package
tests/                               test_core.py, test_contracts.py, test_stack.py, test_plugin.py
```

## Task 1 (M0): branch hygiene and skeleton

- [ ] Commit the uncommitted prototype in `harnessstudio-vla-eval` to branch `prototype/runner-hook` (no push).
- [ ] On `v2`: `git rm -r harnessstudio tests examples` (v1 tree), keep `docs/`, `LICENSE`.
- [ ] Write `pyproject.toml` (hatchling; packages `harnessstudio`; package data `contracts/*.yaml`; entry point `hs = harnessstudio.cli:main`; group `harnessstudio.methods` registers `identity`).
- [ ] `uv pip install -e '.[test]'` into `.venv`; `.venv/bin/hs --help` runs.
- [ ] Commit `chore: v2 skeleton on vla-eval 0.8.0`.

## Task 2: core (errors, categories, capabilities, method, card, registry, noise, arms)

Interfaces:
- `Fidelity`: `faithful | component | tf_variant | adapted`; `tf_variant`/`adapted` require `substitutions`, the others forbid them.
- `ModuleRef(factory="pkg.mod:attr", params={...})` JSON-normalized params, `.build()`.
- `Budgets(policy_calls:int, llm_calls:int=0, tool_calls:int=0, max_tokens:int=0)`.
- `MethodConfig(name, version, paper, fidelity, substitutions, categories, required, controller: ModuleRef|None, slots: dict[str, ModuleRef], null: "derived"|"shadow"|"custom", budgets, artifacts=(), operators=(), trainable_parameters=0, description="")`. `validate()` rules: name regex; categories subset of the 12; required capabilities known; slot names known; `null="derived"` only without a controller; `null="shadow"` requires a controller; selector requires proposer `num_samples>=2`; `trainable_parameters==0`.
- `PaperCard` loads `paper.yaml` (fields of spec 6.2) and `MethodConfig.from_card(card, controller=..., slots=...)`.
- `registry.discover() -> dict[name, callable]`, `registry.load(name) -> MethodConfig` (validated).
- `crn_seed(namespace, benchmark, task, episode_idx, decision, sample) -> int` (sha256 → uint32).

Tests (`tests/test_core.py`): fidelity/substitution rules; unknown slot/category/capability rejected; selector without K≥2 rejected; shadow without controller rejected; `crn_seed` stable and distinct across each argument; `registry.load("identity")` works.

Gate: `pytest tests/test_core.py` passes. Commit.

## Task 3: contracts

`Contract(name, public_keys, instruction_key, image_keys, state_keys, action_dim, control_mode, horizons: dict, default_horizon)`. `load_contract(name|path)`; forbidden keys (`reward, success, seed, episode_idx, task_id, sim_state, arm`) rejected at load; `filter_public(obs)` returns a deep copy with only public keys (nested dicts kept whole); `horizon(task_name)`.
Contracts from the allenai benchmark code: LIBERO (`images`, `task_description`, `states`, `controller_states`; 7-D delta EEF), RoboCasa365 (`images`, `state`, `task_description`), LIBERO-Pro (same as LIBERO), fake.

Tests (`tests/test_contracts.py`): forbidden key rejected; filter drops undeclared keys and copies arrays; horizon fallback.

Gate: tests pass. Commit.

## Task 4: fake policy + fake benchmark (tier-1 substrate)

- `FakePolicyServer(PredictModelServer-free ModelServer)`: deterministic. T0: action = f(sha256(public obs)) of `action_dim`. T1 (payload key `harnessstudio`): returns `proposals` with `chunk_len` actions per seed, action = f(obs hash, seed); sample for the native seed equals the T0 action stream. Advertises capabilities via a constructor list (HELLO capabilities are empty in 0.8.0; the harness server takes them from config).
- `FakeBenchmark(StepBenchmark)`: N tasks, deterministic obs from (task, episode_idx, step, last action), success iff sum of actions crosses a threshold, horizon from contract `fake`.
- Both run in-process: `serve_background(FakePolicyServer(...))`, `vla_eval.api.run(cfg, server_url=..., docker=False, no_save=True)`.

Test: direct run (benchmark → fake policy) completes and returns per-episode results.

## Task 5: trace, policy handle, bridge, episode context

- `PolicyHandle(url, capabilities, timeout)`: `connect()`, `start_episode(payload)`, `act(obs) -> np.ndarray`, `sample(obs, seeds, sampler=None, readouts=()) -> list[Proposal]`, `end_episode(result)`, `close()`. Counters: calls, samples, seconds.
- `Bridge`: `start(tg, method_coro_factory)`; `async step(obs) -> action | None` (None = method finished/stopped: caller falls back to native); `async emit(action) -> obs | None` used by `EpisodeContext.execute`; `cancel()`.
- `EpisodeContext` (spec 5.1): `observe, instruction, steps, remaining, ended, policy (proxy with budget), execute, decide, record, stop, memory, tools`. Private: arm, case key, contract, trace. Budget exhaustion raises inside the method, ends control, native fallback.

Tests come through Task 6's stack tests.

## Task 6: harness server + identity + DecisionLoop

- `HarnessServer(method, policy_url, contract, arm, out_dir, namespace, capabilities, timeout)`:
  - `on_serve_start(tg)`: probe policy HELLO, cache `action_spec`, `observation_spec`, `observation_params`; keep `tg`.
  - `on_episode_start(payload, ctx)`: open a `PolicyHandle` per session (cached across episodes of the session), forward the payload unchanged to the policy, build context + bridge, start the method task.
  - `on_observation(obs, ctx)`: public-filter, `bridge.step(obs)`; on None run native (`policy.act(obs)` with the original observation). Send `{"actions": a}`.
  - `on_episode_end(result, ctx)`: cancel method, forward end to policy, write `accounting` + `result` into the trace.
- `identity`: controller-less, no slots → `DecisionLoop` that calls `policy.act` and executes the returned action(s).
- `DecisionLoop.decide()` per spec 5.3 with derived-null rules from `core/arms.py` (null/baseline: sample 0, native chunk, original instruction, sampler off, cuts ignored; baseline draws 1 sample).

Tests (`tests/test_stack.py`, all in-process):
1. passthrough: identity through harness gives byte-identical actions and identical per-episode results to the direct run (compare trace action hashes and episode results).
2. the policy never receives undeclared keys (fake policy records the keys it saw).
3. derived null: a test method with proposer K=4 + selector "pick last": null-arm actions == baseline-arm actions; candidate differs; null policy samples == candidate samples.
4. budget: a method with `policy_calls=3` falls back to native after 3 calls and the episode completes, `termination=budget_exhausted`.
5. shadow: a test controller that executes zero actions `[0,...]` for 2 steps then decides — null arm executes native actions (== baseline) while the controller's calls are still counted.
6. method exception → the episode is reported as an error by allenai (not success/failure).

Gate: all pass. Commit.

## Task 7: verification tiers + CLI

- `tier0.check(config)`: validate, artifacts exist + sha256 + license, required capabilities satisfiable by a registered policy profile.
- `tier1.run(config, policy_profile)`: runs baseline/null/candidate on the fake stack (2 tasks × 2 episodes) and checks: public keys only, determinism (two candidate runs identical), null == baseline (derived) or shadow parity, budget bound respected, case-order independence (episode order shuffled → same per-case actions).
- CLI: `hs list [--category C]`, `hs verify M [--tier 0|1] [--policy fake|gr00t-n1.5|pi0.5]`, `hs serve M --policy-url URL --contract C --arm A --port P --out DIR`, `hs eval M --config BENCH.yaml --policy-url URL --contract C --arms baseline,null,candidate --out DIR`.

Gate: `hs verify identity` passes tiers 0-1. Commit.

## Task 8: plugin path

`examples/plugin_method/` (pyproject with `harnessstudio.methods` entry point, `paper.yaml`, a K=2 first-sample method). Test: after `pip install -e examples/plugin_method`, `hs list` shows it and `hs verify first-sample` passes tier 1.

## Task 9 (M1 gate, GPU): LIBERO parity through the harness server

Uses the direct stack prepared in `/home/sejain/research/cvpr27/hs2-gpu/README.md` (π0.5 LIBERO server on GPU 4, port 8101).
- [ ] Start `hs serve identity --policy-url ws://localhost:8101 --contract libero --arm baseline --port 8201`.
- [ ] Run the same LIBERO config against `ws://localhost:8201` and against `ws://localhost:8101` directly, 10 episodes each, same seed.
- [ ] Byte-identity check: harness trace observation hashes and action hashes vs a direct-run trace. The direct run gets traced by running the same harness server in `--record-only` passthrough... (simpler: compare the policy-side inputs by enabling allenai recording in both runs and diffing the recorded actions per step).
- [ ] 50-episode success rate through the harness within the direct run's 95% CI; two shards in parallel.

Gate (spec M1 a-c). Record results in `docs/results/m1_libero_parity.md`. Commit.

## Task 10 (M2 start, GPU): policy_ext + GR00T N1.5

- `HarnessPolicyMixin` for allenai `PredictModelServer` subclasses: requests with `harnessstudio` key return `proposals` (full chunks, one per seed); otherwise native behavior. Subclass hook `sample_chunks(obs, seeds, sampler, readouts, ctx) -> list[(actions, extras)]`.
- `harnessstudio/policies/gr00t_n15_robocasa365.py`: subclass of allenai `robocasa365_groot` server with seeded initial noise (port v1 `backends/gr00t_robocasa365` seeding).
- Gate: same seeds → bitwise-identical chunks across two server processes; sample 0 equals the native action stream for that seed.

## Task 11 (M3): port v1 methods as packages

`methods/medoid_bon`, `methods/chunk_interrupt`, `methods/aac`, `methods/llm_subtask_planner`, `methods/subtask_sequencer` (+ `chunktrust_ahs`, `rtc` when their policy-side hooks exist). Each: `paper.yaml`, config, components; tier 0-1 pass on the fake policy.

## Execution order and parallelism

Tasks 1-8 sequential in the main session (tightly coupled). In parallel: a background agent prepares the direct LIBERO + π0.5 stack (feeds Task 9). After Task 6 lands, Task 11 ports can go to subagents (one per method), and Task 10 to a subagent.
