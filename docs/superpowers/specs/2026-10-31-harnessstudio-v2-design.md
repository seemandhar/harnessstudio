# HarnessStudio v2: design

Status: draft for review (2026-10-31)
Branch: `v2` of `seemandhar/harnessstudio`

## 1. Goal

HarnessStudio v2 is one framework in which any published harness around a frozen robot policy can
be implemented as a method, installed as a package, verified, and compared fairly with the plain
policy. It is meant to grow by one paper at a time, the way nerfstudio grows by method configs.

A *harness* is everything around a frozen policy that is not its weights: what it observes, what
instruction it gets, how many samples are drawn and which one runs, how much of a chunk executes,
when execution is interrupted, which scripted skills run instead of the policy, what memory is
consulted, and which LLM agent plans the episode.

Success for v2 means:

1. A new paper becomes a method by writing one package with an entry point and a paper card, with
   no change to the framework unless the paper needs an interface that does not exist yet.
2. Every method runs on the benchmarks that harness papers report: LIBERO, LIBERO-Pro and
   RoboCasa365 first, then others that allenai's `vla-eval` supports.
3. Every comparison keeps v1's guarantees: frozen hashed artifacts, public inputs only, seeded
   paired arms, a compute-matched control, and per-pair attribution.

### Non-goals

- Training anything: no policy, verifier, critic, adapter, probe or embedding fitting (section 9).
- Real robots and XPolicyLab in v2.0. They come after milestone M4 as a second adapter (section 13).
- Forking `vla-eval`, or changing benchmark code, policy weights or benchmark Docker images.
- CI and test scaffolding beyond the verification tiers, which are product features (section 8).
- A public leaderboard site.

## 2. Decisions

| ID | Decision | Reason |
|---|---|---|
| D1 | Depend on `vla-eval==0.8.0` from PyPI (released 2026-09-30). No fork. | allenai ships LIBERO, LIBERO-Pro, RoboCasa365, Kinetix and ~20 policies, maintained upstream. A fork would have to be rebased forever. |
| D2 | HarnessStudio runs as a **harness server** between the allenai benchmark and the policy server. | No allenai changes; method dependencies stay out of benchmark images; methods can only see what a policy would see; every allenai policy works for black-box methods. |
| D3 | allenai first; XPolicyLab is a second adapter after M4. | The first target papers report on LIBERO, LIBERO-Pro and RoboCasa365, which allenai has and XPolicyLab does not. |
| D4 | A paper is a pip package registered through entry points, with a paper card. | Heavy dependencies (SAM3, LLM clients) stay per paper; external authors can ship methods without touching core. |
| D5 | Methods compose typed components; component families follow the 12 categories in `RoboMetaHarness/knowledge/HARNESS_CATEGORIES.md`. | One vocabulary for HarnessStudio and RoboMetaHarness; components from different papers can be combined. |
| D6 | Runtime LLMs default to NRP-hosted open-weight models; Astra is never used. | Workspace rules in `cvpr27/AGENTS.md`. Runs that substitute the paper's LLM are labeled `adapted`. |

The prototype in `harnessstudio-vla-eval/` (a custom allenai `EpisodeRunner` running inside the
benchmark process) is superseded by D2. It is preserved on a branch in M0, and its
`MethodContext` checks and `prepare`/`run`/`run_null` method shape are reused.

## 3. Architecture

```
allenai benchmark process            HarnessStudio harness server                  policy server(s)
(Docker, unchanged)                  (host, one process per arm)                   (host GPU)
 Benchmark + SyncEpisodeRunner  ◄─ws─►  ModelServer facade                     ◄─ws─►  allenai policy server
   sends observation, wants              per-session EpisodeContext                    unchanged (T0), or a
   one action back                       method coroutine + components                 HarnessStudio subclass
                                         tools: LLM/VLM, perception, skills            (T1-T3, section 4)
                                         budgets, trace, accounting
```

### 3.1 Harness server

The harness server subclasses `vla_eval.model_servers.base.ModelServer` and is started with
`vla_eval.model_servers.serve.serve`. To the benchmark it is an ordinary policy server.

- `on_episode_start(config, ctx)`: builds a fresh method instance and `EpisodeContext` for the
  session. The `task` payload (task name, `episode_idx`, etc.) is kept by the framework for noise
  seeds and trace keys and is **never** passed to the method.
- `on_observation(obs, ctx)`: filters `obs` to the contract's public keys (section 3.3), then
  resumes the method coroutine until it produces the next action, and sends it with
  `ctx.send_action({"actions": a})`.
- `on_episode_end(result, ctx)`: cancels the method coroutine, writes the trace and accounting.
  `result` holds benchmark metrics including success; the framework records it and the method
  never sees it.
- `get_action_spec`, `get_observation_spec`, `get_observation_params`: forwarded from the policy
  server and merged with the method's declared extra observation needs (for example a depth
  camera). Requests go through allenai's normal HELLO mechanism, so they stay public.

Each benchmark session gets its own method instance and context, so allenai's episode sharding
and parallel clients work unchanged.

### 3.2 Coroutine bridge

Methods are written as ordinary async loops:

```python
async def run(self, ctx):
    while not ctx.ended:
        chunk = (await ctx.policy.sample())[0]
        await ctx.execute(chunk)
```

`ctx.execute(actions)` queues actions and suspends the method. Each incoming observation releases
one queued action to the benchmark. When the queue is empty, the bridge resumes the method with the
new observation. Policy calls, LLM calls and tool calls happen while the benchmark waits for its
action (sync mode). The allenai client timeout in the run config must exceed the longest expected
method step; the launcher sets it from the method's declared budget.

### 3.3 Benchmark contracts

A contract per benchmark configuration (`contracts/<benchmark>.yaml`) declares:

- public observation keys (allowlist) and the instruction key;
- camera names and image layout;
- state layout, including where end-effector pose and gripper state live, when present;
- action layout and control mode (for example LIBERO: 7-D delta end-effector + gripper);
- official horizon per task (the harness server does not see `max_steps`; see section 14).

Keys outside the allowlist never reach a method. The framework refuses to start if a contract
lists a forbidden key (`reward`, `success`, `seed`, `episode_idx`, `task_id`, `sim_state`, `arm`).

### 3.4 Error handling

| Event | Behavior |
|---|---|
| Policy server error or protocol mismatch | Episode is **invalid** (excluded from rates and pairing), not a failure. |
| Method raises | Episode is invalid; the trace keeps the exception. Tier 1 must catch this class of bug. |
| Policy, LLM or tool budget exhausted | Method control ends; the remaining steps run the native policy with the original instruction; recorded as `budget_exhausted`. |
| Method stops early (`ctx.stop(reason)`) | Remaining steps run natively, as above. |
| Non-finite or wrongly shaped action | Episode invalid. |
| Live mode | Not supported in v2.0; the harness server refuses live sessions. Async methods (RTC, FreeBar) are a later extension. |

## 4. Policy side

### 4.1 Capability tiers

| Tier | Policy server | Capabilities | Unlocks |
|---|---|---|---|
| T0 | Any allenai policy server, unchanged | `act` (one action or chunk per observation) | agents, instruction and observation ops, monitors, skills |
| T1 | HarnessStudio subclass | full chunk, seeded K samples, isolated noise streams | best-of-N, executors, the derived null |
| T2 | HarnessStudio subclass | denoising-step hooks (`Sampler`), velocity readout | VLS, TraceFlow, ChunkTrust |
| T3 | HarnessStudio subclass | internals: attention, logits, predicted futures | Knowing When to Stop, SCALE, world-action-model methods |

Capabilities are named strings (v1's `Capability` enum, extended). Methods declare what they need;
`hs verify` refuses a method on a policy that lacks a required capability.

### 4.2 Policy extension protocol (T1-T3)

`harnessstudio.policy_ext` is a small mixin for allenai policy servers. It uses allenai's free-form
payloads:

- request: the observation dict plus `"harnessstudio": {"seeds": [...], "chunk": true,
  "sampler": {"name": ..., "params": ...}, "readouts": [...]}`;
- response: `{"proposals": [{"actions": [H, A], "extras": {...}}, ...]}`.

Sample 0 always uses the native noise stream for its seed. Sampler hooks run inside the policy
server, loaded by entry point (`harnessstudio.samplers`) from a frozen copy of the method source
that the launcher puts on the policy server's `PYTHONPATH`, as v1 does.

Capabilities are advertised in the HELLO reply. allenai currently sends `capabilities={}`; until an
upstream PR lands (section 14), the launcher passes the policy server's capability list to the
harness server from the policy config.

### 4.3 Policy servers in v2.0

| Policy | Base allenai server | Tiers | Notes |
|---|---|---|---|
| GR00T N1.5 | `model_servers/robocasa365_groot.py` | T0-T3 | Port v1's seeded sampling, velocity readout (ChunkTrust) and attention readout (`ext/attn`). Check that it loads v1's Human300 checkpoint-120000. |
| π0.5 | openpi server (`model_servers/pi0.py`) | T0-T1 | Port v1's seeding and `--xla_gpu_autotune_level=0` determinism fix; the flow loop is jitted, so no T2 in v2.0. |
| Any other allenai policy | as shipped | T0 | Includes the LeRobot bridge (π0.5, GR00T N1.7, MolmoAct2, FastWAM, VLA-JEPA). |

Determinism requirement for T1+: the same request with the same seeds returns bitwise-identical
actions across server processes. Tier 2 checks it.

## 5. Method side

### 5.1 EpisodeContext

The only object a method or component receives.

| Member | Meaning |
|---|---|
| `observe()` | deep copy of the current public observation |
| `instruction` | original instruction (string) |
| `steps`, `remaining`, `ended` | episode progress against the contract horizon |
| `policy.act(obs=None, instruction=None)` | T0 call |
| `policy.sample(k, instruction=None, sampler=None, readouts=())` | T1+ seeded samples; returns `Proposal(actions, extras)` |
| `policy.capabilities` | the policy's capability set |
| `execute(actions)` | queue one or more actions; returns after they run (section 3.2) |
| `decide(instruction=None)` | run the method's `DecisionLoop` once (section 5.3) |
| `skills[name](**params)` | run a declared skill (section 5.4) |
| `tools.llm(messages)`, `tools.vlm(...)`, `tools[name](...)` | pinned, budgeted operators |
| `memory` | read-only frozen `ExperienceStore` from the method's artifacts, plus `scratch` (per-episode dict) |
| `record(event, **fields)` | append a JSON event to the trace |
| `stop(reason)` | end method control; remaining steps run natively |

### 5.2 Components and categories

Each component is a Protocol in `harnessstudio.components`. Every component has an `identity` dict
(name, version, params) that enters the method fingerprint.

| Category (RoboMetaHarness ID) | Component | Runs in | Interface (summary) |
|---|---|---|---|
| `observation_grounding` | `ObservationOp` | harness | `transform(obs) -> obs` (public keys only) |
| `language_alignment` | `InstructionOp` | harness | `rewrite(instruction, obs, memory) -> str` (v1 `episode_op`, now also per subtask) |
| `proposal_selection` | `Proposer`, `Selector` | harness | `num_samples`; `select(proposals, obs) -> index` |
| `sampler_guidance` | `Sampler` | policy server | `on_step(step, x_t, v_t, ctx) -> v_t or None`; `guide(...)` for differentiable denoisers (v1) |
| `uncertainty_compute` | `Gate` | harness | `budget(obs, history) -> {num_samples, invoke: set[str]}` |
| `execution_monitoring`, `contact_feedback` | `Executor`, `Monitor` | harness | v1 executor (`plan`, `continue_after`, `record_executed`); `check(obs, history) -> continue / interrupt / recover(reason)` |
| `memory_retrieval` | `ExperienceStore`, `ActionOp` | harness | `retrieve(obs, progress) -> items`; `transform(chunk, obs, memory) -> chunk` |
| `skill_composition` | `Skill` | harness | named, parameterized, bounded routine (section 5.4) |
| `agentic_orchestration` | `Controller` | harness | `run(ctx)`: owns the episode |
| `world_model_verification` | `Selector` over `futures` readout | both | needs T3 `predicted_futures` |
| `experience_evolution` | `Preparer` | offline | `prepare(dev) -> frozen artifacts` (section 7) |

A component that fits none of these is the signal to add a new interface (section 12).

### 5.3 DecisionLoop and controllers

The default controller is `DecisionLoop`. One decision runs, in order:

`ObservationOp → InstructionOp → Gate → Proposer (+ Sampler in the policy server) → Selector → ActionOp → Executor, with Monitor checked per executed step`

Simple methods only fill slots of the default loop. Agent methods provide a `Controller` and call
`ctx.decide()` whenever they want the policy to act, so their policy calls can themselves use
components from other papers (for example a planner whose `VLA_ACT` skill uses a medoid selector).

### 5.4 Skills

A skill is a named routine with typed parameters, a step bound and a declared control mode.

- `PolicySkill` (`VLA_ACT`): runs `ctx.decide(instruction=subtask)` until a step bound or a monitor
  stops it.
- Analytic skills (`MOVE_TO`, `MOVE_POSE`, `ROTATE_WRIST`, `SET_GRIPPER`, `RELEASE`, ...): closed-loop
  controllers over the contract's state and action layout. One library per control mode
  (`skills/eef_delta.py` first, for LIBERO and LIBERO-Pro).
- Code skills: programs written by a method (for example HarnessPAI), loaded from frozen artifacts.

Skills read only public observations and tool outputs. Object locations come from perception tools,
never from simulator state.

### 5.5 Tools

- LLM/VLM operator: v1's pinned operator (`core/llm.py`): one model on an OpenAI-compatible endpoint,
  `/models` check on first call, no fallback, key from one named environment variable, requests
  logged by hash and responses in full. Default endpoint NRP. Tier 1 uses scripted responses.
- Perception tools (SAM3, depth, detectors): separate tool server processes with pinned, hashed
  weights, declared in the paper card. Budgeted like LLM calls.

## 6. Method packages

### 6.1 Layout

```
methods/<slug>/
  pyproject.toml     # [project.entry-points."harnessstudio.methods"] <name> = "hs_<slug>:config"
  paper.yaml         # paper card (6.2)
  hs_<slug>/         # config() and components
  README.md          # what is reproduced, what is substituted, how to run
  reference/         # optional: the original implementation's run on our seeds, for `hs reproduce`
```

Every method, including ported v1 methods, lives in `methods/`. Only `identity` is in core.

Entry point groups: `harnessstudio.methods`, `harnessstudio.components`, `harnessstudio.skills`,
`harnessstudio.samplers`, `harnessstudio.tools`. A method can reference another package's component
by entry point name.

### 6.2 Paper card (`paper.yaml`)

| Field | Content |
|---|---|
| `name`, `version` | method name (`[a-z0-9-]+`), version |
| `paper` | arXiv id or `none`; title; venue |
| `categories` | primary and secondary category IDs (5.2) |
| `mechanisms` | novelty IDs from `RoboMetaHarness/knowledge/novelty_catalog.json`, when they exist |
| `fidelity` | `faithful`, `component`, `tf_variant` or `adapted` |
| `substitutions` | required for `tf_variant` and `adapted`; each names the original part and its replacement |
| `requires` | policy capabilities, contract features (for example `eef_pose`), tools |
| `artifacts` | name, path or URL, sha256, license |
| `operators` | LLM/VLM model ids, endpoint env var, `max_calls`, `max_tokens`; original model kept for provenance |
| `budgets` | policy calls per episode, tool calls per episode |
| `null` | `derived`, `shadow` or `custom` (section 8.1) |
| `prepare` | absent, or the development protocol (section 7) |
| `original` | repo URL and commit; reported numbers with benchmark, checkpoint and protocol |

`trainable_parameters` must be 0. Registration rejects cards that break these rules.

## 7. Development stage (`prepare`)

Some papers build memory or evolve programs from rollouts before evaluation (Harness VLA few-shot
memory, HarnessPAI, DynaHarness, FastHarness experience). `hs prepare <method>`:

- runs allenai on a declared **development** case list that must be disjoint from every evaluation
  case list (checked by `(benchmark, task, episode_idx)`);
- may use benchmark outcomes, because it is development;
- writes artifacts that are hashed, pinned in a generated card overlay, and then frozen;
- evaluation refuses artifacts produced on any evaluation case, and never updates memory from
  evaluation outcomes.

Methods with a `prepare` stage report zero-shot and prepared results separately.

## 8. Fairness and verification

### 8.1 Arms

| Arm | What runs |
|---|---|
| baseline | `identity`: native policy, sample 0 stream, original instruction |
| null | pays the candidate's compute, executes natively |
| candidate | the method |

Null rules:

- `derived` (methods that only fill `DecisionLoop` slots): v1's per-slot rules (`core/arms.py`):
  draw K samples but execute sample 0, full native chunk, original instruction, sampler inactive,
  cuts ignored.
- `shadow` (controllers): the controller runs and pays every LLM, tool and policy call, but
  `execute`, `decide` and skills run native policy decisions with the original instruction.
- `custom`: the paper card names a method-specific control and justifies it. Allowed only when
  neither rule above applies.

Common random numbers: the noise seed for decision `d`, sample `i` is
`sha256(namespace, benchmark, task, episode_idx, d, i)`. Paired arms run the same allenai benchmark
config (same seed, same episode list). Each pair is classified M (matched), R (branched by renderer
noise, unattributable) or X (harness leak) by v1's first-divergence attribution. Any X blocks the
method.

Evaluation follows the workspace rule: a small predeclared paired subset with official horizons
first, and the full benchmark only after the complete subset improves over baseline and null.

### 8.2 Verification tiers

| Tier | Command | Cost | Checks |
|---|---|---|---|
| 0 | `hs verify M --tier 0` | CPU, instant | card schema, categories, fidelity and substitutions, artifact hashes and licenses, zero trainable parameters, capability satisfiable by some registered policy |
| 1 | `hs verify M --policy P` | CPU, seconds | synthetic contract fixtures and a fake policy: public inputs only (recording proxy), determinism, budget bounds, null derivation or shadow mode, case-order independence, scripted LLM, plugin closure (imports in a clean child interpreter) |
| 2 | `hs verify M --policy P --tier 2 --gpu G` | 1 GPU, one case | real baseline, null and candidate episodes; null vs baseline must be M (or R, then M on one re-pair); sampler native-equality probe; measured overhead |
| reproduce | `hs reproduce M --reference DIR` | GPUs | pair-by-pair comparison with the original implementation's run on the same cases |
| screen | `hs eval M --cases ... --gpus ...` | GPUs | paired arms, validity, M/R/X, wins/losses, paired p, policy calls, LLM calls and tokens, wall time |

Tier 0 also runs on `pip install` of a method package via `hs list`, which shows failures.

## 9. Training-free rules

From `HARNESS_CATEGORIES.md`, enforced at registration and review:

- The policy and all auxiliary models stay frozen; only pinned, already-available artifacts.
- No fine-tuning, LoRA, RL, or fitting of probes, verifiers, dynamics, embeddings, retrieval heads or PCA.
- Gradients may steer actions or noise through frozen models; they never update parameters.
- Code, prompt and symbolic memory construction and bounded configuration selection are allowed only
  in `prepare` on development cases.
- Everything is frozen before evaluation.

Papers whose full method needs training (RoboMonkey, CoVer-VLA, VLA-Corrector, V-GPS, ...) can
enter only as `component` or `tf_variant` methods that name the replaced part.

## 10. Recording and accounting

- The harness server writes `trace.jsonl` per episode: every sample (input hash, seeds, action
  hashes), execution (action hash), tool and LLM call (hashes, tokens, wall time), method event, and
  stop reason. Rows are keyed by allenai's `eval_id` and episode id so they join with allenai results.
- Accounting per episode: steps, decisions, policy calls, samples, LLM calls and tokens, tool calls,
  policy and tool wall time, termination (`environment`, `horizon`, `stop`, `budget_exhausted`).
- Each run directory stores the frozen method source, paper card, contract, resolved configs,
  `vla-eval` version, policy checkpoint identity and the protocol fingerprint. Runs with different
  fingerprints never pair.

## 11. Repository layout

```
harnessstudio/                 # package name unchanged, version 2.0.0
  core/        method config, paper card schema, registry (entry points), capabilities, arms, noise, budget, errors
  context/     EpisodeContext, PolicyHandle, operators, trace recorder
  components/  protocols, DecisionLoop, reusable components (medoid selector, stasis executor, instruction rewrite)
  server/      harness server (allenai ModelServer facade), coroutine bridge, contract filtering
  policy_ext/  mixin for policy servers: seeded K samples, chunk mode, sampler host, readouts, capability advert
  policies/    uv-script policy servers subclassing allenai servers (gr00t_n15_robocasa365.py, pi05_openpi.py)
  contracts/   libero.yaml, libero_pro.yaml, robocasa365.yaml
  skills/      analytic skill libraries per control mode (M4)
  tools/       perception tool servers (M4)
  eval/        launcher (policy servers, harness servers, `vla-eval run` per arm), result collection, M/R/X, GPU leases and ledger
  verify/      tiers 0-2 and synthetic fixtures
  cli.py       `hs`
methods/       one package per paper
docs/
```

## 12. Reuse and growth

Port from v1 (`main`, `ext/aac`, `ext/aac-ckpt`, `ext/async`, `ext/attn`):

- `core/method.py`, `capabilities.py`, `arms.py`, `noise.py`, `budget.py`, `llm.py`, `controller.py`
  (tools API, shadow mode); `eval/attribution.py`, ledger and lease handling; tier 1 fixture ideas.
- Methods: `medoid-bon`, `chunk-interrupt`, `aac`, `chunktrust-ahs`, `llm-subtask-planner`,
  `subtask-sequencer`, `rtc`; the GR00T attention readout.
- Policy knowledge: GR00T N1.5 seeded sampling and velocity readout; π0.5 seeding and the XLA
  autotune fix.

From the prototype: `MethodContext` input and action checks, fingerprints, `DevelopmentContext`.

Drop: vendored `bench/` runners, `backends/*/worker.py` episode loops, `runtime/` parity guard,
and the custom-runner patch to `vla-eval`.

How v2 grows: one PR per paper. If a paper does not fit, the PR adds or extends a component or
capability and records it in `docs/interfaces.md` (what changed, which paper needed it). Every
existing method must still pass tier 1.

## 13. Roadmap

Each milestone gets its own implementation plan. The first plan covers M0 and M1.

| Milestone | Work | Exit gate |
|---|---|---|
| M0 | Preserve the prototype on branch `prototype/runner-hook` in `harnessstudio-vla-eval`. On `v2`: package skeleton pinned to `vla-eval==0.8.0`, `hs` CLI stub, contract schema and `libero.yaml`. | `pip install -e .` works; `hs list` shows `identity`. |
| M1 | Harness server, coroutine bridge, T0 policy client, `EpisodeContext` (observe, policy.act, execute, record, stop, budgets), trace and accounting, `identity`. | (a) On LIBERO with an allenai π0.5 server, every observation the policy receives and every action the benchmark receives through the harness server are byte-identical to the direct connection (hash logs on both sides, 10 episodes). (b) A 50-episode LIBERO run through the harness server matches the direct run's success rate within its 95% CI. (c) Two parallel shards work. |
| M2 | `policy_ext`; GR00T N1.5 (T1-T3) and π0.5 (T1) servers; RoboCasa365 contract. | v1's exactness results reproduced: `identity` bitwise equal to the native seeded server on one case; sampler no-op probe bitwise equal; null vs baseline 0 X on 8 cases. Try allenai's CPU render backend and report the R rate. |
| M3 | Components, `DecisionLoop`, paper card, registry, tiers 0-2, arms launcher, M/R/X. Port the seven v1 methods as packages. | All ported methods pass tiers 0-1; each passes tier 2 on one case; a re-run of v1's 8-case `medoid-bon` screen gives 0 X and a candidate/null inference ratio near 1.0. |
| M4 | Skills (`eef_delta`), perception tool servers, `ExperienceStore`, `prepare`, `Monitor`, `ObservationOp`, `ActionOp`, controller shadow null; LIBERO-Pro contract. First agent method: Harness VLA zero-shot on LIBERO-Pro with an NRP LLM (`adapted`). | Tiers 0-2 pass. `hs reproduce` against RPent's own run (same NRP model) on a predeclared subset: success rates within CI and per-pair agreement reported. |
| M5 | Paper intake, one PR per paper, one per category first: FastHarness, RoboHarness, RTCF, VLS, Knowing When to Stop, SCALE (allenai OpenVLA server), BYOVLA, Rephrase Before You Act, HarnessPAI, Gated GeoBoN (needs a world-action-model server), RTC/FreeBar (needs live mode). | Per paper: tiers 0-2, then `reproduce` when original code exists, otherwise a subset screen. |
| After M4 | XPolicyLab adapter: harness server as an XPolicyLab `Model` (RoboDojo, RoboTwin, real robots) and an XPolicyLab policy client (T0 for its 69 policies). Real-robot comparison protocol. | Separate spec. |

## 14. Upstream PRs to allenai (optional, not blocking)

1. `ModelServer.get_capabilities()` advertised in HELLO (4 lines; the prototype already has it).
2. `max_steps` included in the `episode_start` payload, so the harness server can read the horizon
   instead of the contract.

## 15. Risks and open questions

- allenai's LIBERO-Pro and RoboCasa365 integrations are marked "awaiting first reproduction". Each
  benchmark needs a baseline parity run (as v1 did: 4/80 through HarnessStudio vs 5/80 native)
  before any method result.
- Renderer nondeterminism branches pairs (R), more often on long tasks. M2 measures whether the CPU
  render backend reduces it.
- Some papers may read simulator state. Each paper is checked; such parts get a perception
  substitute (`tf_variant`) or the paper is excluded.
- Open (decide in M2): does allenai's GR00T N1.5 server load v1's Human300 checkpoint-120000?
- Open (decide in M4): which π0.5 checkpoint for LIBERO-Pro: RLinf `pi05_libero130` (Harness VLA)
  or openpi `pi05_libero` (Pigey). Baselines across papers differ (12.8% to 50.0%), so one is pinned
  per benchmark.
