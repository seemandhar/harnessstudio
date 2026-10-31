<h1 align="center">HarnessStudio</h1>

<p align="center">
  <b>Training-free harnesses around frozen robot policies, as installable methods.</b><br>
  One paper = one package. Verify it in seconds, then run it against matched controls on real benchmarks.
</p>

A *harness* is everything around a frozen robot policy that is not its weights: what it observes, which
instruction it gets, how many samples are drawn and which one runs, how much of a chunk executes, when
execution is interrupted, which scripted skills run instead of the policy, and which LLM agent plans the
episode. HarnessStudio turns each published harness into a **method package** that plugs into typed
slots around a frozen policy, the way nerfstudio turns NeRF variants into method configs.

v2 runs on [allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)
(`vla-eval==0.8.0`, a pip dependency, not a fork), so every method runs on its benchmarks (LIBERO,
LIBERO-Pro, RoboCasa365, ...) and with its policy servers (π0.5, GR00T, ...).

```
vla-eval benchmark (Docker, unchanged)  ◄─ws─►  HarnessStudio harness server  ◄─ws─►  policy server
 thinks it talks to a policy                    method + components + tools          frozen π0.5 / GR00T / ...
                                                budgets, traces, paired arms
```

## Install

```bash
git clone -b v2 https://github.com/seemandhar/harnessstudio && cd harnessstudio
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/hs list
```

## Quick start

```bash
hs list                                   # installed methods, categories, required capabilities
hs verify identity --tier 0               # static checks
hs verify medoid-bon                      # tier 1: baseline/null/candidate through real vla-eval on a fake stack (CPU, seconds)

# real benchmark: run one arm's harness server in front of any vla-eval policy server
hs serve identity --policy-url ws://localhost:8101 --contract libero --profile pi05-libero-stock --port 8201
vla-eval run --config my_libero.yaml      # with server.url: ws://localhost:8201

# or let hs run all three arms and pair them
hs eval medoid-bon --config my_libero.yaml --policy-url ws://localhost:8101 --contract libero \
   --profile pi05-libero-hs --out runs/medoid-bon
```

## Methods are packages

```
methods/<slug>/
  pyproject.toml          # [project.entry-points."harnessstudio.methods"] <name> = "hs_<slug>:config"
  hs_<slug>/paper.yaml    # paper card: categories, fidelity, substitutions, capabilities, artifacts, LLM operators
  hs_<slug>/__init__.py   # components + config()
```

`pip install -e methods/<slug>` and it shows up in `hs list`. Full guide: [docs/writing-a-method.md](docs/writing-a-method.md).

**Slot methods** fill the default decision loop:
`observation_op → instruction_op → gate → proposer (+ sampler in the policy server) → selector → action_op → executor (+ monitor)`.
**Controller methods** own the episode through `ctx` (`observe`, `policy.act/sample`, `execute`, `decide`,
`tools["llm"]`), for agent harnesses such as LLM planners.

Slots map onto the 12 harness categories shared with RoboMetaHarness:

| Category | Component |
|---|---|
| observation_grounding | `observation_op` |
| language_alignment | `instruction_op` |
| proposal_selection | `proposer` + `selector` |
| sampler_guidance | `sampler` (in the policy server) |
| uncertainty_compute | `gate` |
| execution_monitoring, contact_feedback | `executor` + `monitor` |
| memory_retrieval | `action_op` (+ experience store) |
| agentic_orchestration | controller |
| skill_composition | skills (planned) |
| world_model_verification | `selector` over predicted futures (planned) |
| experience_evolution | `prepare` stage (planned) |

## Fair by construction

Every comparison runs three arms on the same seeded episodes:

| Arm | What runs |
|---|---|
| baseline | the native policy (sample 0 noise stream, full native chunk, original instruction) |
| null | pays the candidate's compute but executes natively (derived automatically for slot methods; *shadow* mode for controllers) |
| candidate | the method |

Methods only ever see the public observation (the contract's allowlist); the policy still gets the raw
observation, so the baseline is exactly the native vla-eval policy. Per-step observation and action
hashes give each null/baseline pair an **M** (matched), **R** (environment branched, e.g. renderer) or
**X** (harness leak) label. Budgets cap policy samples, LLM calls and tokens; a spent budget ends method
control and the episode finishes natively. Runtime LLMs are pinned (one exact model, `/models` check, no
fallback).

## Policy capability tiers

| Tier | Server | Unlocks |
|---|---|---|
| T0 | any vla-eval policy server, unchanged (`allenai-passthrough`, `pi05-libero-stock`) | agent, instruction and monitoring methods |
| T1 | HarnessStudio subclass via `HarnessPolicyMixin`: seeded, isolated full chunks | best-of-N, executors, exact derived nulls |
| T2/T3 | denoising hooks, velocity / attention / logits / predicted futures | steering and introspection methods |

## Status

v2.0.0.dev0. See [docs/superpowers/specs/2026-10-31-harnessstudio-v2-design.md](docs/superpowers/specs/2026-10-31-harnessstudio-v2-design.md)
for the design and roadmap, and `docs/results/` for gate results.
