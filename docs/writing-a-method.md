# Writing a method (one paper = one package)

A method is a pip package under `methods/<slug>/` that registers a `harnessstudio.methods` entry point.
The framework owns the episode loop, the policy connection, seeds, budgets, arms and traces. A method
only decides what to do with observations, policy samples and actions.

## Start from a scaffold

```bash
hs new rtcf --category memory_retrieval --kind slot --paper arXiv:2608.04527 --title "RTCF"
hs new harness_vla --category agentic_orchestration --kind controller --llm --paper arXiv:2607.08448
uv pip install --python .venv/bin/python -e methods/rtcf && hs verify rtcf
```

The scaffold installs and passes tier 1 before any paper logic is written; replace the TODOs.

## Layout

```
methods/<slug>/
  pyproject.toml     # name "hs-<slug>"; entry point below; depends on "harnessstudio"
  paper.yaml         # the paper card
  hs_<slug>/__init__.py   # components + config()
  README.md          # what is reproduced, what is substituted, how to run
```

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "hs-<slug>"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = ["harnessstudio"]

[project.entry-points."harnessstudio.methods"]
<method-name> = "hs_<slug>:config"

[tool.hatch.build.targets.wheel]
packages = ["hs_<slug>"]
```

Install and check:

```bash
uv pip install --python .venv/bin/python -e methods/<slug>
.venv/bin/hs list
.venv/bin/hs verify <method-name> --tier 0
.venv/bin/hs verify <method-name>            # tier 1 on the fake stack
```

## Paper card (`paper.yaml`)

```yaml
name: medoid-bon                  # [a-z0-9-]+, equals the entry point name
version: "2"
paper: {id: "arXiv:2601.02295v2", title: "CycleVLA ...", venue: "..."}
categories: {primary: proposal_selection, secondary: []}   # IDs from harnessstudio.core.categories
fidelity: component               # faithful | component | tf_variant | adapted
substitutions: []                 # required for tf_variant / adapted: "original -> replacement"
requires: {capabilities: [act, chunk, seeded_sampling, isolated_streams]}
budgets: {policy_calls: null, llm_calls: 0, tool_calls: 0, max_tokens: 0}
null: derived                     # derived (slot methods) | shadow | custom (controllers)
artifacts: []                     # {name, path, sha256, license}
operators: []                     # {name, model, base_url_env, api_key_env, original_model}
original: {repo: "...", commit: "...", reported: "..."}
description: "One sentence."
tier1: {instruction: "open the drawer and put the bowl in it", horizon: 24, chunk_len: 16}   # optional fake-stack settings
```

`config()` builds the MethodConfig from the card and wires components:

```python
from pathlib import Path
from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "proposer": ModuleRef("hs_medoid_bon:Pool", {"num_samples": 8}),
        "selector": ModuleRef("hs_medoid_bon:MBRSelect"),
    })
```

Put `paper.yaml` inside the Python package directory (`hs_<slug>/paper.yaml`) so it ships with the
wheel, and add `[tool.hatch.build.targets.wheel] packages = ["hs_<slug>"]`.

## Two kinds of methods

**Slot methods** fill slots of the default `DecisionLoop`
(`harnessstudio/components/decision_loop.py`). One decision:

`observation_op → instruction_op → gate → proposer (+ sampler in the policy server) → selector → action_op → executor (+ monitor after every step)`

Interfaces are in `harnessstudio/components/protocols.py`:

| Slot | Interface |
|---|---|
| `proposer` | attribute `num_samples` (K seeded samples; sample 0 is the native stream) |
| `selector` | `select(chunks: list[np.ndarray [H, A]], extras: list[dict], obs: dict) -> int` |
| `executor` | `min_k`, `per_step`, `plan(chunk, extras, obs) -> k`, `continue_after(executed, obs) -> bool`, `record_executed(actions)`; `extras["native_length"]` = actions the native policy would run; set `sees_proposals = True` to get `extras["proposals"]`; never cut before `min_k` |
| `monitor` | `check(obs, history) -> "continue" | "interrupt"` |
| `instruction_op` | `rewrite(instruction, obs, memory) -> str` |
| `observation_op` | `transform(obs) -> dict of replaced public keys` |
| `action_op` | `transform(chunk, obs, memory) -> chunk` |
| `gate` | `plan(obs, history) -> {"num_samples": k, ...}` |

Readouts: a proposer may set `readouts = ("velocity",)` (needs capability `velocity_readout`); each
proposal's extras then hold `velocity` and companion keys such as `velocity_timesteps`. Nothing else
from the server reaches methods (seeds are stripped).

Any component can define `diagnostics() -> dict` (JSON); it is written to the trace after every
decision, in every arm. Any component can define `bind(embodiment)`. It receives `Embodiment.action_layout`
(`name -> (start, end)` from the policy's action spec, e.g. `position (0,3)`, `rotation (3,6)`,
`gripper (6,7)`) and can use `embodiment.dims("position", "rotation")`.
`embodiment.eef_position(obs)` returns the end-effector position from a public observation (the
contract says where it lives: LIBERO `states[0:3]`, RoboCasa365 `state["state.end_effector_position_relative"]`,
fake `state[0:2]`), or `None` if the benchmark does not expose it. When a policy declares its actions
as one raw block (vla-eval's pi0 servers), the contract's `action_layout` supplies the names.

The null arm for slot methods is derived automatically: every component still runs, but the policy
gets the original observation and instruction, sample 0 executes for its native length, and cuts
are ignored. Do not branch on the arm; components never see it.

**Controller methods** own the episode: `ModuleRef` in `controller=`, class with
`async def run(self, ctx)`. `ctx` (`harnessstudio/context/episode.py`) offers:

| Member | Meaning |
|---|---|
| `ctx.observe()` | public observation (deep copy; private benchmark keys removed) |
| `ctx.instruction`, `ctx.steps`, `ctx.remaining`, `ctx.ended` | episode state |
| `await ctx.policy.act(obs=None, instruction=None)` | one policy request; returns the chunk `[H, A]` |
| `await ctx.policy.sample(k, obs=None, instruction=None)` | K seeded samples (`Proposal.actions`, `.extras`) |
| `await ctx.execute(actions)` | run actions, one per step |
| `await ctx.decide(instruction=None)` | one DecisionLoop decision with this method's slots |
| `ctx.embodiment` | action layout |
| `ctx.memory` | per-episode scratch dict |
| `ctx.record(event, **fields)`, `ctx.stop(reason)` | trace an event; end control (the episode finishes natively) |

**Runtime LLMs.** Declare operators in the card and budgets for them:

```yaml
operators: [{name: planner, model: "Qwen/Qwen3-VL-32B-Instruct", base_url: nrp, api_key_env: NRP_API_KEY,
             original_model: "claude-opus-4-7"}]
budgets: {policy_calls: 2000, llm_calls: 20, max_tokens: 512}
```

Then `text = await ctx.tools["llm"](messages, operator="planner")`. Images:
`from harnessstudio.tools.llm import image_data_url` and send
`{"type": "image_url", "image_url": {"url": image_data_url(obs["images"]["agentview"])}}`.
The model is pinned (checked against `/models`, no fallback; Astra models are refused). OpenAI models
(`base_url: openai`) get `max_completion_tokens` automatically; set `temperature: null` for models that
only accept their default. `await ctx.tools["llm"](messages, return_usage=True)` returns `(text, usage)`;
`ctx.tools["llm"].last_usage` holds the latest call's token counts. Slow LLM-reviewed methods: pass
`--action-timeout 1800` to `hs eval` / `hs prepare` / tier 2. A spent budget
ends control and the episode finishes natively. Tier 1 runs without network: the controller class must
define `TIER1_LLM_RESPONSES = [...]` (scripted replies). If the operator replaces the paper's model,
use `fidelity: adapted` and list the substitution.

Controllers declare `null: shadow` (the controller runs and pays its calls in the null arm while the
executed actions are native) and must cap `budgets.policy_calls`.

## Development stage (memory, evolved programs)

Methods that build memory or programs from rollouts before evaluation add `prepare=ModuleRef(...)`
to `config_from_card(...)`. The Preparer:

```python
class Preparer:
    collect_arm = "baseline"     # which arm runs on development cases ("candidate" to explore with the method)
    record = ("state",)          # public observation keys kept per step (the executed action is always kept)
    def build(self, records, out_dir) -> dict:   # write files into out_dir; return {artifact name: file name}
        ...
```

```bash
hs prepare M --config dev.yaml --case-namespace dev-seed1007 --policy-url ... --contract libero --profile ... --out prep/
hs eval M --config eval.yaml --case-namespace seed7 --prepared prep/ ...
```

Chained development: `hs prepare M ... --prepared prep_round1/ --out prep_round2/` starts from the earlier
artifacts (passed to `Preparer.build(records, out_dir, previous={name: path})` when it accepts `previous`) and
keeps all earlier development cases forbidden. Separately run arms can be paired with
`hs pair --arm baseline=DIR --arm null=DIR --arm candidate=DIR --out OUT`.

`prepared.json` pins every artifact by sha256 and lists the development case keys; evaluation refuses
those cases (they show up as invalid and drop out of pairing). A case key is
`(contract[@case-namespace], task, episode_idx)`. On LIBERO / LIBERO-Pro the benchmark seed does not change
the initial state, so do not set a namespace for seed changes: develop on episodes 0..k-1 and evaluate on more
episodes; the development ones are excluded automatically. Set a namespace only when the development run uses
a benchmark seed that really changes the scenes (RoboCasa365 `seed`).

**Scenes depend on run history.** vla-eval's LIBERO benchmark reuses one environment per task and LIBERO
re-places fixtures (stove, cabinet, rack) on every reset, so the scene of (task, episode k) depends on which
earlier episodes of that task ran in the same process; RoboCasa365 resets depend on process history too.
Paired arms are exact only when they run the same case list in the same order, each in a fresh process, as
`hs eval` does. Do not pair runs that split one task's episodes differently (e.g. an episode offset or a subset
of episodes) unless the benchmark replays the earlier resets (see `methods/harness_vla/bench_ext`,
`match_sequential_resets`). Components receive artifacts through `load_artifacts(artifacts)` (name -> path); controllers
read `ctx.artifacts`. `harnessstudio.components.memory` has `write_trajectories` / `TrajectoryStore`
(nearest-neighbour retrieval over stored states). Report zero-shot and prepared results separately.

## Skills

`harnessstudio.skills.eef_delta` provides analytic primitives for end-effector-delta benchmarks
(LIBERO, LIBERO-Pro): `move_to(ctx, xyz)`, `move_by(ctx, dxyz)`, `set_gripper(ctx, "open"|"close")`,
`release(ctx)` and `vla_act(ctx, instruction, steps=...)` (the frozen policy as a skill). They act through
`ctx.execute` / `ctx.decide`, so budgets, traces and the shadow null apply.

## Rules

- Training-free: no fitting of anything. Frozen, hashed artifacts only.
- Public inputs only: use `ctx.observe()` and `obs` arguments; never read simulator state.
- Deterministic given inputs: no unseeded randomness, no wall-clock dependence.
- Fresh state per episode: components are built per episode; keep state on `self`.
- Name every substitution of a trained or unavailable part (`tf_variant`, `adapted`).

## Runtime models, services and framework changes

- Every LLM/VLM provider, model and local service a method may use is in `docs/ENDPOINTS.md`
  (machine-readable: `harnessstudio/endpoints.yaml`). Generate operator pins with
  `.venv/bin/python -m harnessstudio.endpoints pin PROVIDER MODEL --name ROLE`; NRP is the default, OpenAI and
  Claude are pinned opt-ins, Astra needs the user's explicit confirmation for each use.
- Porting a paper end to end (audit, fidelity, parity checks, development stage, evaluation, retrospective) is
  described by the self-improving porting skill
  `/home/sejain/research/cvpr27/RoboMetaHarness-claude/.claude/skills/harnessstudio-v2-porting/`.
- If a paper needs something the framework cannot express, do not monkeypatch it from the method: propose a
  framework extension (`docs/extensions/README.md`), build it in a worktree and pass `scripts/regress.py`.
  `python3 scripts/regress.py --inflight` lists running `hs` processes; never edit files they import.
