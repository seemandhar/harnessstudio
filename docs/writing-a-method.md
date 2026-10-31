# Writing a method (one paper = one package)

A method is a pip package under `methods/<slug>/` that registers a `harnessstudio.methods` entry point.
The framework owns the episode loop, the policy connection, seeds, budgets, arms and traces. A method
only decides what to do with observations, policy samples and actions.

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
| `executor` | `min_k`, `per_step`, `plan(chunk, extras, obs) -> k`, `continue_after(executed, obs) -> bool`, `record_executed(actions)`; set `sees_proposals = True` to get `extras["proposals"]` |
| `monitor` | `check(obs, history) -> "continue" | "interrupt"` |
| `instruction_op` | `rewrite(instruction, obs, memory) -> str` |
| `observation_op` | `transform(obs) -> dict of replaced public keys` |
| `action_op` | `transform(chunk, obs, memory) -> chunk` |
| `gate` | `plan(obs, history) -> {"num_samples": k, ...}` |

Any component can define `bind(embodiment)`. It receives `Embodiment.action_layout`
(`name -> (start, end)` from the policy's action spec, e.g. `position (0,3)`, `rotation (3,6)`,
`gripper (6,7)`) and can use `embodiment.dims("position", "rotation")`.

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

Controllers declare `null: shadow` (the controller runs and pays its calls in the null arm while the
executed actions are native) and must cap `budgets.policy_calls`.

## Rules

- Training-free: no fitting of anything. Frozen, hashed artifacts only.
- Public inputs only: use `ctx.observe()` and `obs` arguments; never read simulator state.
- Deterministic given inputs: no unseeded randomness, no wall-clock dependence.
- Fresh state per episode: components are built per episode; keep state on `self`.
- Name every substitution of a trained or unavailable part (`tf_variant`, `adapted`).
