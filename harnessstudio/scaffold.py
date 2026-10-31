"""`hs new`: scaffold a method package that installs and passes tier 1 before any paper logic is written."""
from __future__ import annotations

import re
from pathlib import Path

from .core.categories import CATEGORIES

PYPROJECT = '''[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "hs-{slug_dash}"
version = "0.1.0"
description = "{title}"
requires-python = ">=3.10"
dependencies = ["harnessstudio"]

[project.entry-points."harnessstudio.methods"]
{name} = "{pkg}:config"

[tool.hatch.build.targets.wheel]
packages = ["{pkg}"]
'''

CARD = '''name: {name}
version: "0.1"
paper: {{id: "{paper}", title: "{title}", venue: ""}}
categories: {{primary: {category}, secondary: []}}
fidelity: faithful            # faithful | component | tf_variant | adapted (list substitutions for the last two)
substitutions: []
requires: {{capabilities: {caps}}}
budgets: {{policy_calls: {policy_calls}, llm_calls: {llm_calls}, tool_calls: 0, max_tokens: {max_tokens}}}
null: {null}
artifacts: []                 # {{name, path, sha256, license}} for frozen checkpoints/prompts/memory
operators: {operators}
original: {{repo: "", commit: "", reported: ""}}
description: "TODO: one sentence on what the harness does."
# tier1: {{instruction: "open the drawer and put the bowl in it", horizon: 24}}
'''

SLOT_INIT = '''"""{title} ({paper}) as a HarnessStudio method. See docs/writing-a-method.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef


class Proposer:
    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples          # K seeded samples per decision; sample 0 = native stream


class Selector:
    def bind(self, embodiment) -> None:         # optional: action layout (embodiment.dims("position", ...))
        self.embodiment = embodiment

    def select(self, chunks, extras, obs) -> int:
        # TODO: the paper's selection rule. chunks: list of [H, A] arrays; obs: public observation.
        return int(np.argmin([float(np.abs(c).sum()) for c in chunks]))


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={{
        "proposer": ModuleRef("{pkg}:Proposer", {{"num_samples": 4}}),
        "selector": ModuleRef("{pkg}:Selector"),
    }})
'''

CONTROLLER_INIT = '''"""{title} ({paper}) as a HarnessStudio controller method. See docs/writing-a-method.md."""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef


class Controller:
{tier1}
    def __init__(self, steps_per_subtask: int = 120) -> None:
        self.steps_per_subtask = steps_per_subtask

    async def run(self, ctx) -> None:
        # TODO: the paper's loop. Act only through ctx: ctx.observe(), ctx.policy.sample/act,
        # ctx.decide(instruction=...), ctx.execute(actions), {llm_hint}harnessstudio.skills.eef_delta.
        subtasks = [ctx.instruction]
{plan}        for subtask in subtasks:
            done = 0
            while done < self.steps_per_subtask and not ctx.ended:
                done += await ctx.decide(instruction=subtask)
        # returning hands the rest of the episode to the native policy


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"),
                            controller=ModuleRef("{pkg}:Controller", {{"steps_per_subtask": 120}}))
'''

README = '''# {name}

{title} ({paper}) as a HarnessStudio method.

- What is reproduced: TODO
- What is substituted (and why the fidelity label): TODO

```bash
uv pip install --python .venv/bin/python -e methods/{slug}
hs verify {name} --tier 0 && hs verify {name}
hs eval {name} --config <vla-eval config> --policy-url ws://... --contract libero --profile <profile> --out runs/{name}
```
'''


def create(slug: str, *, name: str, category: str, kind: str, paper: str, title: str, llm: bool,
           root: Path = Path("methods")) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise ValueError("slug must match [a-z0-9_]+")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError("name must match [a-z0-9][a-z0-9-]*")
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; choose from {sorted(CATEGORIES)}")
    if kind not in ("slot", "controller"):
        raise ValueError("kind must be 'slot' or 'controller'")
    pkg = f"hs_{slug}"
    base = Path(root) / slug
    if base.exists():
        raise FileExistsError(f"{base} already exists")
    (base / pkg).mkdir(parents=True)
    slot = kind == "slot"
    caps = "[act, chunk, seeded_sampling, isolated_streams]" if slot else "[act, chunk]"
    operators = ('[{name: planner, model: "qwen3-vl", base_url: nrp, api_key_env: NRP_API_KEY, '
                 'original_model: "", extra_body: {chat_template_kwargs: {enable_thinking: false}}}]') if llm else "[]"
    fields = dict(slug=slug, slug_dash=slug.replace("_", "-"), name=name, pkg=pkg, category=category,
                  paper=paper, title=title.replace('"', "'"), caps=caps, operators=operators,
                  policy_calls="null" if slot else 2000, llm_calls=16 if llm else 0, max_tokens=512 if llm else 0,
                  null="derived" if slot else "shadow")
    (base / "pyproject.toml").write_text(PYPROJECT.format(**fields))
    (base / pkg / "paper.yaml").write_text(CARD.format(**fields))
    if slot:
        init = SLOT_INIT.format(**fields)
    else:
        tier1 = '    TIER1_LLM_RESPONSES = [\'["first subtask", "second subtask"]\']\n' if llm else ""
        plan = ('        import json\n'
                '        reply = await ctx.tools["llm"]([{"role": "user", "content": f"Split into subtasks as a JSON '
                'list: {ctx.instruction}"}])\n'
                '        try:\n            subtasks = [str(s) for s in json.loads(reply)] or subtasks\n'
                '        except ValueError:\n            pass\n') if llm else ""
        init = CONTROLLER_INIT.format(tier1=tier1, plan=plan, llm_hint='await ctx.tools["llm"](messages), ' if llm else "",
                                      **fields)
    (base / pkg / "__init__.py").write_text(init)
    (base / "README.md").write_text(README.format(**fields))
    return base
