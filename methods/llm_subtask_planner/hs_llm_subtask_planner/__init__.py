"""llm-subtask-planner: training-free variant of the Inner Monologue loop (arXiv:2207.05608v1).

A frozen LLM splits the instruction into short subtasks, the frozen robot policy executes each subtask as
its prompt, and (optionally) a frozen VLM looks at the current camera image to decide when a subtask is
done. When the plan is finished or the LLM budget is spent, the episode continues natively.
Everything the controller sees is public: the instruction and the policy's public camera images.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef
from harnessstudio.tools.llm import image_data_url

PLAN_PROMPT = (
    "You control a household robot arm with a language-conditioned policy. Split the task into at most "
    "{n} short imperative subtasks the policy can follow, in order. Reply with a JSON list of strings only.\n"
    "Task: {instruction}")
CHECK_PROMPT = ("The robot is working on the subtask: \"{subtask}\" (overall task: \"{instruction}\"). "
                "Looking at the image, is this subtask complete? Answer only yes or no.")


def parse_plan(text: str, limit: int) -> list:
    """First JSON list of strings in the reply; falls back to numbered or bulleted lines."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    match = re.search(r"\[.*?\]", text, flags=re.S)
    items = []
    if match:
        try:
            items = [str(x).strip() for x in json.loads(match.group(0)) if str(x).strip()]
        except (ValueError, TypeError):
            items = []
    if not items:
        items = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
        items = [x for x in items if x]
    return items[:limit]


def pick_camera(keys) -> str:
    """A third-person public camera: '*agentview*' first, then any non-wrist image."""
    keys = sorted(keys)
    for test in (lambda k: "agentview" in k, lambda k: k == "observation/image",
                 lambda k: "wrist" not in k and "hand" not in k):
        hits = [k for k in keys if test(k)]
        if hits:
            return hits[0]
    raise ValueError("No public camera to check progress with")


class SubtaskPlanner:
    TIER1_LLM_RESPONSES = ['["reach the object", "grasp it", "place it"]', "no", "yes"]

    def __init__(self, max_subtasks: int = 4, steps_per_subtask: int = 240, check_every: int = 96,
                 check_camera: str | None = None) -> None:
        self.max_subtasks, self.steps_per_subtask = max_subtasks, steps_per_subtask
        self.check_every, self.check_camera = check_every, check_camera

    async def act(self, ctx, prompt: str, steps: int) -> int:
        """Whole policy decisions with this prompt until `steps` are executed (v1 tools.act)."""
        executed = 0
        while executed < steps and not ctx.ended:
            executed += await ctx.decide(instruction=prompt)
        return executed

    async def run(self, ctx) -> None:
        llm = ctx.tools["llm"]
        reply = await llm([{"role": "user", "content": PLAN_PROMPT.format(n=self.max_subtasks,
                                                                           instruction=ctx.instruction)}])
        plan = parse_plan(reply, self.max_subtasks)
        ctx.memory["plan"] = plan
        ctx.record("plan", subtasks=plan)
        for subtask in plan:
            spent = 0
            while spent < self.steps_per_subtask and not ctx.ended:
                chunk = self.check_every or self.steps_per_subtask
                spent += await self.act(ctx, subtask, min(chunk, self.steps_per_subtask - spent))
                if self.check_every and not ctx.ended:
                    images = ctx.observe().get("images") or {}
                    camera = self.check_camera or pick_camera(images.keys())
                    answer = await llm([{"role": "user", "content": [
                        {"type": "text", "text": CHECK_PROMPT.format(subtask=subtask, instruction=ctx.instruction)},
                        {"type": "image_url", "image_url": {"url": image_data_url(images[camera])}}]}], max_tokens=8)
                    if answer.strip().lower().startswith("yes"):
                        break
            if ctx.ended:
                return
        # plan finished: the method returns and the episode continues natively


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), controller=ModuleRef(
        "hs_llm_subtask_planner:SubtaskPlanner",
        {"max_subtasks": 4, "steps_per_subtask": 240, "check_every": 96, "check_camera": None}))
