"""llm-subtask-planner: a controller method (milestone 2 demo of the agent-harness slot).

Training-free variant of the Inner Monologue loop (Huang et al., arXiv:2207.05608v1): a frozen LLM
splits the instruction into short subtasks, the frozen robot policy executes each subtask as its
prompt, and (optionally) a frozen vision-language model looks at the current camera image to decide
when a subtask is done. When the plan is finished or the LLM budget is spent, the episode continues
natively with the original instruction.

Everything the controller sees is public: the instruction and the policy's public camera images.
"""
from __future__ import annotations

import json
import re

from ..core.capabilities import caps
from ..core.method import Fidelity, MethodConfig, ModuleRef

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
    """A third-person public camera: '*agentview*' first, then 'observation/image', then any image."""
    keys = sorted(keys)
    for test in (lambda k: "agentview" in k, lambda k: k == "observation/image",
                 lambda k: ("image" in k or k.startswith("video.")) and "wrist" not in k and "hand" not in k):
        hits = [k for k in keys if test(k)]
        if hits:
            return hits[0]
    raise ValueError("No public camera to check progress with")


class SubtaskPlanner:
    identity = {"module": "llm_subtask_planner", "version": "1"}
    TIER1_LLM_RESPONSES = ['["reach the object", "grasp it", "place it"]', "no", "yes"]

    def __init__(self, min_act_steps, llm, max_subtasks=4, steps_per_subtask=200, check_every=0,
                 check_camera=None):
        self.max_subtasks, self.steps_per_subtask = max_subtasks, steps_per_subtask
        self.check_every, self.check_camera = check_every, check_camera
        if check_every and check_every < min_act_steps:
            raise ValueError("check_every must be >= min_act_steps")

    def run(self, tools):
        reply = tools.llm([{"role": "user", "content": PLAN_PROMPT.format(n=self.max_subtasks,
                                                                           instruction=tools.instruction)}])
        plan = parse_plan(reply, self.max_subtasks)
        tools.memory["plan"] = plan
        for subtask in plan:
            spent = 0
            while spent < self.steps_per_subtask and not tools.ended:
                chunk = self.check_every or self.steps_per_subtask
                spent += tools.act(subtask, steps=min(chunk, self.steps_per_subtask - spent))
                if self.check_every and not tools.ended:
                    camera = self.check_camera or pick_camera(tools.observe().values)
                    image = tools.image_data_url(tools.image(camera))
                    answer = tools.llm([{"role": "user", "content": [
                        {"type": "text", "text": CHECK_PROMPT.format(subtask=subtask, instruction=tools.instruction)},
                        {"type": "image_url", "image_url": {"url": image}}]}], max_tokens=8)
                    if answer.strip().lower().startswith("yes"):
                        break
            if tools.ended:
                return


# Pinned 2026-10-28 after GET /models and a real image probe (alias qwen3-vl answered as
# Qwen/Qwen3.8-Flash-Next-FP8). Hidden reasoning is disabled: with it on, short answers come back empty.
NRP_VL = {"provider": "openai_compatible", "base_url": "nrp", "model": "qwen3-vl",
          "api_key_env": "NRP_API_KEY", "max_calls": 16, "max_tokens": 256, "temperature": 0.0,
          "timeout_seconds": 180, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def config() -> MethodConfig:
    return MethodConfig(
        name="llm-subtask-planner", version="1", paper="arXiv:2207.05608v1", fidelity=Fidelity.TF_VARIANT,
        substitutions=("NRP-hosted open-weight qwen3-vl replaces the paper's InstructGPT planner and success detector",
                       "the frozen robot policy's own prompt replaces the paper's skill library",
                       "success detection reads one public camera image (no scene descriptor)"),
        required=caps("action_chunk", "seeded_sampling"),
        slots={"controller": ModuleRef("harnessstudio.methods.llm_subtask_planner:SubtaskPlanner",
                                       {"min_act_steps": 16, "llm": NRP_VL, "max_subtasks": 4,
                                        "steps_per_subtask": 240, "check_every": 96,
                                        "check_camera": None})},
        description="LLM splits the instruction into subtasks; the policy executes each as its prompt; "
                    "a VLM checks completion every 96 steps. Null arm: same LLM calls, native actions.")
