"""harness-vla: Harness VLA (arXiv:2607.08448) zero-shot mode as a HarnessStudio controller method.

A planner REPL over the paper's fixed primitive library (MOVE_TO, MOVE_POSE, ROTATE_WRIST, ROTATE_PITCH,
SET_GRIPPER, RELEASE, VLA_ACT) with SAM3 + depth perception, ported from the authors' RPent code
(Apache-2.0). See README.md and paper.yaml for what is reproduced and what is substituted.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

from .controller import HarnessVLA  # noqa: F401

CONTRACT = Path(__file__).with_name("contracts") / "libero_pro_rgbd.yaml"


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), controller=ModuleRef(
        "hs_harness_vla.controller:HarnessVLA",
        {"max_turns": 40, "max_tool_calls": 60, "image_scale": 2, "wrist_image": True, "history_turns": 8,
         "llm_retries": 2, "sam3_url": "http://127.0.0.1:8117", "max_tokens": 800}))


def config_sol():
    """harness-vla with OpenAI gpt-6.1-sol as the planner (reasoning model: larger completion cap)."""
    return config_from_card(Path(__file__).with_name("paper_sol.yaml"), controller=ModuleRef(
        "hs_harness_vla.controller:HarnessVLA",
        {"max_turns": 40, "max_tool_calls": 60, "image_scale": 2, "wrist_image": True, "history_turns": 8,
         "llm_retries": 2, "sam3_url": "http://127.0.0.1:8117", "max_tokens": 6000}))


def config_sol_fewshot():
    """harness-vla-sol in the paper's few-shot (memory) mode: a development stage explores the reference scene of
    each task and distills task memory (hs_harness_vla.memory.Preparer); evaluation injects it into the planner."""
    return config_from_card(Path(__file__).with_name("paper_sol_fewshot.yaml"), controller=ModuleRef(
        "hs_harness_vla.controller:HarnessVLA",
        {"max_turns": 40, "max_tool_calls": 60, "image_scale": 2, "wrist_image": True, "history_turns": 8,
         "llm_retries": 2, "sam3_url": "http://127.0.0.1:8117", "max_tokens": 6000, "memory": True,
         "memory_chars": 9000}),
        prepare=ModuleRef("hs_harness_vla.memory:Preparer"))
