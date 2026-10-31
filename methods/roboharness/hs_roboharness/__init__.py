"""roboharness: RoboHarness (arXiv:2603.24060v3; repo LZY-1021/RoboHarness, MIT, commit 716defc) as a
HarnessStudio controller method.

A frozen VLA wrapped by Dual-Memory RAG (CLIP+MiniLM+hash retrieval of success/failure experiences), a VLM
orchestrator that edits the agent-view frame (SAM3 overlay / inpainting), rewrites the instruction and triggers
control flow (Encore rollback, key-step retry, subtask decomposition), and a memory built from a development
pass. Ported from the authors' evaluation code; see README.md and paper.yaml for every substitution.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

from .controller import RoboHarness  # noqa: F401

CONTROLLER_PARAMS = {"max_attempts": 3, "perceive_every": 10, "n_action_steps": 10, "modification_active_limit": 40,
                     "encore_reverse_steps": 40, "encore_buffer_steps": 20, "retry_reverse_steps": 50,
                     "retry_buffer_steps": 25, "success_top_k": 1, "failure_top_k": 2, "retrieval_threshold": 0.4,
                     "image_key": "agentview", "tool_url": "http://127.0.0.1:8131", "keyframe_every": 5}


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"),
                            controller=ModuleRef("hs_roboharness.controller:RoboHarness", dict(CONTROLLER_PARAMS)),
                            prepare=ModuleRef("hs_roboharness.prepare:Preparer", {"tool_url": "http://127.0.0.1:8131"}))
