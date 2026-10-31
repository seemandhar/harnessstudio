"""FastHarness (arXiv:2610.31427, "Experience-Driven Review Scheduling for Vision-Language-Action Policies",
github.com/FastHarness/fast-harness @ dfe070f, MIT) as a HarnessStudio controller method.

A frozen policy proposes every chunk; an LLM reviewer supervises it. A fast path skips reviews as stage
memory and reviewed outcomes accumulate; a recovery path concentrates review on fixing errors; failure
experience is kept. See README.md for what is reproduced and what is adapted.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

from .controller import FastHarnessController
from .prepare import Preparer

# The paper's live RoboDojo settings (README run command + CLI defaults): --contact-gating --commit-gating
# --persist-error-review, memory decay 1.0 (pure accumulation), commit_cap 6, max_interval 6, min_samples 4.
CONTROLLER_PARAMS = {
    "max_interval": 6, "min_samples": 4, "cooldown_successes": 3, "max_unreviewed_steps": 60,
    "match_threshold": 0.06, "ambiguity_margin": 0.01, "error_radius": 0.04, "audit_probability": 0.05,
    "contact_gating": True, "commit_gating": True, "commit_cap": 6, "persist_error_review": True,
    "memory_decay": 1.0, "cold_full_review": True, "max_proposal_steps": 15, "schema_retries": 2,
    "max_completion_tokens": 4096, "image_max_side": None, "gripper_open_value": 1.0,
    "gripper_closed_value": 0.0, "seed": 0, "operator": "reviewer",
}


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"),
                            controller=ModuleRef("hs_fast_harness:FastHarnessController", CONTROLLER_PARAMS),
                            prepare=ModuleRef("hs_fast_harness:Preparer", {"memory_decay": 1.0}))


__all__ = ["FastHarnessController", "Preparer", "config"]
