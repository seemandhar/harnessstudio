"""subtask-sequencer: prompt the frozen policy with one step of a composite instruction at a time.

Observed failure (plain GR00T N1.5, RoboCasa365 Composite-Unseen, 76/80 failures in v1): the policy
often completes the first step of a multi-step instruction and then stalls. A reactive policy sees the
same full instruction at every step and has no notion of which part is done.

Mechanism (training-free variant of Inner Monologue, arXiv:2207.05608v1):
- split the instruction into self-contained imperative steps with fixed text rules (no model);
- run whole DecisionLoop decisions with the current step as the prompt (``ctx.decide(instruction=step)``);
- advance to the next step when the public end-effector position stalls after a minimum dwell; the last
  step runs until the episode ends. If the benchmark exposes no end-effector position, each step but the
  last gets a fixed budget: an equal share of the steps remaining when control starts.
Instructions without separable steps return immediately, so the episode runs exactly as the plain policy.
No extra policy inference and no LLM calls, so the shadow-mode null executes the baseline's actions at the
baseline's cost.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

__all__ = ["split_steps", "SubtaskSequencer", "config"]

VERBS = frozenset("""put turn close open place pick move take return start pour grab retrieve select sort set
dump gather press push pull rinse wash insert remove fill carry bring transfer slide flip stir""".split())
PRONOUNS = frozenset({"it", "them", "this", "that", "these", "those"})


# --- instruction splitting (verbatim from v1) ------------------------------------------------------------
def _starts_with_verb(text: str) -> bool:
    words = text.strip().split()
    return bool(words) and words[0].lower() in VERBS


def _self_contained(text: str) -> bool:
    """An imperative clause that does not start by referring back to an earlier object."""
    words = text.strip().split()
    return len(words) >= 2 and words[0].lower() in VERBS and words[1].lower() not in PRONOUNS


def _clean(text: str) -> str:
    text = re.sub(r"^\s*(?:and\s+)?then\s+", "", text.strip(), flags=re.I).strip(" ,.")
    return text[:1].upper() + text[1:] + "." if text else ""


def split_steps(instruction: str) -> list:
    """Deterministic split into imperative steps. Returns [instruction] when nothing separates cleanly."""
    text = " ".join(instruction.split())
    sentences = [s for s in re.split(r"(?<=\.)\s+", text) if s.strip()]
    steps = []
    for sentence in sentences:
        body = sentence.strip().rstrip(".")
        # Candidate cut points: ", then", ", and then", " then " (sequencing), ", and VERB", ", VERB".
        parts = [body]
        for pattern in (r",?\s+(?:and\s+)?then\s+", r",\s+and\s+(?=\w)", r",\s+(?=\w)"):
            nxt = []
            for part in parts:
                pieces = re.split(pattern, part)
                if len(pieces) == 1:
                    nxt.append(part)
                    continue
                merged = [pieces[0]]
                for piece in pieces[1:]:
                    sequencing = pattern.startswith(",?")
                    if (sequencing and _starts_with_verb(piece)) or (
                            not sequencing and _self_contained(piece) and _starts_with_verb(merged[-1])):
                        merged.append(piece)
                    else:
                        joiner = " then " if sequencing else (", and " if "and" in pattern else ", ")
                        merged[-1] = merged[-1] + joiner + piece
                nxt.extend(merged)
            parts = nxt
        steps.extend(_clean(p) for p in parts if p.strip())
    steps = [s for s in steps if s]
    return steps if len(steps) > 1 else [instruction]


# --- controller ------------------------------------------------------------------------------------------
class SubtaskSequencer:
    """Controller: ``run(ctx)`` once per episode (see docs/writing-a-method.md, Controller methods)."""

    def __init__(self, min_dwell_steps: int = 96, stall_window_steps: int = 48, stall_tolerance_m: float = 0.01,
                 fallback_subtask_steps: Optional[int] = None) -> None:
        if stall_window_steps < 1 or min_dwell_steps < stall_window_steps or stall_tolerance_m <= 0:
            raise ValueError("need min_dwell_steps >= stall_window_steps >= 1 and stall_tolerance_m > 0")
        if fallback_subtask_steps is not None and (type(fallback_subtask_steps) is not int or fallback_subtask_steps < 1):
            raise ValueError("fallback_subtask_steps must be a positive integer or None")
        self.min_dwell, self.window, self.tolerance = min_dwell_steps, stall_window_steps, stall_tolerance_m
        self.fallback_steps = fallback_subtask_steps

    def stalled(self, history) -> bool:
        """history: [(step, eef_xyz)] at decision boundaries for the current subtask.

        True once the step has run >= min_dwell steps and the end effector stayed within ``tolerance`` of
        its position at the window anchor (the last boundary at or before ``latest - window``)."""
        if len(history) < 2 or history[-1][0] - history[0][0] < self.min_dwell:
            return False
        latest = history[-1][0]
        anchor = max(i for i, (s, _) in enumerate(history) if s <= latest - self.window)  # exists: dwell >= window
        recent = [p for _, p in history[anchor:]]
        if any(p is None for p in recent):
            return False
        return max(math.dist(recent[0], p) for p in recent) < self.tolerance

    @staticmethod
    def _eef(ctx) -> Optional[tuple]:
        pos = ctx.embodiment.eef_position(ctx.observe())
        return None if pos is None else tuple(float(v) for v in pos)

    async def run(self, ctx) -> None:
        steps = split_steps(ctx.instruction)
        ctx.memory["steps"] = steps
        ctx.record("subtask_plan", steps=steps)
        if len(steps) == 1:
            return  # nothing to sequence: the episode finishes natively with the original instruction

        budget = None
        if self._eef(ctx) is None:
            budget = self.fallback_steps or max(1, ctx.remaining // len(steps))
            ctx.record("eef_unavailable", advance_rule="fixed_budget", steps_per_subtask=budget)
        rule = "stall" if budget is None else "fixed_budget"
        ctx.memory["advance_rule"] = rule
        advanced = ctx.memory["advanced_at"] = []

        for index, step in enumerate(steps):
            last = index == len(steps) - 1
            start = ctx.steps
            history = [(start, self._eef(ctx))]
            ctx.record("subtask_start", index=index, prompt=step)
            advance = False
            while not ctx.ended and not advance:
                await ctx.decide(instruction=step)  # one whole decision with this step as the prompt
                if last:
                    continue
                if budget is None:
                    history.append((ctx.steps, self._eef(ctx)))
                    advance = self.stalled(history)
                else:
                    advance = ctx.steps - start >= budget
            if not advance or ctx.ended:
                break  # the episode ended on this step
            advanced.append(ctx.steps)
            ctx.record("subtask_advance", index=index, rule=rule, steps_on_subtask=ctx.steps - start)


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), controller=ModuleRef(
        "hs_subtask_sequencer:SubtaskSequencer",
        {"min_dwell_steps": 96, "stall_window_steps": 48, "stall_tolerance_m": 0.01, "fallback_subtask_steps": None}))
