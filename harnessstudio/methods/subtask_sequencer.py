"""subtask-sequencer: prompt the frozen policy with one step of a composite instruction at a time.

Observed failure (plain GR00T N1.5, RoboCasa365 Composite-Unseen, 76/80 failures): the policy often
completes the first step of a multi-step instruction and then stalls. A reactive policy sees the same
full instruction at every step and has no notion of which part is done.

Mechanism (training-free variant of Inner Monologue, arXiv:2207.05608v1, closest to the catalog's
grounded subgoal routing):
- split the instruction into self-contained imperative steps with fixed text rules (no model);
- execute whole native policy decisions with the current step as the prompt;
- advance to the next step when public end-effector motion stalls after a minimum dwell.
Instructions without separable steps run exactly as the plain policy. No extra policy inference and
no LLM calls, so the shadow-mode null executes the baseline's actions at the baseline's cost.
"""
from __future__ import annotations

import math
import re

from ..core.capabilities import caps
from ..core.method import Fidelity, MethodConfig, ModuleRef

VERBS = frozenset("""put turn close open place pick move take return start pour grab retrieve select sort set
dump gather press push pull rinse wash insert remove fill carry bring transfer slide flip stir""".split())
PRONOUNS = frozenset({"it", "them", "this", "that", "these", "those"})


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


class SubtaskSequencer:
    identity = {"module": "subtask_sequencer", "version": "1"}

    def __init__(self, min_act_steps, min_dwell_steps=96, stall_window_steps=48, stall_tolerance_m=0.01):
        if stall_window_steps < 1 or min_dwell_steps < stall_window_steps or stall_tolerance_m <= 0:
            raise ValueError("need min_dwell_steps >= stall_window_steps >= 1 and stall_tolerance_m > 0")
        self.min_dwell, self.window, self.tolerance = min_dwell_steps, stall_window_steps, stall_tolerance_m

    def stalled(self, history) -> bool:
        """history: [(step, eef_xyz)] at decision boundaries for the current subtask."""
        if not history or history[-1][0] - history[0][0] < self.min_dwell:
            return False
        latest = history[-1][0]
        recent = [p for s, p in history if s >= latest - self.window]
        if len(recent) < 2 or latest - min(s for s, _ in history if s >= latest - self.window) < self.window:
            return False
        return max(math.dist(recent[0], p) for p in recent) < self.tolerance

    def run(self, tools):
        steps = split_steps(tools.instruction)
        tools.memory["steps"] = steps
        if len(steps) == 1:
            return  # nothing to sequence: the episode finishes natively with the original instruction
        advanced = []
        for index, step in enumerate(steps):
            history = [(tools.steps, tools.eef_position())]
            while not tools.ended:
                tools.act(step)  # one whole native decision with this step as the prompt
                history.append((tools.steps, tools.eef_position()))
                if index < len(steps) - 1 and self.stalled(history):
                    advanced.append(tools.steps)
                    break
            if tools.ended:
                break
        tools.memory["advanced_at"] = advanced


def config() -> MethodConfig:
    return MethodConfig(
        name="subtask-sequencer", version="1", paper="arXiv:2207.05608v1", fidelity=Fidelity.TF_VARIANT,
        substitutions=("fixed text rules split the instruction instead of an LLM planner",
                       "public end-effector stall after a minimum dwell replaces the paper's success detector",
                       "the frozen robot policy's own prompt replaces the paper's skill library"),
        required=caps("action_chunk", "seeded_sampling", "proprio"),
        slots={"controller": ModuleRef("harnessstudio.methods.subtask_sequencer:SubtaskSequencer",
                                       {"min_act_steps": 16, "min_dwell_steps": 96, "stall_window_steps": 48,
                                        "stall_tolerance_m": 0.01})},
        description="One instruction step at a time; advance when the end effector stalls (<1 cm over 48 steps, "
                    "after >= 96 steps on the step). No extra inference; null = baseline actions.")
