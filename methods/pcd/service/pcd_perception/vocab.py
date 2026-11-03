"""Instruction parsing, class list and GroundingDINO aliases (re-derived; data values restated from the paper code).

Upstream references (Koorye/PCD @ 31b027e):
- contrast_utils/instruction_templates.py:4-13   the eight regular expressions, tried in order with re.match
- contrast_utils/instruction_templates.py:33-36  four literal str.replace rewrites before matching; NO lowercasing
- contrast_utils/instruction_templates.py:37-44  first matching template wins; groups are stripped; ValueError otherwise
- contrast_utils/instruction_templates.py:15-30  drawer-part expansion, only when get_all_parts (False in
                                                 properties.py CONTRAST_IMAGE_CONFIG, never overridden by the script)
- contrast_utils/contrast_image_generator.py:97-99 keep objects for the predictor path are exactly ["robot"]
- contrast_utils/mask_predictors.py:11-14        class list; tracker object id = index + 1 (:463, :473)
- contrast_utils/mask_predictors.py:18-30        GDINO alias table
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

# (regex, applied with re.match) -- instruction_templates.py:4-13, same order.
TEMPLATES: List[str] = [
    r"pick (.+)",
    r"move (.+) near (.+)",
    r"open (.+)",
    r"close (.+)",
    r"place (.+) into (.+)",
    r"put (.+) into (.+)",
    r"put (.+) on (.+)",
    r"stack (.+) on (.+)",
]

# Literal rewrites, applied in this order -- instruction_templates.py:33-36.
REWRITES: List[tuple] = [("the ", ""), ("plate", "plate larger"), ("towel", "table cloth shorter"),
                         ("block", "cube")]

DRAWER_PARTS: List[str] = ["body", "top drawer", "middle drawer", "bottom drawer"]  # instruction_templates.py:15

# mask_predictors.py:11-14 (order matters: object id = index + 1).
CLASSNAMES: List[str] = ["robot", "coke can", "pepsi can", "redbull can", "7up can", "blue plastic bottle",
                         "apple", "orange", "sponge", "bottom drawer", "middle drawer", "top drawer",
                         "eggplant", "spoon", "carrot", "plate larger", "table cloth shorter",
                         "yellow basket", "green cube", "yellow cube", "goal_site"]

# mask_predictors.py:18-30.
GDINO_ALIASES: Dict[str, str] = {
    "7up can": "white 7up pop can",
    "redbull can": "redbull pop can",
    "coke can": "coke pop can",
    "pepsi can": "pepsi pop can",
    "sponge": "green sponge",
    "plate larger": "plate",
    "table cloth shorter": "blue towel cloth",
    "top drawer": "dresser",
    "middle drawer": "dresser",
    "bottom drawer": "dresser",
    "robot": "robot manipulator",
}

KEEP_OBJECTS: List[str] = ["robot"]  # contrast_image_generator.py:98 (predictor branch)


class InstructionParseError(ValueError):
    pass


def parse_objects(instruction: str, get_all_parts: bool = False) -> List[str]:
    """Objects to mask for an instruction, exactly as instruction_templates.get_objects_from_instruction."""
    text = instruction
    for old, new in REWRITES:
        text = text.replace(old, new)
    for pattern in TEMPLATES:
        found = re.match(pattern, text)
        if found is None:
            continue
        objects = list(found.groups())
        if get_all_parts and any(part in objects for part in DRAWER_PARTS):
            objects = objects + [part for part in DRAWER_PARTS if part not in objects]
        return [obj.strip() for obj in objects]
    raise InstructionParseError(f"Instruction '{text}' does not match any template")


def gdino_text(prompts: Sequence[str], aliases: Dict[str, str]) -> str:
    """Text query: '. '.join(aliases) + '.' -- mask_predictors.py:108-109."""
    return ". ".join(aliases.get(p, p) for p in prompts) + "."


def alias_to_name(prompts: Sequence[str], aliases: Dict[str, str]) -> Dict[str, str]:
    """Reverse alias map restricted to the current prompts; later keys win -- mask_predictors.py:126."""
    out: Dict[str, str] = {}
    for name, alias in aliases.items():
        if name in prompts:
            out[alias] = name
    return out


def label_to_name(label: str, reverse: Dict[str, str]) -> str:
    """GDINO drops the word 'can' from '... pop can' phrases; the code appends it back -- mask_predictors.py:128-132."""
    if label.endswith("pop"):
        label = label + " can"
    return reverse.get(label, label)
