"""Selectable instruction parsers.

"simpler" (default) is the authors' parser and vocabulary exactly (vocab.py; Koorye/PCD@31b027e). Other modes are
frozen data files in this directory, loaded only when selected at /reset and refused unless their sha256 matches the
pin below (Setting B's LIBERO table must not drift after it is frozen).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .. import vocab

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = {"libero_goal": ("libero_goal.json", "eb67f16396877052eb17654b5a1fabe3ffa59cfd362dc9ece400981fc31724e8")}


@dataclass(frozen=True)
class ParserSpec:
    name: str
    templates: Tuple[str, ...]
    rewrites: Tuple[Tuple[str, str], ...]
    classnames: Tuple[str, ...]
    aliases: Dict[str, str] = field(hash=False)
    keep_objects: Tuple[str, ...] = ("robot",)
    sha256: str = ""

    def parse(self, instruction: str) -> List[str]:
        text = instruction
        for old, new in self.rewrites:
            text = text.replace(old, new)
        for pattern in self.templates:
            found = re.match(pattern, text)
            if found is not None:
                return [g.strip() for g in found.groups()]
        raise vocab.InstructionParseError(f"Instruction '{text}' does not match any template ({self.name})")


SIMPLER = "simpler"


def load(name: str) -> ParserSpec:
    if name not in FROZEN:
        raise ValueError(f"unknown parser '{name}' (known: {[SIMPLER] + sorted(FROZEN)})")
    fname, pin = FROZEN[name]
    path = os.path.join(HERE, fname)
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin:
        raise RuntimeError(f"parser table {fname} sha256 {digest} != frozen pin {pin}")
    d = json.loads(raw)
    return ParserSpec(name=d["name"], templates=tuple(d["templates"]),
                      rewrites=tuple(tuple(r) for r in d["rewrites"]), classnames=tuple(d["classnames"]),
                      aliases=dict(d["aliases"]), keep_objects=tuple(d["keep_objects"]), sha256=digest)
