"""fork30 (hs_fork30): native greedy OpenVLA; at decision 30 a frozen public repeat cue (>= 6 exact repeats of the
executed action in steps 1-29) switches decoding permanently to SCALE (hs_scale, read-only, identity-guarded).
Review runs/scalepp-20261102/reviews/fork-r4-review.json. See controller.py.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_fork30.controller:Fork30Controller"
CARD = Path(__file__).with_name("paper.yaml")


def config():
    return config_from_card(CARD, controller=ModuleRef(FACTORY, dict(load_card(CARD)["params"])))
