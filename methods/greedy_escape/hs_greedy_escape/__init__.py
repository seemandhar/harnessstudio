"""greedy-escape (hs_greedy_escape): native greedy OpenVLA decoding by default; a public stall trigger (15 identical
near-zero executed actions, or the 3rd empty close since the last window) opens a bounded 40-decision escape window
decoded by a frozen server-side operator (default SCALE, hs_scale.decoder:ScaleDecoder at the paper defaults), then
greedy again. See controller.py (episode loop) and monitor.py (frozen trigger rules).

Approved proposal: runs/scalepp-20261102/ideation/idea-r1g.json (greedy-escape); review
runs/scalepp-20261102/reviews/proposals-r1-review.json (GE1-GE7). The escape operator is the frozen `escape`
parameter of the card (GE2): another operator (e.g. a later scale-gtok decoder) is a NEW card/entry point and a new
frozen candidate, never an edit of this one.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_greedy_escape.controller:GreedyEscapeController"
CARD = Path(__file__).with_name("paper.yaml")


def config():
    params = dict(load_card(CARD)["params"])
    return config_from_card(CARD, controller=ModuleRef(FACTORY, params))
