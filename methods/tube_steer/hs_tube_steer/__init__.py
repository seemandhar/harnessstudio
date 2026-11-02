"""tube-steer (hs_tube_steer): success-tube orientation steering for pi0_fast_libero (see action_op.py).

Entry points: ``tube-steer`` (candidate) and ``tube-steer-ortho`` (DEVELOPMENT-ONLY control, duration-capped).
Self-contained: imports no other method package.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_tube_steer.action_op:TubeSteerActionOp"
ENTRIES = ("tube-steer", "tube-steer-ortho")


def _config(card: str):
    path = Path(__file__).with_name(card)
    params = dict(load_card(path)["params"])
    return config_from_card(path, slots={"action_op": ModuleRef(FACTORY, params)})


def config():
    return _config("paper.yaml")


def config_ortho():
    return _config("paper_ortho.yaml")
