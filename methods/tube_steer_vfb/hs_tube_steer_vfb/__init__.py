"""tube-steer-vfb (hs_tube_steer_vfb): tube-steer plus a SigLIP visual retrieval fallback for instructions without
success memory (see action_op.py). Self-contained copy of hs_tube_steer 1.0.0 (not imported).

Entry points: ``tube-steer-vfb`` (TV) and ``tube-steer-vfb-ortho`` (DEVELOPMENT-ONLY control, duration-capped).
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_tube_steer_vfb.action_op:TubeSteerActionOp"
ENTRIES = ("tube-steer-vfb", "tube-steer-vfb-ortho")


def _config(card: str):
    path = Path(__file__).with_name(card)
    params = dict(load_card(path)["params"])
    return config_from_card(path, slots={"action_op": ModuleRef(FACTORY, params)})


def config():
    return _config("paper.yaml")


def config_ortho():
    return _config("paper_ortho.yaml")
