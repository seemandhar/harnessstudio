"""tilt-guard v1.2 (hs_tilt_guard_v12): tilt-guard v1.1 with G removed from the composite (order Z -> B). B, Z and the
Bsat saturation-arming code are byte-identical to v1.1 (action_op.py, geometry.py); composite.py forces guard=False.
No hs_band_guard import (amendment A2.1).

Entry points (one card each, every scalar frozen in the card's ``params`` block):
  tilt-guard-v12-b         B
  tilt-guard-v12-b-yaw     yaw-matched control at B's active decisions   (DEVELOPMENT-ONLY)
  tilt-guard-v12-bz        B + Z
  tilt-guard-v12-bsat      Bsat
  tilt-guard-v12-bsatz     Bsat + Z
  tilt-guard-v12-bsat-yaw  B + yaw-sat control                           (DEVELOPMENT-ONLY)
  tilt-guard-v12-z         Z
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_tilt_guard_v12.composite:TiltGuardV12ActionOp"
ENTRIES = ("b", "b-yaw", "bz", "bsat", "bsatz", "bsat-yaw", "z")


def _config(variant: str):
    path = Path(__file__).with_name(f"paper_{variant}.yaml")
    params = dict(load_card(path)["params"])
    return config_from_card(path, slots={"action_op": ModuleRef(FACTORY, params)})


def config_b():
    return _config("b")


def config_b_yaw():
    return _config("b-yaw")


def config_bz():
    return _config("bz")


def config_bsat():
    return _config("bsat")


def config_bsatz():
    return _config("bsatz")


def config_bsat_yaw():
    return _config("bsat-yaw")


def config_z():
    return _config("z")
