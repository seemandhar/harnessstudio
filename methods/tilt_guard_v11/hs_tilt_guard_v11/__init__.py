"""tilt-guard v1.1 (hs_tilt_guard_v11): the frozen tilt-guard v1.0 composite (Z -> G -> B) plus the r2 saturation-arming
flag for B (see action_op.py). With saturation_arming='off' outputs are bitwise identical to v1.0.

Entry points (one card each, every scalar frozen in the card's ``params`` block):
  tilt-guard-v11-gb, -gyaw, -gbz, -gz, -gbzsign, -gzsign, -z, -zsign   v1.0 variants, saturation_arming='off'
  tilt-guard-v11-gbsatz      G + Bsat + Z
  tilt-guard-v11-gbsat       G + Bsat
  tilt-guard-v11-gbsatz-yaw  G + B + yaw-sat + Z  (DEVELOPMENT-ONLY control)
  tilt-guard-v11-gbsat-yaw   G + B + yaw-sat      (DEVELOPMENT-ONLY control)
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_tilt_guard_v11.action_op:TiltGuardActionOp"
ENTRIES = ("gb", "gyaw", "gbz", "gz", "gbzsign", "gzsign", "z", "zsign", "gbsatz", "gbsat", "gbsatz-yaw", "gbsat-yaw")


def _config(variant: str):
    path = Path(__file__).with_name(f"paper_{variant}.yaml")
    params = dict(load_card(path)["params"])
    return config_from_card(path, slots={"action_op": ModuleRef(FACTORY, params)})


def config_gb():
    return _config("gb")


def config_gyaw():
    return _config("gyaw")


def config_gbz():
    return _config("gbz")


def config_gz():
    return _config("gz")


def config_gbzsign():
    return _config("gbzsign")


def config_gzsign():
    return _config("gzsign")


def config_z():
    return _config("z")


def config_zsign():
    return _config("zsign")


def config_gbsatz():
    return _config("gbsatz")


def config_gbsat():
    return _config("gbsat")


def config_gbsatz_yaw():
    return _config("gbsatz-yaw")


def config_gbsat_yaw():
    return _config("gbsat-yaw")
