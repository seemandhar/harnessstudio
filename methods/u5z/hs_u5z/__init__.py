"""U5+Z (hs_u5z): the single A6.Z composition of rtcfpp-20261101 -- frozen tube-steer (U5 = T) with the frozen tilt-guard
v1.2 'z' zero-chunk substitution acting only before U5's first arming (glue.py). Components are the installed frozen
packages hs-tube-steer 1.0.0, hs-tilt-guard-v12 1.2.0 (Z) and hs-tilt-guard-v11 1.1.0 (control 'zsign'), sha256-verified
at construction and never edited.

Entry points (one card each; params = the frozen component cards' params, copied by scripts/make_cards.py and checked for
equality at construction):
  u5z-t       U5 = T (tube-steer) + Z (tilt-guard v1.2 'z')
  u5zsign-t   U5 = T + Zsign (tilt-guard v1.1 'zsign', the Z control)      (DEVELOPMENT-ONLY control)
The -tv entries (U5 = tube-steer-vfb) are not defined: see README.md (used only if A5.3(5) makes U5 = TV).
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_u5z.glue:U5ZActionOp"
ENTRIES = ("u5z-t", "u5zsign-t")


def _config(name: str):
    path = Path(__file__).with_name(f"paper_{name}.yaml")
    params = dict(load_card(path)["params"])
    return config_from_card(path, slots={"action_op": ModuleRef(FACTORY, params)})


def config_u5z_t():
    return _config("u5z-t")


def config_u5zsign_t():
    return _config("u5zsign-t")
