"""H1-BCP (hs_bcp): Boundary Clause Prompt for pi0_fast_libero on LIBERO-10 (rtcfpp-20261101 round 8).

Entry points (one card each, params in the card):
  bcp            mode shadow_probe: the K1 shadow kill test (run only as the NULL arm; native actions executed).
  bcp-switch     mode switch: K2 candidate (clause prompt after the custody boundary). Not run now.
  bcp-switch-dot mode switch_control: K2 churn control (original + ' .' after the boundary). Not run now.
See controller.py and rules.py.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_bcp.controller:BCPController"
ENTRIES = ("bcp", "bcp-switch", "bcp-switch-dot")


def _config(name: str):
    path = Path(__file__).with_name(f"paper_{name}.yaml")
    params = dict(load_card(path)["params"])
    return config_from_card(path, controller=ModuleRef(FACTORY, params))


def config_bcp():
    return _config("bcp")


def config_bcp_switch():
    return _config("bcp-switch")


def config_bcp_switch_dot():
    return _config("bcp-switch-dot")
