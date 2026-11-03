"""gpp-bdu (hs_gpp_bdu): the frozen gpp v1.1 controller (hs_gpp, read-only, identity-guarded) plus the bdu
blocked-descent unload module. Entries: ge-gpp-bdu / ge-gpp-bdusham (primary vehicle), ge-bdu / ge-bdusham,
gpp-bdu / gpp-bdusham (review proposals-r3-review.json R3-SEL vehicle rule). See controller.py and bdu.py.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_gpp_bdu.controller:GppBduController"


def _config(entry: str):
    path = Path(__file__).with_name(f"paper_{entry}.yaml")
    return config_from_card(path, controller=ModuleRef(FACTORY, dict(load_card(path)["params"])))


def config_ge_gpp_bdu():
    return _config("ge_gpp_bdu")


def config_ge_gpp_bdusham():
    return _config("ge_gpp_bdusham")


def config_ge_bdu():
    return _config("ge_bdu")


def config_ge_bdusham():
    return _config("ge_bdusham")


def config_gpp_bdu():
    return _config("gpp_bdu")


def config_gpp_bdusham():
    return _config("gpp_bdusham")
