"""pcd (hs_pcd): Policy Contrastive Decoding (arXiv:2505.13255, Koorye/PCD @ 31b027e) for OpenVLA as a HarnessStudio
v2 `null: shadow` controller on the EXT openvla-contrast `logits` readout (docs/extensions/openvla-contrast.md).

See controller.py (per-step loop), combine.py (the code-faithful combine and fallbacks), perception.py (service
client + tier-1 stub) and paper.yaml (settings, substitutions, reported numbers). The perception service itself is
service/ (separate venv, separate owner).
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_pcd.controller:PCDController"
CARD = Path(__file__).with_name("paper.yaml")


def config():
    """Setting A (and the default): the service's SIMPLER parser (the authors' table)."""
    params = dict(load_card(CARD)["params"])
    return config_from_card(CARD, controller=ModuleRef(FACTORY, params))


def config_libero_goal():
    """Setting B: the same method with the service's frozen LIBERO-Goal parser (SETTING.md Addendum 4)."""
    import dataclasses
    base = config()
    params = {**base.controller.params, "perception_parser": "libero_goal"}
    return dataclasses.replace(base, name="pcd-libero-goal", controller=ModuleRef(FACTORY, params))
