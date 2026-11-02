"""vlm-rehome: VLM-flagged post-boundary home re-entry (VPR), round-4 PERCEPTION P1 of run rtcfpp-20261101, as a
HarnessStudio v2 controller method (original RoboMetaHarness mechanism, not a paper port).

Entry points (cards in this directory; every scalar lives in the card's ``params`` block):
- ``vlm-rehome``           candidate: re-home when the frozen NRP VLM judge says 'abnormal' after a subtask boundary.
- ``vlm-rehome-tiltgate``  admission control (critic C-V5): same detector, schedule and VLM calls (paid, ignored);
                           re-home iff decision-time tilt > 55 deg with the gripper empty.
- ``vlm-rehome-forced``    development-only diagnostic (critic C-V6 forced probe): re-home at the first boundary check.

``hs_vlm_rehome.judge`` holds the frozen operator pin, prompt, image encoding, payload and verdict parser shared by
the controller and the Q1 qualification harness (scripts/q1_qualify.py).
"""
from __future__ import annotations

from pathlib import Path

FACTORY = "hs_vlm_rehome.controller:VLMRehome"


def _config(card: str):
    from harnessstudio.core.card import config_from_card, load_card
    from harnessstudio.core.method import ModuleRef
    path = Path(__file__).with_name(card)
    params = dict(load_card(path)["params"])
    return config_from_card(path, controller=ModuleRef(FACTORY, params))


def config():
    return _config("paper.yaml")


def config_tiltgate():
    return _config("paper_tiltgate.yaml")


def config_forced():
    return _config("paper_forced.yaml")
