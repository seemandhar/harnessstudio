"""band-guard: saturation-gated rx band reconstruction for pi0_fast_libero (original RoboMetaHarness method,
run rtcfpp-20261101, proposal r1g-p1) as a HarnessStudio v2 slot method with one action_op.

Entry points:
- ``band-guard``           candidate: a'_rx = m_rx + clip(s_rx, state q01, state q99) when gated (see action_op.py).
- ``band-guard-signctrl``  DEVELOPMENT-ONLY norm-matched sign control (same gate, same magnitude, public-coin sign).

Every scalar lives in the card's ``params`` block (frozen, critic C1); the op refuses band/saturation edges that differ
from the pinned checkpoint norm_stats. Pure API for tests/replay: ``hs_band_guard.action_op.BandGuardActionOp``
(``gate``, ``transform``), ``is_openpi_zero_chunk``, ``public_coin``.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_band_guard.action_op:BandGuardActionOp"


def _config(card: str):
    path = Path(__file__).with_name(card)
    params = dict(load_card(path)["params"])
    return config_from_card(path, slots={"action_op": ModuleRef(FACTORY, params)})


def config():
    return _config("paper.yaml")


def config_signctrl():
    return _config("paper_signctrl.yaml")
