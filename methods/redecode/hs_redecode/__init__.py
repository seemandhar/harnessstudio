"""ReDecode (hs_redecode): on an openpi FAST decode failure (zero chunk), re-query the frozen pi0_fast_libero with the
public state shifted by one state-token bin (+, then -), execute the first valid decode re-based to the true state, and
fall back to the frozen tilt-guard v1.2 Z rule; the selected chunk passes through the frozen TiltGuardV12ActionOp
(hs_tilt_guard_v12, read-only). See controller.py.

Entry points: ``redecode`` (ReDecode -> Z) and ``redecode-b`` (ReDecode -> Z -> B). Controls are the frozen v1.2 entries
``tilt-guard-v12-z`` and ``tilt-guard-v12-bz`` (amendment A4.1).
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_redecode.controller:ReDecodeController"
ENTRIES = ("redecode", "redecode-b")


def _config(variant: str):
    path = Path(__file__).with_name(f"paper_{variant}.yaml")
    params = dict(load_card(path)["params"])
    return config_from_card(path, controller=ModuleRef(FACTORY, params))


def config_redecode():
    return _config("redecode")


def config_redecode_b():
    return _config("redecode-b")
