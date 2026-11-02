"""tilt-guard: composite action_op for pi0_fast_libero on LIBERO (original RoboMetaHarness method, run rtcfpp-20261101,
round 1): Z decode-failure substitution -> G saturation-gated rx band guard (hs_band_guard, imported read-only) -> B
boundary tilt envelope, each switchable by a frozen card parameter (see action_op.py for the exact rules).

Entry points (one card each, every scalar frozen in the card's ``params`` block):
  tilt-guard-gb        G + B                          (screen wave 2)
  tilt-guard-gyaw      G + yaw-matched control of B   (DEVELOPMENT-ONLY control, domain review C9)
  tilt-guard-gbz       G + B + Z                      (screen wave 2)
  tilt-guard-gz        G + Z                          (selection rule when B is dropped)
  tilt-guard-gbzsign   G + B + Z sign control         (DEVELOPMENT-ONLY, conditional wave)
  tilt-guard-gzsign    G + Z sign control             (DEVELOPMENT-ONLY, conditional wave if B is dropped)
  tilt-guard-z         Z alone                        (general review C8 standalone rule)
  tilt-guard-zsign     Z sign control alone           (DEVELOPMENT-ONLY, general P2 F3)
G alone is the ``band-guard`` entry point of hs_band_guard.

Pure API: hs_tilt_guard.action_op.{TiltGuardActionOp, BoundaryDetector, is_openpi_zero_chunk, yaw_coin, zero_coins},
hs_tilt_guard.geometry.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card, load_card
from harnessstudio.core.method import ModuleRef

FACTORY = "hs_tilt_guard.action_op:TiltGuardActionOp"
ENTRIES = ("gb", "gyaw", "gbz", "gz", "gbzsign", "gzsign", "z", "zsign")


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
