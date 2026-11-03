"""bdu unit tests, frozen identities (existing fingerprints unchanged) and BDU-RC2 parity on all 100 greedy records."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import replay  # noqa: E402

from harnessstudio.core import registry  # noqa: E402
from hs_gpp_bdu.bdu import BduMonitor, BduWindow  # noqa: E402

RUN = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/scalepp-20261102"
BD_REPLAY = f"{RUN}/ideation/idea-r3a-evidence/bd_trigger_replay.json"
FROZEN = {"gpp-core": "6c80527fa03e49812fdc723fd56ab8df32fe9dba9a072dfad39d0bd179f4efe4",
          "gpp-core-sham": "31f43fb78fd56ff373106f522efdab0e2d9caa41111afb69a2eb4aad8b893d1d",
          "gpp-h2": "c9be8a65e7215b88ffde02296365519d4259488c511b3572714f775b08256434",
          "gpp-h2-sham": "982a3edb3ca848b73441dfb9d75ad4658b63f23c8c01f56689cc0f7b82f840e8",
          "ge-gpp-core": "91c88e6ff30086cbe5400cb0bb170b1c6054823309e31d9f77cb6658d421f09a",
          "greedy-escape": "f03731f11b84f97306ba5080b1def6b72e64468503a25813f46a700e6f9f9fe1"}
ENTRIES = ("ge-gpp-bdu", "ge-gpp-bdusham", "ge-bdu", "ge-bdusham", "gpp-bdu", "gpp-bdusham")


def press(z, lat=0.1, g=-1.0):
    return np.array([lat, 0.0, -0.5, 0.0, 0.0, 0.0, g], dtype=np.float32)


# ---- identities -------------------------------------------------------------------------------------------------------
def test_existing_fingerprints_unchanged():
    for name, fp in FROZEN.items():
        assert registry.load(name).fingerprint() == fp, name


def test_new_entries_distinct_and_carry_bdu_constants():
    fps = {e: registry.load(e).fingerprint() for e in ENTRIES}
    assert len(set(fps.values())) == 6 and not set(fps.values()) & set(FROZEN.values())
    for e in ENTRIES:
        p = registry.load(e).controller.params
        assert p["bdu"] == {"n": 20, "az": -0.2, "dz": -0.0005, "max_fires": 4, "window": 10, "lift_az": 0.15,
                            "lateral": 0.025}
        assert p["modules"]["bdu"] == ("sham" if e.endswith("sham") else "on") and p["modules"]["h3"] == "off"
        assert "analytic_control" in p["declarations"] and registry.load(e).card["benchmark_params"]["send_state"]
    assert registry.load("ge-bdu").controller.params["modules"]["h1"] == "off"
    assert registry.load("gpp-bdu").controller.params["modules"]["ge"] is False


def test_identity_guard_refuses_a_changed_hs_gpp(monkeypatch):
    import hs_gpp_bdu.controller as C
    monkeypatch.setitem(C.GPP_FILES, "rules.py", "0" * 64)
    with pytest.raises(ValueError, match="rules.py"):
        registry.load("gpp-bdu").controller.build()


# ---- monitor / window -------------------------------------------------------------------------------------------------
def test_monitor_fires_after_20_blocked_steps_and_resets():
    m = BduMonitor()
    assert m.feed(press(1.0), 1.0) is False          # b_0 = False (no previous z)
    for k in range(19):
        m.feed(press(1.0), 1.0 - 0.0004 * (k + 1))   # 0.4 mm/step down: still "blocked" (> -0.5 mm)
        assert not m.ready()
    m.feed(press(1.0), 1.0 - 0.0004 * 20)
    assert m.ready()
    m.fired()
    assert not m.ready() and m.count == 0
    m.feed(press(1.0), 0.99)                          # a 2 mm drop breaks the run
    assert m.count == 0


@pytest.mark.parametrize("a,dz", [(press(1, g=1.0), 0.0), (np.array([0.1, 0, -0.19, 0, 0, 0, -1], np.float32), 0.0),  # float32(-0.2) < -0.2 (as offline)
                                  (press(1), -0.0006)])
def test_monitor_conditions(a, dz):
    m = BduMonitor()
    z = 1.0
    m.feed(a, z)
    for _ in range(25):
        z += dz
        m.feed(a, z)
    assert not m.ready()                              # closed gripper / a_z not < -0.2 / moving down


def test_monitor_ge_window_reset_and_cap():
    m = BduMonitor(max_fires=1)
    m.feed(press(1), 1.0)
    for _ in range(25):
        m.feed(press(1), 1.0, reset=True)
    assert m.count == 0
    for _ in range(20):
        m.feed(press(1), 1.0)
    assert m.ready()
    m.fired()
    for _ in range(25):
        m.feed(press(1), 1.0)
    assert not m.ready()                              # max_fires


def test_window_edit_sham_and_ends():
    w = BduWindow(np.array([0.0, 0.0, 1.0]), 1.0)
    a, ed = w.edit(press(1, lat=0.3))
    assert ed and a[2] == pytest.approx(0.15) and a[0] == pytest.approx(0.3) and a[6] == -1.0
    s = BduWindow(np.array([0.0, 0.0, 1.0]), 1.0, sham=True)
    a, ed = s.edit(press(1, lat=0.3))
    assert ed and a[0] == 0.0 and a[1] == 0.0 and a[2] == pytest.approx(0.15)
    a, ed = w.edit(press(1, g=1.0))
    assert not ed and a[2] == pytest.approx(-0.5)     # close passes through natively
    w.after(False, np.array([0.0, 0.0, 1.0]))
    assert w.outcome == "close_passthrough"
    w = BduWindow(np.array([0.0, 0.0, 1.0]), 1.0)
    for k in range(10):
        w.edit(press(1))
        w.after(True, np.array([0.003 * (k + 1), 0.0, 1.0]))
        if w.outcome:
            break
    assert w.outcome == "lateral_done" and w.steps == 9 and w.info()["cos_realized_vs_commanded"] == pytest.approx(1.0)
    w = BduWindow(np.array([0.0, 0.0, 1.0]), 1.0)
    for k in range(12):
        w.edit(press(1))
        w.after(True, np.array([0.0, 0.0, 1.0]))
        if w.outcome:
            break
    assert w.outcome == "max_steps" and w.steps == 10 and w.edited == 10


# ---- BDU-RC2: online parity on all 100 greedy map records -------------------------------------------------------------
pytestmark_map = pytest.mark.skipif(not os.path.isdir(str(replay.SHARDS)), reason="development map not on this machine")


@pytestmark_map
def test_bdu_online_fires_equal_replay_N20_on_all_100():
    ref = json.load(open(BD_REPLAY))["N20"]["greedy"]
    expected = {r["scene"]: r["fire"] for r in ref["F"] + ref["S"]}
    assert len(expected) == 14
    tab = replay.first_acting_table("gpp-bdu", modules={"h1": "off", "h2": "off"})
    got = {s: v["act"]["trigger_at"] for s, v in tab.items() if v["act"] is not None}
    assert all(v["act"]["module"] == "bdu" for v in tab.values() if v["act"])
    assert got == expected
    assert all(tab[s]["act"]["acts_at"] == expected[s] + 1 for s in expected)
