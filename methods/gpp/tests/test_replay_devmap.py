"""H1-RC1 / H2-RC1 tier-1 parity on all 100 recorded greedy development episodes (libero_10 tasks 0-9 x init 0-9; never
init >= 10). The online rules (scripts/fire_set.py -> hs_gpp.rules, the controller's code) must give exactly the
critic's fire set F_gpp with H1 fires equal to the offline replay and the first H2 fire at max(c+2, offline step).
Outcomes are not read."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import fire_set  # noqa: E402

RUN = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/scalepp-20261102"
EVID = f"{RUN}/ideation/idea-r2b-evidence"
CRITIC_F = ["t2i0", "t2i7", "t2i8", "t3i3", "t3i4", "t3i5", "t3i6", "t3i7", "t6i4", "t7i1", "t7i2", "t7i4", "t8i1",
            "t8i2", "t8i4", "t8i8", "t9i9"]

pytestmark = pytest.mark.skipif(not os.path.isdir(str(fire_set.SHARDS)), reason="development map not on this machine")


@pytest.fixture(scope="module")
def table():
    return fire_set.fire_table("gpp-core")


def test_fire_set_equals_the_critics(table):
    assert table["fire_set"] == CRITIC_F
    first = {s: (v["first"]["module"], v["first"]["trigger_at"]) for s, v in table["scenes"].items() if v["first"]}
    assert first["t8i8"][0] == "h2" and first["t8i2"][0] == "h1"      # review: H2 first in t8i8, H1 first in t8i2
    assert first["t8i2"] == ("h1", 212) and first["t3i6"] == ("h2", 117) and first["t7i1"] == ("h2", 66)


def test_online_equals_offline_replay(table):
    h1 = json.load(open(f"{EVID}/h1_trigger_replay.json"))["per_episode"]
    h2 = json.load(open(f"{EVID}/h2_trigger_replay.json"))["per_episode"]
    for s, v in table["scenes"].items():
        assert [f["trigger_at"] for f in v["h1"]] == [f["step"] for f in h1[f"greedy:{s}"]["fires"]], s
        off = h2[f"greedy:{s}"]["fires"]
        on = v["h2"]
        assert bool(on) == bool(off), s
        if off:
            assert on[0]["trigger_at"] == max(off[0]["close"] + 2, off[0]["step"]) and on[0]["close"] == off[0]["close"]


def test_h2_only_fallback_fire_set(table):
    t2 = fire_set.fire_table("gpp-h2")
    assert t2["fire_set"] == sorted({s for s, v in table["scenes"].items() if v["h2"]},
                                    key=lambda s: (int(s[1]), int(s[3:])))
    assert len(t2["fire_set"]) == 11
