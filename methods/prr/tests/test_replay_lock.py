"""C3 lock: the package's planner, re-run over the 80 recorded dev identity episodes (streamed one at a time), reproduces
the locked table (implementation/prr/tables/c3-summary.json) and the replay semantics (0 compat mismatches)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
LOCK = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtcfpp-20261101/implementation/prr/tables/c3-summary.json")


@pytest.mark.skipif(not LOCK.exists(), reason="locked C3 table not present on this host")
def test_dev80_table_matches_lock_and_replay():
    sys.path.insert(0, str(SCRIPTS))
    import replay_table as rt
    params = rt.params_from_card()
    rows, compat = rt.dev_table(params)
    lock = json.loads(LOCK.read_text())
    got = {r["key"]: {k: r["first_restoration"][k] for k in ("route", "fire", "checkpoint", "L", "latch", "holding",
                                                              "width_at_fire", "native_steps_left_min")}
           for r in rows if r["activated"]}
    assert got == lock["first_restorations_dev80"]
    assert [c for c in compat if not (c["A"]["equal"] and c["B"]["equal"])] == []
    assert rt.lists(rows) == lock["lists"]
    assert not [r for r in rows if r["activated"] and r["identity_success"]]          # 0/45 exposure (in-sample)
