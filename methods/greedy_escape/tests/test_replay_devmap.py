"""Replay of the recorded development map (scalepp-20261102 eval/devbase, libero_10 tasks 0-9 x init 0-9 ONLY, env
seed 7, namespace '') through the frozen StallMonitor.

1. On every greedy record the monitor's first fire equals the proposal's offline rule (ideation/idea-r1g-evidence
   trig2.json, W 15 / eps 0.01 / n_empty 3): the online implementation of GE1 is the frozen offline rule.
2. The CRN seed of decision d of the SCALE map record equals crn_seed('hs', 'libero', task, init, d, 0): a window
   decision d of greedy-escape (one sample per decision) sends the seed the SCALE map used at d (GE3).
Development evidence only; outcomes are not read (trig2.json's success flags are ignored).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pytest

from harnessstudio.core.noise import crn_seed

RUN = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/scalepp-20261102"
SHARDS = f"{RUN}/eval/devbase/shards"
TRIG2 = f"{RUN}/ideation/idea-r1g-evidence/trig2.json"
INITS = range(10)     # the development map; never libero_10 init 10-49

pytestmark = pytest.mark.skipif(not os.path.isdir(SHARDS), reason="development map not on this machine")


def greedy_rows(t, i):
    d = f"{SHARDS}/baseline-t{t}/baseline/vla_eval/episodes/LIBEROBenchmark_libero_10"
    fs = glob.glob(f"{d}/task0000_ep{i:04d}_*.jsonl")
    assert len(fs) == 1, fs
    rows = [json.loads(line) for line in open(fs[0])]
    return np.array([r["action"] for r in rows], dtype=np.float32), np.array([r["state"] for r in rows], dtype=np.float32)


# The SAME replay function screen_summary.py uses for its S3 trigger-parity expectation.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from screen_summary import first_fire  # noqa: E402


def test_online_first_fire_equals_offline_rule_on_all_100_greedy_map_records():
    trig2 = json.load(open(TRIG2))
    mismatches, fired = [], {}
    for t in range(10):
        for i in INITS:
            A, S = greedy_rows(t, i)
            step, reasons = first_fire(A, S)
            _, _, off, n = trig2[f"t{t}i{i}"]
            off = off if (off is not None and off < n) else None
            if step != off:
                mismatches.append((t, i, step, off))
            if step is not None:
                fired[f"t{t}i{i}"] = (step, reasons)
    assert not mismatches, mismatches
    assert len(fired) == 30          # 27 map greedy failures + 3 successes (proposal: 27/46 and 3/54)


def grid_fires():
    out = {}
    for t in range(10):
        for i in range(4):
            A, S = greedy_rows(t, i)
            step, reasons = first_fire(A, S)
            if step is not None:
                out[f"t{t}i{i}"] = {"step": step, "reasons": reasons}
    return out


def test_grid_trigger_list_matches_the_proposal():
    fires = grid_fires()
    expected = {"t0i1": 70, "t2i2": 98, "t3i0": 108, "t7i2": 42, "t7i0": 175, "t0i0": 278, "t3i3": 211, "t4i3": 247,
                "t2i0": 475, "t8i2": 511}
    assert {k: v["step"] for k, v in fires.items()} == expected


def test_window_seed_equals_scale_map_seed():
    checked = 0
    for t in (0, 3, 7):
        for line in open(f"{SHARDS}/candidate-t{t}/candidate/records.jsonl"):
            r = json.loads(line)
            if r["episode_idx"] not in INITS:
                continue
            assert r["benchmark"] == "libero"
            for ev in r["events"]:
                if ev["t"] == "sample":
                    assert ev["seeds"] == [crn_seed("hs", r["benchmark"], r["task"], r["episode_idx"], ev["decision"], 0)]
                    checked += 1
    assert checked > 1000
