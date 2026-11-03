#!/usr/bin/env python3
"""fork30 first-fire replay (review F3): the REAL frozen controller (registry fork30, hs_scale identity guard active)
runs observe-only on the 100 map greedy records (libero_10 tasks 0-9 x init 0-9; per-step jsonl action rows only, no
outcome field). A replay context serves each recorded greedy action; the fork is the first decision that requests the
SCALE sampler (decision 30) and is cross-checked against the logged ctx.record("fork"). Output: per scene repeats,
fired, fork decision; the sorted fire set and its sha256.

    harnessstudio/.venv/bin/python harnessstudio/methods/fork30/scripts/fire_set.py --out FIRE_SET.json
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

SHARDS = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/scalepp-20261102/eval/devbase/shards")


class _Stop(Exception):
    pass


class _P:
    def __init__(self, a):
        self.actions = np.asarray(a, dtype=np.float32).reshape(1, -1)
        self.extras = {}


class _Acc:
    def __init__(self):
        self.extra = {}


class ReplayCtx:
    def __init__(self, actions):
        self.A = actions
        self.k = 0
        self.accounting = _Acc()
        self.events = []
        self.fork_at = None
        outer = self

        class Policy:
            async def sample(self, k=1, obs=None, instruction=None, sampler=None, readouts=()):
                if sampler is not None:
                    outer.fork_at = outer.k
                    raise _Stop
                return [_P(outer.A[outer.k])]
        self.policy = Policy()

    @property
    def ended(self):
        return self.k >= len(self.A)

    def observe(self):
        return {}

    def record(self, event, **fields):
        self.events.append({"name": event, **fields})

    async def execute(self, chunk):
        if not np.array_equal(np.asarray(chunk, dtype=np.float32).reshape(-1), np.asarray(self.A[self.k], np.float32)):
            raise AssertionError("replay: executed action differs from the record")
        self.k += 1
        return 1


def greedy_rows():
    rows, hashes = {}, {}
    for t in range(10):
        for i in range(10):                      # development map only; never init >= 10
            fs = glob.glob(str(SHARDS / f"baseline-t{t}/baseline/vla_eval/episodes/LIBEROBenchmark_libero_10/"
                                         f"task0000_ep{i:04d}_*.jsonl"))
            assert len(fs) == 1, fs
            raw = open(fs[0], "rb").read()
            rows[f"t{t}i{i}"] = [json.loads(line)["action"] for line in raw.splitlines() if line.strip()]
            hashes[fs[0]] = hashlib.sha256(raw).hexdigest()
    return rows, hashes


def replay_one(controller, actions) -> dict:
    ctx = ReplayCtx(actions)
    try:
        asyncio.run(controller.run(ctx))
    except _Stop:
        pass
    ev = [e for e in ctx.events if e["name"] == "fork"]
    s = ctx.accounting.extra.get("fork30") or {}
    return {"fork_at": ctx.fork_at, "repeats": s.get("repeats"), "fired": bool(ev and ev[0]["fired"]),
            "logged": ev[0] if ev else None, "steps": len(actions)}


def fire_table(entry: str = "fork30") -> dict:
    from harnessstudio.core import registry
    cfg = registry.load(entry)
    ctrl = cfg.controller.build()
    rows, hashes = greedy_rows()
    scenes = {s: replay_one(ctrl, A) for s, A in rows.items()}
    fset = sorted((s for s, v in scenes.items() if v["fired"]), key=lambda s: (int(s[1]), int(s[3:])))
    return {"entry": entry, "fingerprint": cfg.fingerprint(),
            "rule": "R = #{k in 1..29: executed a[k] == a[k-1]}; fork at decision 30 iff R >= 6",
            "fire_set": fset, "scenes": scenes, "greedy_jsonl_sha256": hashes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t = fire_table()
    text = json.dumps(t, indent=1, sort_keys=True)
    Path(args.out).write_text(text)
    print(len(t["fire_set"]), t["fire_set"], hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
