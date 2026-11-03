#!/usr/bin/env python3
"""Observe-only first-acting replay of a gpp-bdu controller on the 100 map greedy records (development map, libero_10
tasks 0-9 x init 0-9 ONLY; per-step jsonl rows, no outcome field read). Same harness as eval/ge-gpp/a5/a5_replay.py:
the REAL controller (built from a registry card, identity guards active) runs on a replay context that serves each
recorded greedy action and post-step `states`; the first decision departing from the recorded action is the first
acting module: a GE window request ("ge"), a gpp primitive fire record ("h1"/"h2"/"h3") or a bdu fire record ("bdu").

    from replay import first_acting_table
    table = first_acting_table("ge-gpp-bdu")                    # or a MethodConfig-like override via modules=...
"""
from __future__ import annotations

import asyncio
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

SHARDS = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/scalepp-20261102/eval/devbase/shards")


class _Stop(Exception):
    pass


class _Proposal:
    def __init__(self, a):
        self.actions = np.asarray(a, dtype=np.float32).reshape(1, -1)
        self.extras = {}


class _Acc:
    def __init__(self):
        self.extra = {}


class ReplayCtx:
    def __init__(self, actions, states):
        self.A, self.S = actions, states
        self.k = 0
        self.accounting = _Acc()
        self.memory = {}
        self.act = None
        outer = self

        class P:
            async def sample(self, k=1, obs=None, instruction=None, sampler=None, readouts=()):
                if sampler is not None:
                    outer.act = {"module": "ge", "acts_at": outer.k}
                    raise _Stop
                return [_Proposal(outer.A[outer.k])]
        self.policy = P()

    @property
    def ended(self):
        return self.k >= len(self.A)

    def observe(self):
        return {"states": None if self.k == 0 else list(self.S[self.k - 1])}

    def record(self, event, **fields):
        if fields.get("phase") == "fire" and self.act is None and event in ("gpp", "bdu"):
            module = fields.get("module", "bdu") if event == "gpp" else "bdu"
            self.act = {"module": module, "acts_at": fields["start_at"], "trigger_at": fields["trigger_at"]}
            raise _Stop

    async def execute(self, chunk):
        a = np.asarray(chunk).reshape(-1)
        if not np.array_equal(a.astype(np.float32), np.asarray(self.A[self.k], dtype=np.float32)):
            raise AssertionError(f"pre-fire action at step {self.k} differs from the record without a fire")
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
            recs = [json.loads(line) for line in raw.splitlines() if line.strip()]
            rows[f"t{t}i{i}"] = ([r["action"] for r in recs], [r["state"] for r in recs])
            hashes[fs[0]] = hashlib.sha256(raw).hexdigest()
    return rows, hashes


def build(entry: str, modules=None):
    from harnessstudio.core import registry
    from harnessstudio.core.method import ModuleRef
    cfg = registry.load(entry)
    params = json.loads(json.dumps(cfg.controller.params))
    if modules:
        params["modules"].update(modules)
    return cfg, ModuleRef(cfg.controller.factory, params).build()


def replay_one(controller, actions, states) -> dict:
    ctx = ReplayCtx(actions, states)
    try:
        asyncio.run(controller.run(ctx))
    except _Stop:
        pass
    out = {"act": ctx.act, "steps": len(actions)}
    g = (ctx.accounting.extra.get("gpp") or {}).get("ge") or {}
    if g.get("fires"):
        out["ge_first_fire"] = {"step": g["fires"][0]["step"], "reasons": list(g["fires"][0]["reasons"])}
        if ctx.act and ctx.act["module"] == "ge":
            ctx.act["trigger_at"] = g["fires"][0]["step"]
            ctx.act["reasons"] = list(g["fires"][0]["reasons"])
    if ctx.act and ctx.act["module"] != "ge" and g.get("fires") and g["fires"][0]["step"] + 1 == ctx.act["acts_at"]:
        ctx.act["concurrent_ge_fire"] = out["ge_first_fire"]
    return out


def first_acting_table(entry: str, modules=None, rows=None) -> dict:
    rows = rows if rows is not None else greedy_rows()[0]
    _, ctrl = build(entry, modules)
    return {s: replay_one(ctrl, A, S) for s, (A, S) in rows.items()}


def scene_key(s: str):
    return (int(s[1]), int(s[3:]))
