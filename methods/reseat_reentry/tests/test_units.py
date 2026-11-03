"""Unit tests: K1 trigger-replay parity on the saved development traces, gap-trigger rules, hover target geometry,
re-seat primitive (never closes, <= 45 steps), planner schema/validation/scope rule, prompt hygiene, cards (frozen
scalars, send_state, declared operator), sampler identity with plain scale, static audit, freeze-closure tokens."""
from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_fake_bench import OBJS, TOP, TRAY, RGBDTwoBlockBench, pixel_of  # noqa: E402

from hs_reseat_reentry import controller as C  # noqa: E402
from hs_reseat_reentry import motion as M  # noqa: E402
from hs_reseat_reentry import perception as P  # noqa: E402
from hs_reseat_reentry import planner as PL  # noqa: E402
from hs_reseat_reentry import triggers_gap as T  # noqa: E402

ROOT = "/home/sejain/research/cvpr27/RoboMetaHarness-claude"
SEED9 = ROOT + "/runs/scaleagent-codex-20261103/eval/trigger-precision/runs"
DEVBASE = ROOT + "/runs/scalepp-20261102/eval/devbase"
PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hs_reseat_reentry")


# ---- K1 parity on saved development traces --------------------------------------------------------------------------
def _c_script_first_fire(S, succ, steps, W=140, MINLEFT=150):
    """Scientist C's gap.py logic (non-causal: uses the start row of a later carry), for comparison only."""
    c = T.carries(S)
    for j, (s, en, how, det) in enumerate(c):
        if en is None:
            continue
        nxt = c[j + 1][0] if j + 1 < len(c) else 10 ** 9
        k = en + W
        if nxt > k and k <= 520 - MINLEFT and not (succ and steps <= k):
            return k
    return None


def _load(files):
    out = {}
    for f in files:
        rows = [json.loads(line) for line in open(f)]
        out[f] = (np.array([r["state"] for r in rows], dtype=np.float64), f.endswith("_success.jsonl"), len(rows))
    return out


@pytest.mark.skipif(not os.path.isdir(SEED9), reason="seed-9 trigger-precision traces not present")
def test_k1_seed9_counts_and_fire_steps():
    eps = _load(sorted(glob.glob(SEED9 + "/resp-t*/candidate/vla_eval/episodes/*/*.jsonl")))
    assert len(eps) == 50
    fires = {}
    for f, (S, succ, n) in eps.items():
        r = T.replay(S, 520)
        if r:
            t = int(re.search(r"resp-t(\d+)", f).group(1))
            e = int(re.search(r"_ep(\d+)_", f).group(1))
            fires[f"t{t}e{e}"] = (r[0]["step"], succ)
        assert _c_script_first_fire(S, succ, n) == (r[0]["step"] - r[0]["deferred_by"] if r else None) or \
            (r and r[0]["deferred_by"] > 0)
    fail = sorted(k for k, (_, s) in fires.items() if not s)
    succ = sorted(k for k, (_, s) in fires.items() if s)
    assert (len(fail), sum(not s for _, s, _ in eps.values())) == (7, 15)
    assert (len(succ), sum(s for _, s, _ in eps.values())) == (1, 35)
    assert {k: v[0] for k, v in fires.items()} == {"t1e2": 288, "t3e0": 229, "t3e1": 257, "t4e3": 243,
                                                   "t7e4": 223, "t8e0": 332, "t8e2": 360, "t9e3": 298}
    # C's listed steps: t7e4@215 and t8e2@353 - deferred +8/+7 by the causal rule (a width run was in progress)


@pytest.mark.skipif(not os.path.isdir(DEVBASE), reason="SCALE++ devbase traces not present")
def test_k1_seed7_counts_documented_deviation():
    files = []
    for pat in ("/shards/candidate-t{t}/candidate", "/seed-diversity/shards/scales1-t{t}/candidate",
                "/seed-diversity/shards/scales2-t{t}/candidate"):
        for t in range(10):
            d = DEVBASE + pat.format(t=t)
            if os.path.isfile(d + "/records.jsonl") and len(open(d + "/records.jsonl").read().splitlines()) == 10:
                files += sorted(glob.glob(d + "/vla_eval/episodes/LIBEROBenchmark_libero_10/task*_ep*.jsonl"))
    eps = _load(files)
    assert len(eps) == 300
    c_ff = sum(1 for S, s, n in eps.values() if not s and _c_script_first_fire(S, s, n) is not None)
    c_fs = sum(1 for S, s, n in eps.values() if s and _c_script_first_fire(S, s, n) is not None)
    assert (c_ff, c_fs) == (36, 4)                     # C's reported numbers reproduce with C's (non-causal) logic
    ff = sum(1 for S, s, n in eps.values() if not s and T.replay(S, 520))
    fs = sum(1 for S, s, n in eps.values() if s and T.replay(S, 520))
    assert (ff, fs) == (33, 4)                         # causal rule: 3 failure fires deferred past steps_left 150


# ---- gap rule ---------------------------------------------------------------------------------------------------
def rows(spec):
    S = []
    for n, z, w in spec:
        for _ in range(n):
            S.append([0, 0, z, 0, 0, 0, w / 2, -w / 2])
    return np.array(S, dtype=np.float64)


def test_carry_detection_and_end_kinds():
    S = rows([(5, 1.0, 0.08), (3, 0.95, 0.04), (5, 1.0, 0.04), (2, 1.0, 0.08)])
    (c,) = T.carries(S)
    assert c[2] == "release" and c[0] == 5 and c[1] == 13
    S = rows([(5, 1.0, 0.08), (3, 0.95, 0.04), (5, 1.0, 0.04), (2, 1.0, 0.0)])
    assert T.carries(S)[0][2] == "lost"


def test_gap_fires_at_en_plus_W_and_respects_steps_left_and_defers():
    S = rows([(5, 1.0, 0.08), (3, 0.95, 0.04), (5, 1.0, 0.04), (200, 1.0, 0.08)])
    f = T.replay(S, 520)
    assert f[0]["step"] == 13 + 140 and f[0]["end_kind"] == "release" and f[0]["deferred_by"] == 0
    assert T.replay(S, 13 + 140 + 149) == []          # steps_left 149 at the window end
    # a width run in progress at the window end defers the fire until it ends without a lift
    S2 = rows([(5, 1.0, 0.08), (3, 0.95, 0.04), (5, 1.0, 0.04), (130, 1.0, 0.08), (20, 1.0, 0.03), (60, 1.0, 0.08)])
    f2 = T.replay(S2, 520)
    assert f2[0]["step"] == 5 + 3 + 5 + 130 + 20 + 1 and f2[0]["deferred_by"] > 0
    # ... and cancels it when that run becomes a carry
    S3 = rows([(5, 1.0, 0.08), (3, 0.95, 0.04), (5, 1.0, 0.04), (130, 1.0, 0.08), (10, 0.95, 0.03), (10, 1.0, 0.03),
               (5, 1.0, 0.03)])
    assert T.replay(S3, 520) == []


def test_release_only_param_and_max_fires():
    S = rows([(5, 1.0, 0.08), (3, 0.95, 0.04), (5, 1.0, 0.04), (200, 1.0, 0.0)])
    assert T.replay(S, 520)                              # default: a loss counts (C's measured script)
    assert T.replay(S, 520, {"end_kinds": ["release"]}) == []
    with pytest.raises(ValueError):
        T.GapMonitor({"end_kinds": ["dropped"]})
    seg = [(3, 0.95, 0.04), (5, 1.0, 0.04), (150, 1.0, 0.08)]
    S = rows([(2, 1.0, 0.08)] + seg * 3)
    assert len(T.replay(S, 2000)) == 2


# ---- perception + motion ------------------------------------------------------------------------------------------
def bench_obs(p=(0.0, 0.0, 1.05)):
    b = RGBDTwoBlockBench()
    raw = b.reset({"name": "t"})
    raw["p"] = np.array(p)
    return b.make_obs(raw, {"name": "t"})


def test_hover_target_geometry():
    pk = P.Packet.capture(bench_obs(), 0)
    r, c = pixel_of([OBJS["B"][0], OBJS["B"][1], TOP])
    h = P.hover_target(pk, r, c, {})
    assert h["ok"] and abs(h["local_top"] - TOP) < 0.005 and abs(h["hover_xyz"][2] - (TOP + 0.10)) < 0.005
    assert abs(h["transit_z"] - (0.90 + 0.25)) < 0.006 and h["extent_xy"] is not None
    assert abs(h["height"] - 0.08) < 0.006
    # a bare-table point is NOT rejected as background (hover only): floor = table + 0.05
    r2, c2 = pixel_of([0.2, -0.2, 0.90])
    h2 = P.hover_target(pk, r2, c2, {})
    assert h2["ok"] and abs(h2["hover_xyz"][2] - (0.90 + 0.05 + 0.10)) < 0.006
    # outside workspace / image
    assert P.hover_target(P.Packet.capture(bench_obs((-0.3, 0.3, 1.05)), 0), *pixel_of([0.25, -0.3, 0.9]), {})[
        "reason"] == "outside_workspace"
    assert P.hover_target(pk, 999, 3, {})["reason"] == "pixel_outside_image"


def test_reseat_never_closes_and_respects_budget():
    state = {"p": np.array([-0.12, 0.13, 1.05])}
    sent = []

    async def step(a):
        sent.append(np.array(a))
        state["p"] = state["p"] + 0.0082 * np.asarray(a[:3], dtype=float)
        return 1

    m = M.Motion(step, lambda: np.array([*state["p"], 0, 0, 0, 0.04, -0.04]), lambda: False)
    res = asyncio.run(m.reseat([-0.06, 0.0, 1.08], 1.15, 45))
    assert res["reached"] and res["steps"] <= 45
    assert all(a[6] == -1.0 for a in sent) and all(np.all(a[3:6] == 0) for a in sent)
    sent.clear()
    res = asyncio.run(m.reseat([0.3, -0.3, 1.0], 1.15, 45))
    assert res["steps"] == 45 and res["budget_exhausted"] and all(a[6] == -1.0 for a in sent)
    sent.clear()
    state["p"] = np.array([-0.12, 0.13, 1.05])
    res = asyncio.run(m.reseat([0.3, -0.3, 1.0], 1.15, 90))
    assert res["steps"] <= 90 and all(a[6] == -1.0 for a in sent)


def test_reseat_budget_rule_three_regimes():
    assert M.reseat_budget(150) == 45 and M.reseat_budget(160) == 55 and M.reseat_budget(400) == 90
    assert M.reseat_budget(195) == 90 and M.reseat_budget(100) == 45


# ---- planner ----------------------------------------------------------------------------------------------------
def _strict(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert set(schema["required"]) == set(schema["properties"]) and schema["additionalProperties"] is False
        for v in schema.values():
            _strict(v)
    elif isinstance(schema, list):
        for v in schema:
            _strict(v)


def test_schemas_strict_and_validation():
    _strict(PL.RESEAT_SCHEMA)
    _strict(PL.VERDICT_SCHEMA)
    ok = {"remaining": [{"label": "moka pot", "kind": "object"}], "target": {"label": "moka pot", "row": 3, "col": 4},
          "action": "reseat"}
    assert PL.validate_reseat(ok, 512) is None
    assert "needs a target" in PL.validate_reseat({**ok, "target": None}, 512)
    assert "outside" in PL.validate_reseat({**ok, "target": {"label": "x", "row": 700, "col": 1}}, 512)
    assert PL.validate_reseat({"remaining": [], "target": None, "action": "none"}, 512) is None
    assert PL.target_kind(ok) == "object"
    assert PL.target_kind({**ok, "remaining": [{"label": "the moka pot", "kind": "articulated_part"}]}) == \
        "articulated_part"


def test_prompt_hygiene_and_size_cue():
    instr = "put both moka pots on the stove"
    texts = [PL.SYSTEM, PL.reseat_prompt(instr, 512), PL.witness_prompt(instr, "moka pot", False, {"height": 0.13})]
    for t in texts:
        rest = t.replace(instr, "").lower()
        for w in ("libero", "success", "succeed", "episode", "benchmark", "reward", "bddl", "task list"):
            assert w not in rest, (w, t[:60])
    assert "about 13.0 cm tall" in texts[2]
    assert "let go of an object" in texts[1] and "put an object down" not in texts[1]
    assert "articulated_part" in texts[1] and "image 1 is the fixed front scene camera at the start, before the robot moved" in texts[1].lower()


def test_sampler_is_plain_scale():
    from harnessstudio.core import registry
    assert C.scale_sampler() == registry.load("scale").slots["sampler"].to_json()


def test_evidence_dirs():
    d = C.evidence_dirs(["hs", "eval", "--out", "/tmp/x"], {})
    assert d["packets"].endswith("x/reseat_reentry_evidence/packets") and d["planner_source"] == "argv:--out"
    assert C.evidence_dirs(["pytest"], {})["packets"] is None
    assert C.evidence_dirs([], {"HS_RR_PACKET_DIR": "/tmp/y"})["packets_source"] == "override:HS_RR_PACKET_DIR"


# ---- cards, audit, closure -------------------------------------------------------------------------------------------
def test_cards_frozen_scalars_send_state_operator():
    import yaml
    for name in ("paper.yaml", "paper_respondoff.yaml"):
        card = yaml.safe_load(open(os.path.join(PKG, name)))
        p = card["params"]
        assert card["benchmark_params"] == {"send_state": True}
        assert p["triggers"]["W"] == 140 and p["triggers"]["min_steps_left"] == 150 and p["triggers"]["max_fires"] == 2
        pc = p["perception"]
        assert (pc["transit_above_table"], pc["hover_above_top"], pc["top_disk"], pc["top_floor_above_table"]) == \
            (0.25, 0.10, 0.04, 0.05)
        assert p["motion"]["tol"] == 0.02 and (p["motion"]["min_steps"], p["motion"]["max_steps"],
                                                 p["motion"]["reserve_policy_steps"]) == (45, 90, 105)
        assert p["planner"]["max_calls_episode"] == 6 == card["budgets"]["llm_calls"]
        assert p["planner"]["timeout_s"] == 60 and p["planner"]["effort"] == "low"
        assert card["method_local_operators"][0]["model"] == "gpt-6.1-sol"
        assert set(p["motion"]) == set(M.DEFAULTS) and set(p["perception"]) == set(P.DEFAULTS)
        assert set(p["triggers"]) == set(T.DEFAULTS)
    assert yaml.safe_load(open(os.path.join(PKG, "paper_respondoff.yaml")))["params"]["mode"] == "respondoff"


def _sources():
    out = {}
    for root, dirs, files in os.walk(os.path.join(PKG, "..")):
        dirs[:] = [d for d in dirs if d not in ("__pycache__",) and not d.endswith(".egg-info")]
        for f in files:
            if f.endswith((".py", ".yaml", ".yml", ".json", ".toml")):
                out[os.path.join(root, f)] = open(os.path.join(root, f), errors="replace").read()
    return out


def test_static_audit_no_privileged_inputs_or_task_tables():
    for path, text in _sources().items():
        if "/tests/" in path or "/vendor/" in path:
            continue
        for pat in (r"\bbddl\b", r"sqlite", r"recording", r"get_result", r"\breward\b", r"ctx\._", r"\.sim\b",
                    r"object_pos", r"alphabet", r"tomato", r"ketchup", r"cream cheese", r"libero_10/"):
            assert not re.search(pat, text, re.I), (path, pat)


def test_freeze_closure_tokens():
    names = ["hs_" + "witnessed_rescue", "hs_" + "harness_vla", "hs_" + "gpp", "hs_" + "vlm_rehome"]
    text = "\n".join(_sources().values())
    for n in names:
        assert not re.search(rf"\b{n}\b", text), n
