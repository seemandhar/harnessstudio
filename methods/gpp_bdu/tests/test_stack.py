"""gpp-bdu through the real handler seam (in-process vla-eval -> HarnessServer -> spy policy, public `states`):
bdu off reproduces the parent entries' requests and actions; forced blocked descents fire bdu (candidate, sham, close
passthrough, lateral end), the shadow null equals the baseline, pre-fire actions equal the parent's."""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdu_fake_bench import GripperPolicy  # noqa: E402

from harnessstudio.contracts import load_contract  # noqa: E402
from harnessstudio.core import registry  # noqa: E402
from harnessstudio.core.arms import Arm  # noqa: E402
from harnessstudio.core.card import config_from_card, load_card  # noqa: E402
from harnessstudio.core.method import ModuleRef  # noqa: E402
from harnessstudio.profiles import load_profile  # noqa: E402
from harnessstudio.server.harness_server import HarnessServer  # noqa: E402
from harnessstudio.verify.stack import serve  # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "hs_gpp_bdu"


def contract(horizon):
    c = load_contract("fake")
    return dataclasses.replace(c, public_keys=tuple(c.public_keys) + ("states",), state_key="states",
                               default_horizon=horizon)


def run_arm(config, arm: Arm, horizon=80, profile="fake-t2", policy_kwargs=None, **bench):
    import contextlib
    import io
    import tempfile

    from vla_eval.api import run
    policy = GripperPolicy(**(policy_kwargs or {}))
    with serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract(horizon), arm=arm,
                                profile=load_profile(profile), scripted_llm=True)
        with serve(harness) as hh:
            cfg = {"server": {"url": hh.url, "timeout": 60.0},
                   "benchmarks": [{"benchmark": "bdu_fake_bench:GripperBench", "mode": "sync", "episodes_per_task": 1,
                                   "params": {"horizon": horizon, **bench}}]}
            with tempfile.TemporaryDirectory(prefix="bdu-stack-") as tmp, contextlib.redirect_stdout(io.StringIO()):
                cfg["output_dir"] = tmp
                run(cfg, docker=False, no_save=False)
            harness.wait_idle()
    (rec,) = harness.records
    assert rec["valid"], rec["invalid_reason"]
    return rec, policy.log


def variant(entry, **modules):
    card = PKG / f"paper_{entry.replace('-', '_')}.yaml"
    params = dict(load_card(card)["params"])
    params["modules"] = {**params["modules"], **modules}
    return config_from_card(card, controller=ModuleRef("hs_gpp_bdu.controller:GppBduController", params))


def gpp(rec):
    return rec["accounting"]["extra"]["gpp"]


def strip(log):
    return [(e["seeds"], e["sampler"], e["z"], e["w"]) for e in log]


PRESS = {"policy_kwargs": {"press": 0.1}, "floor": 1.03}


# ---- bdu off == parent (requests and actions) -------------------------------------------------------------------------
@pytest.mark.parametrize("entry,parent,scene", [
    ("ge-gpp-bdu", "ge-gpp-core", {}),                                   # H2 then H1 scene
    ("ge-gpp-bdu", "ge-gpp-core", {"policy_kwargs": {"frozen": True}}),  # GE window scene
    ("ge-gpp-bdu", "ge-gpp-core", PRESS),                                # blocked descent scene, bdu off
    ("gpp-bdu", "gpp-core", {}),
    ("gpp-bdu", "gpp-core", PRESS),
    ("ge-bdu", "ge-gpp-core", {"policy_kwargs": {"frozen": True}}),
])
def test_bdu_off_reproduces_parent(entry, parent, scene):
    pcfg = registry.load(parent)
    if entry == "ge-bdu":                                                # ge-bdu's parent path = ge-gpp-core with h1/h2 off
        params = dict(pcfg.controller.params)
        params["modules"] = {**params["modules"], "h1": "off", "h2": "off"}
        pcfg = config_from_card(Path(__import__("hs_gpp").__file__).with_name("paper_ge_gpp_core.yaml"),
                                controller=ModuleRef(pcfg.controller.factory, params))
    prof = "fake-t1" if parent == "gpp-core" else "fake-t2"
    a, alog = run_arm(pcfg, Arm.CANDIDATE, profile=prof, **scene)
    b, blog = run_arm(variant(entry, bdu="off"), Arm.CANDIDATE, profile=prof, **scene)
    assert b["action_sha256"] == a["action_sha256"]
    assert strip(blog) == strip(alog)
    assert gpp(b)["bdu_fires"] == 0


# ---- forced bdu fires -------------------------------------------------------------------------------------------------
def test_bdu_fires_on_blocked_descent_and_pre_fire_equals_parent():
    rec, log = run_arm(registry.load("ge-gpp-bdu"), Arm.CANDIDATE, **PRESS)
    par, plog = run_arm(registry.load("ge-gpp-core"), Arm.CANDIDATE, **PRESS)
    s = gpp(rec)
    f = s["fires"][0]
    assert f["module"] == "bdu" and f["variant"] == "candidate"
    # approach 1.05 -> floor 1.03: steps 0-4 move (step 4 clamps), blocked from step 5; 20 blocked -> fire after 24
    assert f["trigger_at"] == 24 and f["start"] == 25
    assert rec["action_sha256"][:25] == par["action_sha256"][:25] and rec["action_sha256"][25] != par["action_sha256"][25]
    assert f["outcome"] == "max_steps" and f["primitive_steps"] == 10 and f["result"]["edited_steps"] == 10
    assert f["result"]["realized_lateral_m"] == pytest.approx(10 * 0.0082 * 0.1, rel=1e-3)
    assert f["result"]["cos_realized_vs_commanded"] == pytest.approx(1.0)
    assert f["frames"]["end"] == 34 and f["frames"]["plus30"] == 64
    assert s["bdu_edited_steps"] == 10 * s["bdu_fires"] and s["bdu_fires"] >= 2   # re-fires after 20 more blocked steps
    # one policy request every step (c = 1), greedy inside the bdu window
    assert len(log) == rec["accounting"]["steps"] and all(e["sampler"] is None for e in log)
    edits = [e for e in rec["events"] if e["t"] == "event" and e.get("name") == "bdu" and e.get("phase") == "edit"]
    assert len(edits) == s["bdu_edited_steps"] and all(e["a_z"] == pytest.approx(0.15) for e in edits)


def test_sham_is_lift_only_and_pairs_before_the_fire():
    cand, _ = run_arm(registry.load("ge-gpp-bdu"), Arm.CANDIDATE, **PRESS)
    sham, _ = run_arm(registry.load("ge-gpp-bdusham"), Arm.CANDIDATE, **PRESS)
    fc, fs = gpp(cand)["fires"][0], gpp(sham)["fires"][0]
    assert fs["variant"] == "sham_lift_only" and fs["trigger_at"] == fc["trigger_at"]
    assert sham["action_sha256"][:fs["start"]] == cand["action_sha256"][:fc["start"]]
    assert fs["result"]["realized_lateral_m"] == pytest.approx(0.0, abs=1e-9)
    assert fs["result"]["mean_commanded_xy"][0] == pytest.approx(0.1)        # commanded lateral recorded, not executed


def test_close_passthrough_and_lateral_end():
    rec, log = run_arm(registry.load("gpp-bdu"), Arm.CANDIDATE, profile="fake-t1",
                       policy_kwargs={"press": 0.1, "close_at": 28}, floor=1.03)
    f = gpp(rec)["fires"][0]
    assert f["outcome"] == "close_passthrough" and f["primitive_steps"] == 4 and f["result"]["edited_steps"] == 3
    rec, _ = run_arm(registry.load("gpp-bdu"), Arm.CANDIDATE, profile="fake-t1", policy_kwargs={"press": 0.5},
                     floor=1.03)
    f = gpp(rec)["fires"][0]
    assert f["outcome"] == "lateral_done" and f["primitive_steps"] == 7 and f["result"]["realized_lateral_m"] >= 0.025


def test_shadow_null_is_baseline_and_logs_the_bdu_fire():
    base, _ = run_arm(registry.load("ge-gpp-bdu"), Arm.BASELINE, **PRESS)
    null, _ = run_arm(registry.load("ge-gpp-bdu"), Arm.NULL, **PRESS)
    assert null["action_sha256"] == base["action_sha256"]
    assert gpp(null)["fires"][0]["module"] == "bdu" and gpp(null)["fires"][0]["trigger_at"] == 24
    acc = null["accounting"]
    assert acc["native_policy_samples"] == acc["steps"]


@pytest.mark.parametrize("entry,parent,scene,prof", [
    ("ge-gpp-bdu", "ge-gpp-core", {}, "fake-t2"),
    ("ge-gpp-bdu", "ge-gpp-core", {"policy_kwargs": {"frozen": True}}, "fake-t2"),
    ("ge-gpp-bdusham", "ge-gpp-core", {}, "fake-t2"),
    ("gpp-bdu", "gpp-core", {}, "fake-t1"),
    ("gpp-bdu", "gpp-core", {"obj_top": 1.04, "stuck": True}, "fake-t1"),
])
def test_bdu_on_without_a_bdu_fire_equals_parent(entry, parent, scene, prof):
    """The audit rule: a run with no bdu fire is bitwise the component path (requests and actions)."""
    a, alog = run_arm(registry.load(parent), Arm.CANDIDATE, profile=prof, **scene)
    b, blog = run_arm(registry.load(entry), Arm.CANDIDATE, profile=prof, **scene)
    assert gpp(b)["bdu_fires"] == 0
    assert b["action_sha256"] == a["action_sha256"] and strip(blog) == strip(alog)
