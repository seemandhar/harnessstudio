"""gpp primitives forced through the real handler seam: in-process vla-eval (GripperBench with public `states`) ->
HarnessServer -> GripperPolicy (spy). Every arm, every primitive outcome; greedy-escape reuse; request contract.
(Lesson 20261102T184420Z: tier 1's fake stack never activates a gated branch.)"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpp_fake_bench import GripperPolicy  # noqa: E402

from harnessstudio.contracts import load_contract  # noqa: E402
from harnessstudio.core import registry  # noqa: E402
from harnessstudio.core.arms import Arm  # noqa: E402
from harnessstudio.core.card import config_from_card, load_card  # noqa: E402
from harnessstudio.core.method import ModuleRef  # noqa: E402
from harnessstudio.core.noise import crn_seed  # noqa: E402
from harnessstudio.profiles import load_profile  # noqa: E402
from harnessstudio.server.harness_server import HarnessServer  # noqa: E402
from harnessstudio.verify.stack import serve  # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "hs_gpp"


def contract(horizon):
    c = load_contract("fake")
    return dataclasses.replace(c, public_keys=tuple(c.public_keys) + ("states",), state_key="states",
                               default_horizon=horizon)


def run_arm(config, arm: Arm, horizon=80, profile="fake-t1", policy_kwargs=None, **bench):
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
                   "benchmarks": [{"benchmark": "gpp_fake_bench:GripperBench", "mode": "sync", "episodes_per_task": 1,
                                   "params": {"horizon": horizon, **bench}}]}
            with tempfile.TemporaryDirectory(prefix="gpp-stack-") as tmp, contextlib.redirect_stdout(io.StringIO()):
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
    return config_from_card(card, controller=ModuleRef("hs_gpp.controller:GppController", params))


def events(rec, phase=None):
    return [e for e in rec["events"] if e["t"] == "event" and e.get("name") == "gpp"
            and (phase is None or e.get("phase") == phase)]


def gpp(rec):
    return rec["accounting"]["extra"]["gpp"]


# ---- candidate: H2 then H1 in one episode -------------------------------------------------------------------------
def test_h2_then_h1_candidate_through_the_seam():
    rec, log = run_arm(registry.load("gpp-core"), Arm.CANDIDATE)
    s = gpp(rec)
    f2, f1 = s["fires"][0], s["fires"][1]
    assert f2["module"] == "h2" and f2["variant"] == "candidate" and f2["trigger_at"] == 7 and f2["start"] == 8
    assert f2["outcome"] == "distance" and f2["primitive_steps"] == 11 and f2["result"]["travel"] <= -0.03
    assert f2["next_close"]["held"] is True                      # the policy's next close (lower) is held
    assert f1["module"] == "h1" and f1["variant"] == "candidate" and f1["outcome"] == "lift_cap"
    assert f1["trigger"]["veto"] == f1["start"] and f1["primitive_steps"] == 12
    assert 0.04 < f1["result"]["rise"] < 0.06 and f1["result"]["w"] > 0.01
    assert f1["reopen_after"] == 1                               # the fake policy releases right after hand-back
    assert f1["frames"] == {"fire": f1["trigger_at"], "start": f1["start"], "end": f1["end"],
                            "plus10": f1["end"] + 10, "plus30": f1["end"] + 30}
    # scripted steps logged one by one and make no policy call (D1)
    scripted = events(rec, "scripted")
    assert len(scripted) == s["scripted_steps"] == 23
    acc = rec["accounting"]
    # policy samples = executed policy actions + the vetoed H1 open (sampled, not executed)
    assert acc["method_policy_samples"] == s["policy_decisions"] == acc["steps"] - 23 + s["h1_fires"]
    assert len(log) == s["policy_decisions"]
    # the vetoed open was requested (one sample) but not executed: the H1 decision is a policy decision
    assert all(e["sampler"] is None for e in log)
    # seeds: one per policy decision, consecutive CRN decision indices (scripted steps take none)
    assert [e["seeds"] for e in log] == [[crn_seed("hs", rec["benchmark"], rec["task"], 0, d, 0)] for d in range(len(log))]


def test_h2_contact_stop_and_h1_blocked_abort():
    rec, _ = run_arm(registry.load("gpp-h2"), Arm.CANDIDATE, floor=1.015)
    f = gpp(rec)["fires"][0]
    assert f["module"] == "h2" and f["outcome"] == "contact_stop" and f["result"]["travel"] > -0.03
    rec, _ = run_arm(registry.load("gpp-core"), Arm.CANDIDATE, obj_top=1.04, stuck=True)
    f = gpp(rec)["fires"][0]
    assert f["module"] == "h1" and f["outcome"] == "abort" and f["result"]["abort_reason"] == "blocked"
    assert f["primitive_steps"] == 5                             # 4 lift steps + the vetoed open
    ends = events(rec, "end")
    assert ends[0]["outcome"] == "abort"


# ---- shams ------------------------------------------------------------------------------------------------------------
def test_shams_same_triggers_opposite_or_no_direction():
    cand, _ = run_arm(registry.load("gpp-core"), Arm.CANDIDATE, obj_top=1.04)
    sham, _ = run_arm(registry.load("gpp-core-sham"), Arm.CANDIDATE, obj_top=1.04)
    fc, fs = gpp(cand)["fires"][0], gpp(sham)["fires"][0]
    assert fc["module"] == fs["module"] == "h1" and fc["trigger_at"] == fs["trigger_at"]   # pre-fire bitwise
    assert cand["action_sha256"][:fc["start"]] == sham["action_sha256"][:fs["start"]]
    assert fs["variant"] == "sham_veto_only" and fs["outcome"] == "veto_done" and fs["primitive_steps"] == 12
    assert abs(fs["result"]["rise"]) < 1e-9
    rec, _ = run_arm(registry.load("gpp-h2-sham"), Arm.CANDIDATE)
    f = gpp(rec)["fires"][0]
    assert f["variant"] == "sham_ascend" and f["trigger_at"] == 7 and f["result"]["travel"] >= 0.03
    assert gpp(rec)["h2_fires"] == 2                             # rose >= 2 cm, so the 2nd fire is allowed


# ---- null -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("entry,bench", [("gpp-core", {}), ("gpp-core", {"obj_top": 1.04, "stuck": True})])
def test_shadow_null_is_baseline_and_logs_primitives(entry, bench):
    base, blog = run_arm(registry.load(entry), Arm.BASELINE, **bench)
    null, nlog = run_arm(registry.load(entry), Arm.NULL, **bench)
    assert null["action_sha256"] == base["action_sha256"]
    s = gpp(null)
    cand, _ = run_arm(registry.load(entry), Arm.CANDIDATE, **bench)
    first = gpp(cand)["fires"][0]
    # the null's first fire is the candidate's (pre-fire trajectories are identical), then native actions execute
    assert s["fires"] and (s["fires"][0]["module"], s["fires"][0]["trigger_at"]) == (first["module"], first["trigger_at"])
    acc = null["accounting"]
    assert acc["native_policy_samples"] == acc["steps"]          # every executed action is native


# ---- off switches / greedy-escape module ------------------------------------------------------------------------------
def test_all_modules_off_is_bitwise_baseline_at_request_level():
    cfg = variant("gpp-core", h2="off", h1="off")
    base, blog = run_arm(cfg, Arm.BASELINE)
    cand, clog = run_arm(cfg, Arm.CANDIDATE)
    assert cand["action_sha256"] == base["action_sha256"]
    strip = lambda log: [(e["seeds"], e["sampler"], e["z"], e["w"]) for e in log]  # noqa: E731
    assert strip(clog) == strip(blog)
    assert gpp(cand)["fires"] == [] and gpp(cand)["scripted_steps"] == 0


def test_h3_off_by_default_and_on_switch_fires_once():
    assert registry.load("gpp-core").controller.params["modules"]["h3"] == "off"
    core, _ = run_arm(registry.load("gpp-core"), Arm.CANDIDATE, policy_kwargs={"frozen": True})
    assert gpp(core)["fires"] == []                               # H3 off: a frozen start is left alone
    for mode, sign in (("on", -1), ("sham", 1)):
        rec, _ = run_arm(variant("gpp-core", h2="off", h1="off", h3=mode), Arm.CANDIDATE, policy_kwargs={"frozen": True})
        s = gpp(rec)
        assert s["h3_fires"] == 1
        f = s["fires"][0]
        assert f["trigger_at"] == 25 and f["start"] == 26 and f["primitive_steps"] == 8 and f["outcome"] == "max_steps"
        assert sign * f["result"]["travel"] > 0.03


def test_ge_only_reproduces_greedy_escape_requests_and_actions():
    """H1/H2 off, GE on: the gpp controller sends exactly greedy-escape's requests (seed, sampler) and actions."""
    cfg = variant("ge-gpp-core", h2="off", h1="off")
    ge_rec, ge_log = run_arm(registry.load("greedy-escape"), Arm.CANDIDATE, profile="fake-t2",
                             policy_kwargs={"frozen": True})
    rec, log = run_arm(cfg, Arm.CANDIDATE, profile="fake-t2", policy_kwargs={"frozen": True})
    assert [(e["seeds"], e["sampler"]) for e in log] == [(e["seeds"], e["sampler"]) for e in ge_log]
    assert rec["action_sha256"] == ge_rec["action_sha256"]
    starts = gpp(rec)["ge"]["window_starts"]
    assert starts == ge_rec["accounting"]["extra"]["greedy_escape"]["window_starts"] and starts[0] == 15
    escape = registry.load("greedy-escape").controller.params["escape"]
    assert [d for d, e in enumerate(log) if e["sampler"] == escape][:2] == [15, 16]


def test_composition_precedence_h2_before_ge():
    """ge-gpp-core on the gripper scene: H2 fires (step 7) long before any GE trigger; no window in the first fire."""
    rec, log = run_arm(registry.load("ge-gpp-core"), Arm.CANDIDATE, profile="fake-t2")
    s = gpp(rec)
    assert s["fires"][0]["module"] == "h2" and s["fires"][0]["trigger_at"] == 7
    assert all(e["sampler"] is None for e in log[:20])


def test_ge_module_off_in_core_and_frozen_in_composition():
    core = registry.load("gpp-core")
    assert core.controller.params["modules"]["ge"] is False and "ge" not in core.controller.params
    comp = registry.load("ge-gpp-core")
    ge = registry.load("greedy-escape").controller.params
    for k in ("fixed_point_steps", "fixed_point_eps", "empty_close_steps", "empty_width_m", "empty_closes",
              "window_decisions", "max_windows", "escape"):
        assert comp.controller.params["ge"][k] == ge[k]
    bad = dict(comp.controller.params)
    bad["ge"] = {**bad["ge"], "window_decisions": 41}
    with pytest.raises(ValueError):
        ModuleRef("hs_gpp.controller:GppController", bad).build()


# ---- v1.1 contact-stop revision (G1): OSC ramp-up from rest ---------------------------------------------------------
@pytest.mark.parametrize("ramp", [0.3, 0.1])
def test_v11_free_descent_with_slow_first_step_reaches_distance(ramp):
    rec, _ = run_arm(registry.load("gpp-h2"), Arm.CANDIDATE, gain=0.013, ramp=ramp, obj_top=0.99)
    f = gpp(rec)["fires"][0]
    assert f["module"] == "h2" and f["outcome"] == "distance" and f["result"]["travel"] <= -0.03
    dz = f["result"]["dz_mm"]
    assert dz[0] > -0.3 * 6.5 - 1e-6 and all(d < -6.0 for d in dz[1:])           # slow first step, then full gain
    if ramp == 0.1:                                                               # the v1.0 rule would have stopped here
        assert dz[0] > -1.0
        old = variant("gpp-h2", h2="on")
        old.controller.params["h2"]["stop_from_move"] = 1
        rec1, _ = run_arm(old, Arm.CANDIDATE, gain=0.013, ramp=ramp, obj_top=0.99)
        f1 = gpp(rec1)["fires"][0]
        assert f1["outcome"] == "contact_stop" and f1["result"]["moved"] == 1


def test_v11_floor_at_start_position_stops_at_move_step_2():
    rec, _ = run_arm(registry.load("gpp-h2"), Arm.CANDIDATE, floor=1.02949, ramp=0.3)
    f = gpp(rec)["fires"][0]
    assert f["outcome"] == "contact_stop" and f["result"]["moved"] == 2 and f["result"]["travel"] > -0.003
    assert len(f["result"]["dz_mm"]) == 2


def test_v11_ascend_sham_with_slow_first_step_reaches_distance():
    rec, _ = run_arm(registry.load("gpp-h2-sham"), Arm.CANDIDATE, gain=0.013, ramp=0.1, obj_top=0.99)
    f = gpp(rec)["fires"][0]
    assert f["variant"] == "sham_ascend" and f["outcome"] == "distance" and f["result"]["travel"] >= 0.03
    assert f["result"]["dz_mm"][0] < 1.0                                          # slow first step not a stop


def test_v11_cards_carry_stop_from_move_2():
    for entry in ("gpp-core", "gpp-core-sham", "gpp-h2", "gpp-h2-sham", "ge-gpp-core"):
        p = registry.load(entry).controller.params
        assert p["h2"]["stop_from_move"] == 2 and p["h3"]["stop_from_move"] == 2
        assert registry.load(entry).version == "1.1"


# ---- G3: greedy-escape pinned by identity ---------------------------------------------------------------------------
def test_g3_ge_module_refuses_a_modified_greedy_escape(tmp_path):
    import shutil
    import subprocess
    import hs_greedy_escape
    src = Path(hs_greedy_escape.__file__).resolve().parent
    dst = tmp_path / "hs_greedy_escape"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    code = ("import sys; sys.path.insert(0, %r)\n"
            "import hs_greedy_escape; assert hs_greedy_escape.__file__.startswith(%r), hs_greedy_escape.__file__\n"
            "from harnessstudio.core import registry\n"
            "c = registry.load('ge-gpp-core')\n"
            "try:\n    c.controller.build()\nexcept ValueError as e:\n    print('REFUSED', e)\nelse:\n    print('BUILT')\n")
    run = lambda: subprocess.run([sys.executable, "-c", code % (str(tmp_path), str(tmp_path))],  # noqa: E731
                                 capture_output=True, text=True, cwd=str(tmp_path))
    out = run()
    assert "BUILT" in out.stdout, out.stdout + out.stderr                       # an identical copy is accepted
    (dst / "monitor.py").write_text((dst / "monitor.py").read_text() + "\n# modified\n")
    out = run()
    assert "REFUSED" in out.stdout and "monitor.py" in out.stdout, out.stdout + out.stderr
