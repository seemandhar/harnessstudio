"""witnessed-rescue forced through the real handler seam: in-process vla-eval (RGBDGripperBench with public
states + RGB-D + calibration) -> HarnessServer -> SpyPolicy, every arm, scripted planner (no network).
Covers H2 grasp+custody+place, H1 lift+custody+place, abstention/transport hand-backs, respondoff bitwise == plain
`scale`, and the framework shadow null (native actions, paid SCALE samples on the shadow session, paid planner calls).
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wr_fake_bench import OBJ, TRAY, SpyPolicy, pixel_of  # noqa: E402

from harnessstudio.contracts import load_contract  # noqa: E402
from harnessstudio.core import registry  # noqa: E402
from harnessstudio.core.arms import Arm  # noqa: E402
from harnessstudio.core.card import config_from_card, load_card  # noqa: E402
from harnessstudio.core.method import ModuleRef  # noqa: E402
from harnessstudio.profiles import load_profile  # noqa: E402
from harnessstudio.server.harness_server import HarnessServer  # noqa: E402
from harnessstudio.verify.stack import serve  # noqa: E402

import hs_witnessed_rescue as W  # noqa: E402
from hs_witnessed_rescue.controller import scale_sampler  # noqa: E402

HORIZON = 300


def contract():
    c = load_contract("fake")
    return dataclasses.replace(c, public_keys=("images", "task_description", "states", "depth", "camera"),
                               state_key="states", default_horizon=HORIZON,
                               eef_position={"path": ["states"], "slice": [0, 3]})


def run_arm(config, arm: Arm, scenario="h2", dtype=np.float32, **bench):
    from vla_eval.api import run
    policy = SpyPolicy(scenario, dtype)
    with serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract(), arm=arm,
                                profile=load_profile("fake-t2"), scripted_llm=True)
        with serve(harness) as hh:
            cfg = {"server": {"url": hh.url, "timeout": 60.0},
                   "benchmarks": [{"benchmark": "wr_fake_bench:RGBDGripperBench", "mode": "sync",
                                   "episodes_per_task": 1, "params": {"horizon": HORIZON, **bench}}]}
            with tempfile.TemporaryDirectory(prefix="wr-stack-") as tmp, contextlib.redirect_stdout(io.StringIO()):
                cfg["output_dir"] = tmp
                run(cfg, docker=False, no_save=False)
            harness.wait_idle()
    (rec,) = harness.records
    assert rec["valid"], rec["invalid_reason"]
    return rec, policy.log


def variant(card="paper.yaml", replies=(), **over):
    path = os.path.join(os.path.dirname(W.__file__), card)
    params = dict(load_card(path)["params"])
    params["planner"] = {**params["planner"], "backend": "scripted", "scripted_responses": list(replies)}
    for k, v in over.items():
        if isinstance(v, dict):
            params[k] = {**params[k], **v}
        else:
            params[k] = v
    return config_from_card(path, controller=ModuleRef(W.FACTORY, params))


def plan(action, target=True, dest=True, clamped=None, at_dest=None, open_top=True):
    tr, tc = pixel_of([OBJ["center"][0], OBJ["center"][1], OBJ["top"]])
    dr, dc = pixel_of([TRAY["center"][0], TRAY["center"][1], 0.90])
    return {"assessment": "scripted", "clamped_task_object": clamped, "clamped_at_destination": at_dest,
            "target": {"label": "red block", "row": tr, "col": tc} if target else None,
            "destination": {"label": "blue tray", "row": dr, "col": dc, "open_top": open_top} if dest else None,
            "grasp_style": "center", "action": action}


CONFIRM = {"verdict": "confirm", "seen": "red block", "reason": "scripted"}
HELD = {"verdict": "confirm", "reason": "scripted"}


def wr(rec):
    return rec["accounting"]["extra"]["witnessed_rescue"]


# ---- candidate ------------------------------------------------------------------------------------------------
def test_h2_grasp_custody_place_through_the_seam():
    rec, log = run_arm(variant(replies=[plan("grasp_and_place"), CONFIRM, HELD]), Arm.CANDIDATE, "h2")
    s = wr(rec)
    (esc,) = s["escalations"]
    assert esc["kind"] == "H2_empty_close" and s["fires"][0]["eligible"]
    assert [c["role"] for c in esc["planner_calls"]] == ["plan", "witness", "custody"]
    assert [c["outcome"] for c in esc["planner_calls"]] == ["accept", "accept", "accept"]
    assert esc["custody"] is True and esc["grasps"][0]["custody"] is True
    assert esc["grasps"][0]["rise"] >= 0.05
    assert esc["place"] is not None and esc["handback"] == "after_place"
    assert rec["success"] is True             # the fake bench reports the block on the tray
    assert s["primitive_steps"] > 0 and 0 < s["analytic_share"] < 1
    assert s["planner"]["calls"] == 3 and s["planner"]["by_status"] == {"ok": 3}
    # every candidate request carried the plain-scale sampler
    assert all(e["sampler"] == scale_sampler() for e in log)
    assert any(e["t"] == "event" and e["name"] == "wr_escalation" for e in rec["events"])


def test_h1_lift_custody_place_through_the_seam():
    rec, _ = run_arm(variant(replies=[plan("lift_and_place", clamped=True, at_dest=False), CONFIRM, HELD]),
                     Arm.CANDIDATE, "h1")
    s = wr(rec)
    (esc,) = s["escalations"]
    assert esc["kind"] == "H1_settled_clamp"
    assert esc["custody"] is True and esc["handback"] == "after_place"
    assert rec["success"] is True


def test_h1_at_destination_hands_back_without_motion():
    rec, log = run_arm(variant(replies=[plan("lift_and_place", clamped=True, at_dest=True)]), Arm.CANDIDATE, "h1")
    (esc,) = wr(rec)["escalations"]
    assert esc["handback"] == "plan_not_accepted" and esc["planner_calls"][0]["outcome"] == "abstain"
    assert wr(rec)["primitive_steps"] == 0


@pytest.mark.parametrize("reply,status", [({"__transport__": "timeout"}, "timeout"),
                                          ({"__transport__": "403"}, "transport_error")])
def test_transport_failure_is_a_handback_not_a_semantic_abstention(reply, status):
    rec, _ = run_arm(variant(replies=[reply]), Arm.CANDIDATE, "h2")
    (esc,) = wr(rec)["escalations"]
    assert esc["planner_calls"][0]["status"] == status and esc["planner_calls"][0]["outcome"] is None
    assert esc["handback"] == "plan_not_accepted" and wr(rec)["primitive_steps"] == 0


def test_wrong_object_witness_hands_back():
    rec, _ = run_arm(variant(replies=[plan("grasp_and_place"), {"verdict": "wrong_object", "seen": "table",
                                                                 "reason": "x"}]), Arm.CANDIDATE, "h2")
    (esc,) = wr(rec)["escalations"]
    assert esc["handback"] == "witness_not_confirmed" and esc["planner_calls"][1]["outcome"] == "wrong_object"
    assert wr(rec)["primitive_steps"] == 0


def test_background_point_is_rejected_by_geometry():
    bad = plan("grasp_and_place")
    bad["target"]["row"], bad["target"]["col"] = pixel_of([0.25, -0.2, 0.90])     # empty table
    rec, _ = run_arm(variant(replies=[bad]), Arm.CANDIDATE, "h2")
    (esc,) = wr(rec)["escalations"]
    assert esc["handback"].startswith("witness_geometry") and "background" in esc["handback"]
    assert len(esc["planner_calls"]) == 1


def test_noplace_hands_back_after_custody():
    rec, _ = run_arm(variant("paper_noplace.yaml", replies=[plan("grasp_and_place"), CONFIRM]), Arm.CANDIDATE, "h2")
    (esc,) = wr(rec)["escalations"]
    assert esc["custody"] is True and esc["handback"] == "place_disabled" and esc["place"] is None


def test_episode_planner_budget_is_enforced():
    rec, _ = run_arm(variant(replies=[{"__transport__": "timeout"}] * 20,
                             planner={"max_calls_episode": 1}), Arm.CANDIDATE, "h2")
    s = wr(rec)
    assert s["planner"]["calls"] <= 1


# ---- SCALE-shadow null (respondoff) == plain scale, bitwise -------------------------------------------------------
@pytest.mark.parametrize("scenario", ["h2", "h1", "idle"])
def test_respondoff_is_bitwise_plain_scale(scenario):
    replies = [plan("grasp_and_place"), CONFIRM] if scenario == "h2" else \
        [plan("lift_and_place", clamped=True, at_dest=False), CONFIRM]
    ref, ref_log = run_arm(registry.load("scale"), Arm.CANDIDATE, scenario)
    rec, log = run_arm(variant("paper_respondoff.yaml", replies=replies), Arm.CANDIDATE, scenario)
    assert rec["action_sha256"] == ref["action_sha256"] and rec["obs_sha256"] == ref["obs_sha256"]
    assert [(e["seeds"], e["sampler"]) for e in log] == [(e["seeds"], e["sampler"]) for e in ref_log]
    if scenario != "idle":
        (esc,) = wr(rec)["escalations"]
        assert esc["handback"] == "respondoff" and esc["calls"] == 2 and wr(rec)["primitive_steps"] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_candidate_unfired_episode_is_bitwise_plain_scale(dtype):
    ref, _ = run_arm(registry.load("scale"), Arm.CANDIDATE, "idle", dtype)
    rec, _ = run_arm(variant(replies=[]), Arm.CANDIDATE, "idle", dtype)
    assert not wr(rec)["fires"]
    assert rec["action_sha256"] == ref["action_sha256"] and rec["obs_sha256"] == ref["obs_sha256"]


# ---- Campaign shadow null ------------------------------------------------------------------------------------------
def test_shadow_null_executes_native_actions_and_pays_its_calls():
    base, _ = run_arm(variant(replies=[]), Arm.BASELINE, "h2")
    rec, log = run_arm(variant(replies=[plan("grasp_and_place"), CONFIRM, HELD] * 3), Arm.NULL, "h2")
    assert rec["action_sha256"] == base["action_sha256"]
    s = wr(rec)
    assert s["fires"] and s["planner"]["calls"] >= 1          # the null paid planner calls on its own trajectory
    sessions = {e["session"] for e in log}
    assert len(sessions) == 2                                  # native session + shadow session
    shadow = [e for e in log if e["sampler"] is not None]
    native = [e for e in log if e["sampler"] is None]
    assert shadow and native and {e["session"] for e in shadow}.isdisjoint({e["session"] for e in native})
    assert all(e["sampler"] == scale_sampler() for e in shadow)


def test_drop_branch_when_enabled():
    replies = [plan("grasp_and_place"), CONFIRM, HELD]
    off, _ = run_arm(variant(replies=list(replies)), Arm.CANDIDATE, "drop", slip_lift=0.035)
    assert not any(f["kind"] == "drop" for f in wr(off)["fires"])
    rec, _ = run_arm(variant(replies=list(replies), triggers={"drop": True}), Arm.CANDIDATE, "drop", slip_lift=0.035)
    s = wr(rec)
    assert s["fires"][0]["kind"] == "drop" and s["escalations"][0]["kind"] == "drop"
    assert s["escalations"][0]["custody"] is True and rec["success"] is True


def test_analytic_mode_continues_without_handback():
    replies = [plan("grasp_and_place"), CONFIRM, HELD, plan("handback", target=False, dest=False)]
    rec, log = run_arm(variant("paper_analytic.yaml", replies=replies), Arm.CANDIDATE, "h2")
    s = wr(rec)
    assert [e["kind"] for e in s["escalations"]] == ["H2_empty_close", "analytic_continue"]
    assert s["escalations"][1]["handback"] == "plan_not_accepted"
    # after the analytic phase the arm holds still: no further policy steps after the first escalation's end
    assert s["policy_steps"] + s["primitive_steps"] >= HORIZON - 1   # the final step may end the episode before it returns
    assert s["primitive_steps"] > HORIZON - s["escalations"][0]["step"] - 5


def test_shadow_null_respondoff_is_bitwise_baseline():
    base, _ = run_arm(variant(replies=[]), Arm.BASELINE, "h1")
    rec, _ = run_arm(variant("paper_respondoff.yaml", replies=[plan("lift_and_place", clamped=True, at_dest=False),
                                                               CONFIRM]), Arm.NULL, "h1")
    assert rec["action_sha256"] == base["action_sha256"]



# ---- fix2: custody frames / abstain policy / send_state / eligibility ----------------------------------------------
def test_custody_without_wrist_sends_before_after_and_abstain_proceeds_on_measured():
    abstain = {"verdict": "abstain", "reason": "cannot tell"}
    rec, _ = run_arm(variant(replies=[plan("grasp_and_place"), CONFIRM, abstain]), Arm.CANDIDATE, "h2", wrist=False)
    (esc,) = wr(rec)["escalations"]
    assert esc["custody_views"] == ["before", "after"]
    assert esc["custody_confirm"] == "abstain_proceed_on_measured" and esc["planner_calls"][2]["outcome"] == "abstain"
    assert esc["place"] is not None and rec["success"] is True


def test_custody_abstain_handback_policy_and_not_held():
    abstain = {"verdict": "abstain", "reason": "x"}
    rec, _ = run_arm(variant(replies=[plan("grasp_and_place"), CONFIRM, abstain], custody_abstain_policy="handback"),
                     Arm.CANDIDATE, "h2")
    (esc,) = wr(rec)["escalations"]
    assert esc["handback"] == "custody_not_confirmed" and esc["custody_views"] == ["before", "after", "wrist"]
    rec, _ = run_arm(variant(replies=[plan("grasp_and_place"), CONFIRM, {"verdict": "not_held", "reason": "x"}]),
                     Arm.CANDIDATE, "h2")
    (esc,) = wr(rec)["escalations"]
    assert esc["handback"] == "custody_not_confirmed" and esc["custody_confirm"] == "not_held"


def test_custody_transport_failure_hands_back_even_with_measured_policy():
    rec, _ = run_arm(variant(replies=[plan("grasp_and_place"), CONFIRM, {"__transport__": "timeout"}]),
                     Arm.CANDIDATE, "h2")
    (esc,) = wr(rec)["escalations"]
    assert esc["handback"] == "custody_not_confirmed" and esc["custody_confirm"] == "transport:timeout"



@pytest.mark.parametrize("label", ["bottom drawer", "wooden cabinet shelf", "microwave", "caddy"])
def test_fix2b_enclosed_destination_label_hands_back_even_if_planner_says_open_top(label):
    p = plan("grasp_and_place")
    p["destination"]["label"] = label            # the planner (wrongly) keeps open_top true
    rec, _ = run_arm(variant(replies=[p, CONFIRM, HELD]), Arm.CANDIDATE, "h2")
    (esc,) = wr(rec)["escalations"]
    assert esc["custody"] is True and esc["handback"] == "destination_enclosed" and esc["place"] is None
    assert [c["role"] for c in esc["planner_calls"]] == ["plan", "witness"]     # no custody call, no place
