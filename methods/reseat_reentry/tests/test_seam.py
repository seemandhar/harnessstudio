"""reseat-reentry through the real handler seam (review K3): in-process vla-eval (RGBDTwoBlockBench: public states +
RGB-D + calibration) -> HarnessServer -> SpyPolicy, every arm, scripted planner (no network). Covers fire + re-seat
(+ the policy then grasping the remaining block), abstain, articulated abstain (kind and label guard), witness
reject, transport timeout, workspace replan (accept and fail), two fires, late fire (ineligible), the shadow null,
respondoff bitwise plain `scale`, unfired bitwise plain `scale`, online fire == offline replay, no primitive close,
<= 45 primitive steps."""
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
from rr_fake_bench import OBJS, TOP, SpyPolicy, pixel_of  # noqa: E402

from harnessstudio.contracts import load_contract  # noqa: E402
from harnessstudio.core import registry  # noqa: E402
from harnessstudio.core.arms import Arm  # noqa: E402
from harnessstudio.core.card import config_from_card, load_card  # noqa: E402
from harnessstudio.core.method import ModuleRef  # noqa: E402
from harnessstudio.profiles import load_profile  # noqa: E402
from harnessstudio.server.harness_server import HarnessServer  # noqa: E402
from harnessstudio.verify.stack import serve  # noqa: E402

import hs_reseat_reentry as R  # noqa: E402
from hs_reseat_reentry.controller import scale_sampler  # noqa: E402
from hs_reseat_reentry.triggers_gap import replay  # noqa: E402


def contract(horizon):
    c = load_contract("fake")
    return dataclasses.replace(c, public_keys=("images", "task_description", "states", "depth", "camera"),
                               state_key="states", default_horizon=horizon,
                               eef_position={"path": ["states"], "slice": [0, 3]})


def run_arm(config, arm: Arm, scenario="gap", horizon=400, dtype=np.float32, out=None):
    from vla_eval.api import run
    policy = SpyPolicy(scenario, dtype)
    with serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract(horizon), arm=arm,
                                profile=load_profile("fake-t2"), scripted_llm=True, out_dir=out)
        with serve(harness) as hh:
            cfg = {"server": {"url": hh.url, "timeout": 60.0},
                   "benchmarks": [{"benchmark": "rr_fake_bench:RGBDTwoBlockBench", "mode": "sync",
                                   "episodes_per_task": 1, "params": {"horizon": horizon}}]}
            with tempfile.TemporaryDirectory(prefix="rr-stack-") as tmp, contextlib.redirect_stdout(io.StringIO()):
                cfg["output_dir"] = tmp
                run(cfg, docker=False, no_save=False)
            harness.wait_idle()
    (rec,) = harness.records
    assert rec["valid"], rec["invalid_reason"]
    return rec, policy.log


def variant(card="paper.yaml", replies=(), **over):
    path = os.path.join(os.path.dirname(R.__file__), card)
    params = dict(load_card(path)["params"])
    params["planner"] = {**params["planner"], "backend": "scripted", "scripted_responses": list(replies)}
    for k, v in over.items():
        params[k] = {**params[k], **v} if isinstance(v, dict) else v
    return config_from_card(path, controller=ModuleRef(R.FACTORY, params))


def point(label="green block", where=None, kind="object", action="reseat"):
    r, c = pixel_of(where or [OBJS["B"][0], OBJS["B"][1], TOP])
    return {"remaining": [{"label": label, "kind": kind}],
            "target": {"label": label, "row": r, "col": c} if action == "reseat" else None, "action": action}


CONFIRM = {"verdict": "confirm", "seen": "green block", "reason": "scripted"}


def rr(rec):
    return rec["accounting"]["extra"]["reseat_reentry"]


# ---- fire + re-seat ---------------------------------------------------------------------------------------------
def test_fire_reseat_then_policy_grasps_remaining_block():
    rec, log = run_arm(variant(replies=[point(), CONFIRM]), Arm.CANDIDATE, "gap")
    s = rr(rec)
    (esc,) = s["escalations"]
    assert s["fires"][0]["kind"] == "post_release_gap" and s["fires"][0]["end_kind"] == "release"
    assert [c["role"] for c in esc["planner_calls"]] == ["point", "witness"]
    assert [c["outcome"] for c in esc["planner_calls"]] == ["accept", "accept"]
    assert esc["handback"] == "after_reseat" and esc["reseat"]["reached"]
    assert 0 < esc["reseat"]["steps"] <= 45 and s["closes_commanded_by_primitive"] == 0
    hz = esc["target"]["hover_xyz"]
    assert abs(hz[2] - (TOP + 0.10)) < 0.01 and abs(hz[0] - OBJS["B"][0]) < 0.012
    assert rec["success"] is True                    # the POLICY picked block B after the re-seat
    assert all(e["sampler"] == scale_sampler() for e in log)
    # the fire step equals the offline replay of the reference trajectory (plain scale, same key)
    ref, _ = run_arm(registry.load("scale"), Arm.CANDIDATE, "gap")
    assert rec["action_sha256"][: s["fires"][0]["step"]] == ref["action_sha256"][: s["fires"][0]["step"]]


@pytest.mark.parametrize("reply,why,outcome", [
    (point(action="none"), "point_not_accepted", "abstain"),
    (point(kind="articulated_part"), "point_not_accepted", "articulated"),
    (point(label="bottom drawer"), "point_not_accepted", "invalid"),
    (point(label="green block knob"), "point_not_accepted", "articulated"),
])
def test_no_motion_on_abstain_or_articulated(reply, why, outcome):
    rec, _ = run_arm(variant(replies=[reply]), Arm.CANDIDATE, "gap")
    (esc,) = rr(rec)["escalations"]
    assert esc["handback"] == why and esc["planner_calls"][0]["outcome"] == outcome
    assert rr(rec)["primitive_steps"] == 0


def test_witness_reject_and_transport_hand_back_without_motion():
    rec, _ = run_arm(variant(replies=[point(), {"verdict": "wrong_object", "seen": "tray", "reason": "x"}]),
                     Arm.CANDIDATE, "gap")
    (esc,) = rr(rec)["escalations"]
    assert esc["handback"] == "witness_wrong_object" and rr(rec)["primitive_steps"] == 0
    rec, _ = run_arm(variant(replies=[{"__transport__": "timeout"}]), Arm.CANDIDATE, "gap")
    (esc,) = rr(rec)["escalations"]
    assert esc["planner_calls"][0]["status"] == "timeout" and esc["planner_calls"][0]["outcome"] is None
    assert esc["handback"] == "point_not_accepted" and rr(rec)["primitive_steps"] == 0


def test_workspace_replan_then_reseat_and_replan_failure():
    far = point(where=[0.25, -0.30, 0.90])   # > 0.5 m from the eef -> outside_workspace -> one replan
    rec, _ = run_arm(variant(replies=[far, point(), CONFIRM]), Arm.CANDIDATE, "gap")
    (esc,) = rr(rec)["escalations"]
    assert [c["role"] for c in esc["planner_calls"]] == ["point", "point", "witness"]
    assert esc["planner_calls"][0].get("geometry_reject") and esc["handback"] == "after_reseat"
    rec, _ = run_arm(variant(replies=[far, far]), Arm.CANDIDATE, "gap")
    (esc,) = rr(rec)["escalations"]
    assert esc["handback"].startswith("workspace_check_failed") and rr(rec)["primitive_steps"] == 0


def test_two_fires_and_call_cap():
    rec, _ = run_arm(variant(replies=[point(action="none"), point(action="none")]), Arm.CANDIDATE, "gap2", horizon=650)
    s = rr(rec)
    assert len(s["fires"]) == 2 and s["fires"][1]["carry_end"] > s["fires"][0]["step"]
    assert s["planner"]["calls"] == 2


def test_late_fire_is_ineligible():
    rec, _ = run_arm(variant(replies=[point(), CONFIRM]), Arm.CANDIDATE, "gap_late", horizon=330)
    assert rr(rec)["fires"] == []


# ---- bitwise SCALE ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["gap", "idle"])
def test_respondoff_is_bitwise_plain_scale(scenario):
    ref, ref_log = run_arm(registry.load("scale"), Arm.CANDIDATE, scenario)
    rec, log = run_arm(variant("paper_respondoff.yaml", replies=[point(), CONFIRM]), Arm.CANDIDATE, scenario)
    assert rec["action_sha256"] == ref["action_sha256"] and rec["obs_sha256"] == ref["obs_sha256"]
    assert [(e["seeds"], e["sampler"]) for e in log] == [(e["seeds"], e["sampler"]) for e in ref_log]
    if scenario == "gap":
        (esc,) = rr(rec)["escalations"]
        assert esc["handback"] == "respondoff" and esc["calls"] == 2 and rr(rec)["primitive_steps"] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unfired_candidate_is_bitwise_plain_scale(dtype):
    ref, _ = run_arm(registry.load("scale"), Arm.CANDIDATE, "idle", dtype=dtype)
    rec, _ = run_arm(variant(replies=[]), Arm.CANDIDATE, "idle", dtype=dtype)
    assert not rr(rec)["fires"]
    assert rec["action_sha256"] == ref["action_sha256"] and rec["obs_sha256"] == ref["obs_sha256"]


def test_online_fire_equals_offline_replay_of_recorded_states():
    rec, _ = run_arm(variant("paper_respondoff.yaml", replies=[point(), CONFIRM]), Arm.CANDIDATE, "gap")
    # rebuild the post-step proprio the recorder would store from the harness's own trace: rerun the bench offline
    from rr_fake_bench import RGBDTwoBlockBench
    from rr_fake_bench import SpyPolicy as _P  # noqa: F401
    online = rr(rec)["fires"][0]["step"]
    b = RGBDTwoBlockBench(horizon=400)
    b.reset({"name": "t"})
    pol = SpyPolicy("gap")

    class _Ctx:
        session_id = "x"
    S = []
    for _ in range(400):
        obs = b.make_obs(b._raw(), {"name": "t"})
        (a, _e), = pol.sample_chunks(obs, [0], None, (), _Ctx())
        r = b.step({"actions": a[0]})
        S.append(b.make_obs(r.obs, {"name": "t"})["states"].astype(np.float32).astype(np.float64))
        if r.done:
            break
    assert replay(np.array(S), 400)[0]["step"] == online


# ---- Campaign shadow null -----------------------------------------------------------------------------------------
def test_shadow_null_executes_native_actions_and_pays_its_calls():
    base, _ = run_arm(variant(replies=[]), Arm.BASELINE, "gap")
    rec, log = run_arm(variant(replies=[point(), CONFIRM] * 2), Arm.NULL, "gap")
    assert rec["action_sha256"] == base["action_sha256"]
    s = rr(rec)
    assert s["fires"] and s["planner"]["calls"] >= 1
    shadow = [e for e in log if e["sampler"] is not None]
    native = [e for e in log if e["sampler"] is None]
    assert shadow and native and {e["session"] for e in shadow}.isdisjoint({e["session"] for e in native})
    assert all(e["sampler"] == scale_sampler() for e in shadow)


def test_reseat_budget_exhausted_stops_at_the_computed_budget():
    far = point(where=[0.15, -0.22, 0.90])     # bare table far from the release point
    # tiny budget via card params (min = max = 20) so the stop is forced; rule = clip(steps_left - 105, 20, 20)
    rec, _ = run_arm(variant(replies=[far, CONFIRM], motion={"min_steps": 20, "max_steps": 20}), Arm.CANDIDATE, "gap")
    (esc,) = rr(rec)["escalations"]
    assert esc["reseat"]["budget"] == 20 and esc["reseat"]["steps"] == 20 and esc["reseat"]["budget_exhausted"]
    assert esc["handback"] == "reseat_stopped" and rr(rec)["reseats"][0]["delivered"] is False
    assert rr(rec)["primitive_steps"] == 20 and rr(rec)["closes_commanded_by_primitive"] == 0


def test_reseat_budget_follows_steps_left():
    rec, _ = run_arm(variant(replies=[point(), CONFIRM]), Arm.CANDIDATE, "gap")
    (esc,) = rr(rec)["escalations"]
    assert esc["reseat"]["budget"] == min(90, max(45, esc["steps_left"] - 105))
    assert rr(rec)["reseats"][0]["budget"] == esc["reseat"]["budget"]
