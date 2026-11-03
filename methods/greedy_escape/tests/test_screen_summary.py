"""screen_summary.py decision rules on synthetic map/screen folders (review greedy-escape-src-review.json S1-S4, F8, F13).
No real outcomes are read."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import screen_summary as ss  # noqa: E402

ZERO = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
STATE = [0.0] * 6 + [0.03, -0.03]


def moving(k):
    return [0.1 + 0.001 * k, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def greedy_actions(kind, n=40):
    if kind == "stall":                       # 5 moving, then identical zeros: frozen fire after action 19
        return [moving(k) for k in range(5)] + [ZERO] * (n - 5)
    return [moving(k) for k in range(n)]


def write_episode(arm_dir: Path, k: int, actions, success: bool):
    d = arm_dir / "vla_eval/episodes/LIBEROBenchmark_libero_10"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"task0000_ep{k:04d}_{'success' if success else 'fail'}.jsonl", "w") as f:
        for a in actions:
            f.write(json.dumps({"action": a, "state": STATE}) + "\n")


def write_records(arm_dir: Path, recs):
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def rec(init, success, steps, ps_per_step=0.75, ge=None, valid=True):
    acc = {"steps": steps, "policy_seconds": ps_per_step * steps, "extra": {}}
    if ge is not None:
        acc["extra"]["greedy_escape"] = ge
    return {"episode_idx": init, "valid": valid, "success": success if valid else None, "accounting": acc}


def ge_extra(fire=None, reasons=("fixed_point",), steps=40, width_missing=0):
    starts = [fire + 1] if fire is not None else []
    fires = [{"step": fire, "reasons": list(reasons), "window": True}] if fire is not None else []
    return {"window_starts": starts, "fires": fires, "width_missing_steps": width_missing,
            "escape_decisions": max(0, min(40, steps - 1 - (fire + 1))) if fire is not None else 0}


class World:
    """Synthetic map (task 0, inits 0..n-1) + screen job folders."""

    def __init__(self, tmp: Path):
        self.tmp, self.shards, self.jobs = tmp, tmp / "map", []

    def map(self, scenes):   # scenes: {init: (kind, greedy_success, scale_success)}
        g = self.shards / "baseline-t0/baseline"
        recs_g, recs_s = [], []
        for i, (kind, gs, sc) in scenes.items():
            acts = greedy_actions(kind)
            write_episode(g, i, acts, gs)
            recs_g.append(rec(i, gs, len(acts)))
            recs_s.append(rec(i, sc, 40))
        write_records(g, recs_g)
        write_records(self.shards / "candidate-t0/candidate", recs_s)

    def job(self, phase, arm, inits, rows, fingerprint=ss.FINGERPRINT):
        """rows: {init: dict(actions=..., success=..., ge=..., valid=..., ps=...)}"""
        jid = f"{phase}-{len(self.jobs)}"
        out = self.tmp / "screen" / jid
        arm_dir = out / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "manifest.json").write_text(json.dumps({"method_fingerprint": fingerprint, "namespace": "hs",
                                                           "case_namespace": ""}))
        recs = []
        for k, i in enumerate(inits):
            r = rows[i]
            write_episode(arm_dir, k, r["actions"], r["success"])
            recs.append(rec(i, r["success"], len(r["actions"]), r.get("ps", 0.75), r["ge"], r.get("valid", True)))
        write_records(arm_dir, recs)
        self.jobs.append({"id": jid, "phase": phase, "arms": arm, "task": 0, "inits": list(inits), "out": str(out)})

    def summary(self, cls=None, expected=None):
        return ss.summarize({"jobs": self.jobs}, "all", self.shards, cls, expected)


def escaped(kind):
    """candidate rows: greedy prefix through the fire (step 19), then different window actions."""
    a = greedy_actions(kind)
    return a[:20] + [[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]] * 20


def good_world(tmp, grid_success=(True, True, False, True), null_rows=None):
    w = World(tmp)
    # init 0: stalled greedy failure that SCALE solves; 1-3 untriggered
    w.map({0: ("stall", False, True), 1: ("move", True, False), 2: ("move", False, False), 3: ("move", True, True)})
    rows = {0: dict(actions=escaped("stall"), success=grid_success[0], ge=ge_extra(19))}
    for i in (1, 2, 3):
        rows[i] = dict(actions=greedy_actions("move"), success=grid_success[i], ge=ge_extra())
    w.job("grid", "candidate", [0, 1, 2, 3], rows)
    nr = null_rows or {0: dict(actions=greedy_actions("stall"), success=False, ge=ge_extra(19), ps=1.5)}
    w.job("null", "null", [0], nr)
    return w


def test_first_fire_replay_and_clean_world(tmp_path):
    assert ss.first_fire(greedy_actions("stall"), [STATE] * 40) == (19, ["fixed_point"])
    assert ss.first_fire(greedy_actions("move"), [STATE] * 40) == (None, None)
    s = good_world(tmp_path).summary(expected={"grid": 4})
    g = s["grid"]
    assert g["infrastructure"] == []
    assert g["vs_scale_w_l_net"] == [1, 0, 1] and g["vs_greedy_w_l_net"] == [1, 0, 1]
    assert g["diverged_at_first_window"] == [1, 1]
    assert s["null"]["status"] == "WITHIN_TOLERANCE" and s["null"]["inference_ratio"]["c"] == pytest.approx(2.0)
    assert g["inference_ratio"]["c"] == pytest.approx(1.0)
    # synthetic scenes t0i0/t0i1 carry RESCUE6 names: 2 "rescued", grid net +1 -> the GE7 extension is pending
    assert g["rescued_of_6"] == ["t0i0", "t0i1"]
    assert s["ge7_decision"].startswith("PENDING GE4 extension")


def test_s3_trigger_parity_mismatch_is_infrastructure(tmp_path):
    w = World(tmp_path)
    w.map({0: ("stall", False, True)})
    w.job("grid", "candidate", [0], {0: dict(actions=escaped("stall"), success=True, ge=ge_extra(21))})
    s = w.summary(expected={"grid": 1})
    assert any("trigger parity" in p for p in s["grid"]["infrastructure"])
    assert s["ge7_decision"].startswith("INFRASTRUCTURE (grid)")
    # reason mismatch too
    w2 = World(tmp_path / "b")
    w2.map({0: ("stall", False, True)})
    w2.job("grid", "candidate", [0], {0: dict(actions=escaped("stall"), success=True,
                                              ge=ge_extra(19, reasons=("empty_close",)))})
    assert any("trigger parity" in p for p in w2.summary(expected={"grid": 1})["grid"]["infrastructure"])


def test_s3_missing_fire_on_untriggered_expectation_and_null(tmp_path):
    w = World(tmp_path)
    w.map({0: ("stall", False, True)})
    # null record that never fired although its (native) greedy rows stall -> parity failure in the null gate
    w.job("null", "null", [0], {0: dict(actions=greedy_actions("stall"), success=False, ge=ge_extra())})
    s = w.summary()
    assert s["null"]["status"] == "INFRASTRUCTURE"


def test_f8_fire_on_last_action_is_unobservable(tmp_path):
    w = World(tmp_path)
    w.map({0: ("move", True, True)})
    acts = greedy_actions("move")
    w.job("grid", "candidate", [0], {0: dict(actions=acts, success=True, ge=ge_extra())})
    row = w.summary(expected={"grid": 1})["rows"][0]
    assert row["trigger_parity"] and row["window_decisions"] == 0
    assert ss.window_decisions([10, 30], 45) == 35        # [10,50) U [30,70) clipped at 45
    assert ss.window_decisions([5], 30) == 25


def test_f13_wrong_fingerprint_is_refused(tmp_path):
    w = World(tmp_path)
    w.map({0: ("move", True, True)})
    w.job("grid", "candidate", [0], {0: dict(actions=greedy_actions("move"), success=True, ge=ge_extra())},
          fingerprint="0" * 64)
    assert any("manifest" in p for p in w.summary(expected={"grid": 1})["grid"]["infrastructure"])


def test_retention_fault_and_missing_states(tmp_path):
    w = World(tmp_path)
    w.map({0: ("move", True, True), 1: ("move", True, True)})
    changed = greedy_actions("move")
    changed[7] = [0.9, 0, 0, 0, 0, 0, -1.0]
    w.job("grid", "candidate", [0, 1], {0: dict(actions=changed, success=True, ge=ge_extra()),
                                        1: dict(actions=greedy_actions("move"), success=True,
                                                ge=ge_extra(width_missing=40))})
    infra = w.summary(expected={"grid": 2})["grid"]["infrastructure"]
    assert any("t0i0 retention" in p for p in infra) and any("t0i1 missing states" in p for p in infra)


# ---- decide(): S1, S2, S4 ordering ----------------------------------------------------------------------------
def G(net=2, rescued=("t0i0", "t0i1"), infra=(), c=1.0):
    return {"infrastructure": list(infra), "vs_scale_w_l_net": [net + 1, 1, net], "rescued_of_6": list(rescued),
            "inference_ratio": {"drop_gt_4": c > 4.0}}


def C(net=0, infra=(), c=1.0):
    return {"infrastructure": list(infra), "vs_scale_w_l_net": [net + 1, 1, net], "inference_ratio": {"drop_gt_4": c > 4.0}}


N_OK = {"status": "WITHIN_TOLERANCE"}


def test_decide_s1_confirm_validity_blocks_advance():
    assert ss.decide(G(2), C(3, infra=["t0i5 invalid"]), N_OK).startswith("INFRASTRUCTURE (confirm)")
    assert ss.decide(G(2), C(3, infra=["59 episodes, expected 60"]), N_OK).startswith("INFRASTRUCTURE (confirm)")
    assert ss.decide(G(2), C(0), N_OK).startswith("ADVANCE (GE4 passed")
    assert ss.decide(G(2), C(-1), N_OK) == "DO NOT ADVANCE (GE4 failed)"
    assert ss.decide(G(1), C(2), N_OK).startswith("ADVANCE (GE7 extension")
    assert ss.decide(G(1), C(1), N_OK) == "DO NOT ADVANCE (GE7 extension rule)"
    assert ss.decide(G(2), None, N_OK).startswith("PENDING GE4")


def test_decide_s2_null_gate():
    assert ss.decide(G(2), C(0), {"status": "OUT_OF_TOLERANCE"}).startswith("NOT ADVANCE YET")
    assert ss.decide(G(2), C(0), None).startswith("PENDING null")
    assert ss.decide(G(2), C(0), {"status": "INFRASTRUCTURE", "infrastructure": [], "X": 1}).startswith(
        "INFRASTRUCTURE (null)")
    assert ss.decide(G(2), C(0), {"status": "PENDING (evaluator R/X classification of differing scenes)",
                                  "unclassified": ["t7i2"]}).startswith("PENDING null R/X")


def test_decide_d1_null_precedes_every_quality_verdict():
    bad = {"status": "INFRASTRUCTURE", "infrastructure": ["t3i3 invalid"], "X": 0}
    pend = {"status": "PENDING (evaluator R/X classification of differing scenes)", "unclassified": ["t3i3"]}
    for g in (G(0), G(3, rescued=("t0i0",)), G(3, c=4.5), G(2), G(1)):
        assert ss.decide(g, None, bad).startswith("INFRASTRUCTURE (null)")
        assert ss.decide(g, None, None).startswith("PENDING null")
        assert ss.decide(g, None, pend).startswith("PENDING null R/X")
    assert ss.decide(G(0), None, {"status": "INFRASTRUCTURE"}).startswith("INFRASTRUCTURE (null)")
    # grid infrastructure still comes first; OUT_OF_TOLERANCE does not invalidate a DROP
    assert ss.decide(G(0, infra=["x"]), None, bad).startswith("INFRASTRUCTURE (grid)")
    assert ss.decide(G(0), None, {"status": "OUT_OF_TOLERANCE"}) == "DROP (R8 F1: net vs SCALE <= 0)"


def test_null_classification_file_forms(tmp_path):
    rows_diff = {0: dict(actions=escaped("stall"), success=True, ge=ge_extra(19))}
    w = good_world(tmp_path, null_rows=rows_diff)
    rx = {"t0i0": {"class": "X", "replay": "eval/greedy-escape/null-replay/t0i0", "rule": "ge-null-rx-rule.md"}}
    s = w.summary(rx, expected={"grid": 4})
    assert s["null"]["status"] == "INFRASTRUCTURE" and s["ge7_decision"].startswith("INFRASTRUCTURE (null)")
    s = w.summary({"t0i0": {"class": "R"}}, expected={"grid": 4})
    assert s["null"]["status"] == "WITHIN_TOLERANCE"
    assert w.summary(None, expected={"grid": 4})["ge7_decision"].startswith("PENDING null R/X")


def test_decide_s4_and_f1_order():
    assert ss.decide(G(3, c=4.5), None, N_OK) == "DROP (R8 F2: inference ratio > 4.0)"
    assert ss.decide(G(0), None, N_OK) == "DROP (R8 F1: net vs SCALE <= 0)"
    assert ss.decide(G(3, rescued=("t0i0",)), None, N_OK).startswith("MECHANISM REJECTED")
    assert ss.decide(G(3, infra=["x"]), C(0), N_OK).startswith("INFRASTRUCTURE (grid)")
    assert ss.decide(G(2), C(0, c=5.0), N_OK).startswith("DROP (R8 F2")


def test_null_tolerance_rules(tmp_path):
    rows_diff = {0: dict(actions=escaped("stall"), success=True, ge=ge_extra(19))}   # differs after the fire
    w = good_world(tmp_path, null_rows=rows_diff)
    s = w.summary(expected={"grid": 4})
    assert s["null"]["status"].startswith("PENDING")
    assert w.summary({"t0i0": "X"}, expected={"grid": 4})["null"]["status"] == "INFRASTRUCTURE"
    r = w.summary({"t0i0": "R"}, expected={"grid": 4})["null"]
    assert r["status"] == "WITHIN_TOLERANCE" and r["discordant_vs_greedy"] == 1 and r["net_vs_greedy"] == 1
    n = ss.null_summary([{"scene": f"t0i{i}", "valid": True, "manifest_ok": True, "manifest_problem": "",
                          "trigger_parity": True, "null_equals_greedy_record": True, "success": True, "greedy": False,
                          "policy_seconds": 1.0, "steps": 1, "greedy_policy_seconds": 1.0, "greedy_steps": 1}
                         for i in range(3)], 3, {})
    assert n["discordant_vs_greedy"] == 3 and n["status"] == "OUT_OF_TOLERANCE"
