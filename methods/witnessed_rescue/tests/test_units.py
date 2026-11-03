"""Unit tests: triggers (H2 / H1 / drop, online == offline replay), back-projection with the published horizontal
flip, witness geometry, custody predicate, grasp yaw, planner schema/validation/label rules, budget accounting,
prompt hygiene (no benchmark/task/success words), sampler identity with the plain scale method."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wr_fake_bench import OBJ, TRAY, RGBDGripperBench, pixel_of, world_to_pixel  # noqa: E402

from hs_witnessed_rescue import motion as M  # noqa: E402
from hs_witnessed_rescue import perception as P  # noqa: E402
from hs_witnessed_rescue import planner as PL  # noqa: E402
from hs_witnessed_rescue import triggers as T  # noqa: E402
from hs_witnessed_rescue.controller import scale_sampler  # noqa: E402


# ---- triggers ---------------------------------------------------------------------------------------------------
def traj(spec):
    """spec: list of (n, action7, (z, width)) segments -> A, S (post-step rows)."""
    A, S = [], []
    for n, a, (z, w) in spec:
        for _ in range(n):
            A.append(list(a))
            S.append([0.0, 0.0, z, 0.0, 0.0, 0.0, w / 2, -w / 2])
    return np.array(A, dtype=float), np.array(S, dtype=float)


OPEN = [0, 0, 0, 0, 0, 0, -1.0]
CLOSE = [0, 0, 0, 0, 0, 0, 1.0]


def test_h2_empty_close_fires_at_first_observable_decision():
    A, S = traj([(10, OPEN, (1.0, 0.08)), (1, CLOSE, (1.0, 0.05)), (1, CLOSE, (1.0, 0.02)), (6, CLOSE, (1.0, 0.0))])
    fires = T.replay(A, S, horizon=520)
    assert len(fires) == 1 and fires[0]["kind"] == "H2_empty_close"
    # close run starts at 10; width <= 4 mm first at row 12 (lat 2) -> fire decision max(12+1, 10+3) = 13
    assert fires[0]["close"] == 10 and fires[0]["step"] == 13 and fires[0]["latency"] == 2


def test_h2_needs_latency_and_ignores_held_close():
    A, S = traj([(5, OPEN, (1.0, 0.08)), (20, CLOSE, (1.0, 0.04))])
    assert T.replay(A, S) == []
    # empty only after 7 steps -> outside the 6-step latency
    A, S = traj([(5, OPEN, (1.0, 0.08)), (7, CLOSE, (1.0, 0.03)), (5, CLOSE, (1.0, 0.0))])
    assert T.replay(A, S) == []


def test_h2_disabled_after_first_lifted_carry():
    A1, S1 = traj([(3, OPEN, (1.0, 0.08)), (5, CLOSE, (1.0, 0.04)), (10, CLOSE, (1.05, 0.04)), (5, OPEN, (1.05, 0.08))])
    A2, S2 = traj([(5, CLOSE, (1.05, 0.0))])
    A, S = np.vstack([A1, A2]), np.vstack([S1, S2])
    assert T.replay(A, S) == []


def test_h1_settled_clamp_fires_on_proposed_open():
    A, S = traj([(5, OPEN, (1.0, 0.08)), (12, CLOSE, (1.0, 0.03)), (3, OPEN, (1.0, 0.06))])
    fires = T.replay(A, S)
    assert [f["kind"] for f in fires] == ["H1_settled_clamp"] and fires[0]["step"] == 17 and fires[0]["close"] == 5
    # online: same decision with the proposal
    mon = T.TriggerMonitor()
    assert mon.check(A[:17], S[:17], A[17], 400).step == 17


@pytest.mark.parametrize("mutate", ["short", "thin", "lifted", "rotated"])
def test_h1_exclusions(mutate):
    n = 8 if mutate == "short" else 12
    w = 0.008 if mutate == "thin" else 0.03
    A, S = traj([(5, OPEN, (1.0, 0.08)), (n, CLOSE, (1.0, w)), (3, OPEN, (1.0, 0.06))])
    if mutate == "lifted":
        S[10:17, 2] = 1.01
    if mutate == "rotated":
        A[5:17, 5] = 0.2
    assert [f for f in T.replay(A, S) if f["kind"] == "H1_settled_clamp"] == []


def test_drop_only_when_enabled():
    A, S = traj([(3, OPEN, (1.0, 0.08)), (5, CLOSE, (1.0, 0.03)), (5, CLOSE, (1.05, 0.03)), (3, CLOSE, (1.05, 0.0))])
    assert T.replay(A, S) == []
    fires = T.replay(A, S, params={"drop": True})
    assert fires and fires[0]["kind"] == "drop" and fires[0]["step"] == 14


def test_eligibility_rearm_and_cap():
    A, S = traj([(10, OPEN, (1.0, 0.08)), (4, CLOSE, (1.0, 0.0))])
    f = T.replay(A, S, horizon=10 + 3 + 100)       # steps_left 100 < 120
    assert f and f[0]["eligible"] is False
    # three separate empty closes 25 steps apart -> 3 escalations, the 4th is capped
    seg = [(18, OPEN, (1.0, 0.08)), (7, CLOSE, (1.0, 0.0))]
    A, S = traj(seg * 5)
    fires = T.replay(A, S, horizon=2000)
    assert len([x for x in fires if x["eligible"]]) == 3
    # re-arm: a second empty close starting < 20 steps after the hand-back is ignored
    A, S = traj([(5, OPEN, (1.0, 0.08)), (4, CLOSE, (1.0, 0.0)), (3, OPEN, (1.0, 0.08)), (4, CLOSE, (1.0, 0.0))])
    assert len(T.replay(A, S)) == 1


def test_online_equals_offline_on_random_trajectories():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = 200
        g = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        g = np.repeat(g[:20], 10)
        A = np.zeros((n, 7))
        A[:, 6] = g
        S = np.zeros((n, 8))
        S[:, 2] = 1.0 + np.cumsum(rng.normal(0, 0.002, n))
        w = np.where(g > 0, rng.choice([0.0, 0.03], n), 0.08)
        S[:, 6], S[:, 7] = w / 2, -w / 2
        off = T.replay(A, S, horizon=520)
        mon, on = T.TriggerMonitor(), []
        for d in range(n):
            f = mon.check(A[:d], S[:d], A[d], 520 - d)
            if f is not None:
                on.append(f.to_json())
                if f.eligible:
                    mon.handback(d)
        assert on == off


# ---- perception -------------------------------------------------------------------------------------------------
def bench_obs(**raw_over):
    b = RGBDGripperBench()
    raw = b.reset({"name": "t"})
    raw.update(raw_over)
    return b.make_obs(raw, {"name": "t"})


def test_backprojection_round_trip_with_published_flip():
    obs = bench_obs()
    rgb, depth, cam = P.views(obs, "agentview")
    # calibration-frame image is the published one mirrored horizontally
    assert np.array_equal(rgb, np.asarray(obs["images"]["agentview"])[:, ::-1])
    wmap = P.world_map(depth, cam)
    r, c = pixel_of([OBJ["center"][0], OBJ["center"][1], OBJ["top"]], scale=1)
    assert np.allclose(wmap[r, c], [OBJ["center"][0], OBJ["center"][1], OBJ["top"]], atol=0.01)
    pr, pc = P.project([0.1, -0.05, 0.9], cam)
    assert np.allclose(wmap[int(round(pr)), int(round(pc))], [0.1, -0.05, 0.9], atol=0.01)


def test_locate_object_and_rejections():
    obs = bench_obs()
    pk = P.Packet.capture(obs, 0)
    r, c = pixel_of([OBJ["center"][0], OBJ["center"][1], OBJ["top"]])
    loc = P.locate(pk, r, c, {})
    assert loc["ok"], loc
    assert abs(loc["z_table"] - 0.90) < 0.006 and abs(loc["height"] - 0.08) < 0.01
    assert np.allclose(loc["center_xyz"][:2], OBJ["center"], atol=0.012)
    assert max(loc["extent_xy"]) < 0.07
    # background
    r2, c2 = pixel_of([0.25, -0.2, 0.90])
    assert P.locate(pk, r2, c2, {})["reason"].startswith("background")
    # robot: point at the projected end effector
    # the projected end effector over bare table: background (the depth sees the table, not a gripper)
    r3, c3 = pixel_of([0.0, 0.0, 1.05])
    assert P.locate(pk, r3, c3, {})["reason"].startswith("background")
    # a point the depth sees inside the gripper volume is rejected as robot
    obs2 = bench_obs(p=np.array([OBJ["center"][0], OBJ["center"][1], OBJ["top"] + 0.01]))
    pk2 = P.Packet.capture(obs2, 0)
    r5, c5 = pixel_of([OBJ["center"][0], OBJ["center"][1], OBJ["top"]])
    assert "robot" in P.locate(pk2, r5, c5, {})["reason"]
    # ... but the same object seen 3.5 cm below a hovering gripper is accepted
    obs3 = bench_obs(p=np.array([OBJ["center"][0], OBJ["center"][1], OBJ["top"] + 0.035]))
    assert P.locate(P.Packet.capture(obs3, 0), r5, c5, {})["ok"]
    # outside the image
    assert P.locate(pk, 9999, 5, {})["reason"] == "pixel_outside_image"
    # destination: z_surface of the tray
    r4, c4 = pixel_of([TRAY["center"][0], TRAY["center"][1], 0.90])
    d = P.locate(pk, r4, c4, {}, role="destination")
    assert d["ok"] and abs(d["z_surface"] - 0.90) < 0.005
    col = P.free_column(pk, d, 0.08, [0.0, 0.0, 1.05], {})
    assert col["free"]
    blocked = P.free_column(pk, {"center_xyz": [*OBJ["center"], 0.9], "z_surface": 0.90}, 0.08, [0.0, 0.0, 1.05], {})
    assert not blocked["free"]


def test_no_depth_means_no_motion_inputs():
    obs = bench_obs()
    obs.pop("depth")
    loc = P.locate(P.Packet.capture(obs, 0), 10, 10, {})
    assert not loc["ok"] and loc["reason"].startswith("perception_unavailable")


def test_marked_views_shapes():
    pk = P.Packet.capture(bench_obs(), 3)
    full, crop = P.marked_views(pk, 40, 50, {})
    assert full.shape == (256, 256, 3) and crop.shape == (320, 320, 3)
    assert (full == np.array([255, 0, 0])).all(axis=-1).sum() > 20


# ---- motion ---------------------------------------------------------------------------------------------------
def test_custody_predicate():
    p = M.DEFAULTS
    assert M.custody(0.03, 0.029, 0.055, p)["custody"]
    assert not M.custody(0.003, 0.003, 0.06, p)["custody"]          # empty
    assert not M.custody(0.03, 0.026, 0.06, p)["custody"]           # slipping
    assert not M.custody(0.03, 0.03, 0.04, p)["custody"]            # not lifted


def test_z_close_and_grasp_yaw():
    assert M.z_close(0.98, 0.90, 0.035) == pytest.approx(0.945)
    assert M.z_close(0.92, 0.90, 0.035) == pytest.approx(0.91)
    y = M.grasp_yaw(0.0, np.pi / 2, np.pi / 2)
    assert abs(y) < 1e-9
    y = M.grasp_yaw(3.0, 0.0, np.pi / 2)
    assert abs(M.wrap(y - 3.0)) <= np.pi / 2 + 1e-9


def test_motion_servo_converges_on_fake_dynamics():
    state = {"p": np.array([0.0, 0.0, 1.0]), "yaw": 0.0}

    async def step(a):
        state["p"] = state["p"] + 0.0082 * np.asarray(a[:3], dtype=float)
        state["yaw"] += 0.08 * float(a[5])
        return 1

    def st():
        return np.array([*state["p"], 0, 0, state["yaw"], 0.04, -0.04])

    m = M.Motion(step, st, lambda: False)
    res = asyncio.run(m.move_to([0.2, -0.1, 1.05], yaw=0.5))
    assert res["error"] < 0.012 and abs(state["yaw"] - 0.5) < 0.06


# ---- planner -------------------------------------------------------------------------------------------------
def _required_everywhere(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert set(schema["required"]) == set(schema["properties"]) and schema["additionalProperties"] is False
        for v in schema.values():
            _required_everywhere(v)
    elif isinstance(schema, list):
        for v in schema:
            _required_everywhere(v)


def test_schemas_are_strict():
    for s in (PL.PLAN_SCHEMA, PL.VERDICT_SCHEMA, PL.CUSTODY_SCHEMA):
        _required_everywhere(s)
        json.dumps(s)


def test_validate_plan_and_labels():
    instr = "put both the alphabet soup and the tomato sauce in the basket"
    ok = {"action": "grasp_and_place", "target": {"label": "alphabet soup", "row": 10, "col": 20},
          "destination": None}
    assert PL.validate_plan(ok, "H2_empty_close", instr, 512) is None
    assert "not allowed" in PL.validate_plan({**ok, "action": "lift_and_place"}, "H2_empty_close", instr, 512)
    assert "outside" in PL.validate_plan({**ok, "target": {"label": "x", "row": 600, "col": 1}}, "H2_empty_close",
                                         instr, 512)
    assert "needs a target" in PL.validate_plan({**ok, "target": None}, "H2_empty_close", instr, 512)
    assert PL.label_ok("the tomato sauce", instr) and not PL.label_ok("red mug", instr)


def test_prompts_carry_no_benchmark_task_or_success_words():
    instr = "turn on the stove and put the moka pot on it"
    texts = [PL.SYSTEM, PL.plan_prompt(instr, "H2_empty_close", 512), PL.plan_prompt(instr, "H1_settled_clamp", 512),
             PL.witness_prompt(instr, "moka pot", False), PL.custody_prompt(instr, "moka pot")]
    for t in texts:
        rest = t.replace(instr, "")
        # K1 (fix2b): generic destination categories (drawer, cabinet, shelf, microwave, oven, caddy) are allowed
        for w in ("libero", "success", "succeed", "episode", "benchmark", "reward", "task list", "bddl", "basket",
                  "stove", "plate"):
            assert w not in rest.lower(), (w, t[:80])


def test_planner_session_budget_and_status_mapping():
    replies = [{"__transport__": "timeout"}, {"verdict": "confirm", "seen": "x", "reason": "y"}, {"x": 1}]
    s = PL.PlannerSession({"backend": "scripted", "scripted_responses": replies, "max_calls_episode": 2})

    async def go():
        a = await s.ask("plan", "p", [], PL.VERDICT_SCHEMA)
        b = await s.ask("witness", "p", [], PL.VERDICT_SCHEMA)
        c = await s.ask("custody", "p", [], PL.VERDICT_SCHEMA)
        return a, b, c

    a, b, c = asyncio.run(go())
    assert a.status == "timeout" and b.status == "ok" and c.status == "budget_refused"
    assert s.calls == 2 and s.summary()["by_status"] == {"timeout": 1, "ok": 1, "budget_refused": 1}


def test_parse_error_is_transport():
    s = PL.PlannerSession({"backend": "scripted", "scripted_responses": [{"verdict": "maybe"}]})
    res = asyncio.run(s.ask("witness", "p", [], PL.VERDICT_SCHEMA,
                            validate=lambda o: None if o.get("verdict") in ("confirm", "abstain") else "bad verdict"))
    assert res.status == "parse_error"


def test_astra_refused():
    with pytest.raises(ValueError):
        PL.PlannerSession({"model": "gpt-6-astra"})


def test_sampler_is_the_plain_scale_sampler():
    from harnessstudio.core import registry
    assert scale_sampler() == registry.load("scale").slots["sampler"].to_json()


def test_source_has_no_task_lookup():
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hs_witnessed_rescue")
    words = ("alphabet", "tomato", "cream cheese", "moka", "ketchup", "libero_10/",
             "KITCHEN_SCENE", "LIVING_ROOM", "(On ", "(In ", "task_id", "bddl")
    for root, _, files in os.walk(pkg):
        if "vendor" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith((".py", ".yaml")):
                text = open(os.path.join(root, f)).read()
                for w in words:
                    if f.endswith(".yaml") and w in ("alphabet", "tomato"):
                        continue   # the tier-1 fake instruction of the card (fake stack only)
                    assert w not in text, (f, w)


# ---- fix1 (review P1, P2/PC10, P3, P4, PC11) ----------------------------------------------------------------------
def _tray_dest(pk):
    r, c = pixel_of([TRAY["center"][0], TRAY["center"][1], 0.90])
    d = P.locate(pk, r, c, {}, role="destination")
    assert d["ok"]
    return d


def test_free_column_not_vacuous_with_eef_near_destination():
    tx, ty = TRAY["center"]
    for dx in (0.0, 0.03, 0.05):
        ee = np.array([tx + dx, ty, 1.06])
        pk = P.Packet.capture(bench_obs(p=ee), 0)
        col = P.free_column(pk, _tray_dest(pk), 0.08, ee, {})
        assert col["column_points"] >= P.DEFAULTS["column_min_points"] and col["free"], col
    # an object standing at the destination under a raised gripper is still seen (occupied)
    ee = np.array([tx + 0.02, ty, 1.12])
    obs = bench_obs(p=ee, obj_c=np.array([tx, ty]))
    pk = P.Packet.capture(obs, 0)
    col = P.free_column(pk, _tray_dest(P.Packet.capture(bench_obs(), 0)), 0.08, ee, {})
    assert not col["free"] and col["reason"] == "occupied", col


def test_free_column_unobserved_is_not_free():
    tx, ty = TRAY["center"]
    ee = np.array([0.0, 0.0, 1.05])
    obs = bench_obs()
    d = _tray_dest(P.Packet.capture(obs, 0))
    _, depth, cam = P.views(obs, "agentview")
    wmap = P.world_map(depth, cam)
    hole = np.linalg.norm(wmap[..., :2] - np.array([tx, ty]), axis=-1) < 0.06
    pub = np.array(obs["depth"]["agentview"])
    pub[hole[:, ::-1]] = 0.0                      # published = calibration mirrored horizontally
    obs["depth"]["agentview"] = pub
    col = P.free_column(P.Packet.capture(obs, 0), d, 0.08, ee, {})
    assert col["free"] is False and col["reason"] == "column_unobserved" and col["column_points"] == 0


def test_evidence_dirs_default_override_none(tmp_path):
    from hs_witnessed_rescue.controller import evidence_dirs
    d = evidence_dirs(["hs", "eval", "m", "--out", str(tmp_path / "o")], {})
    assert d["packets"] == str((tmp_path / "o" / "witnessed_rescue_evidence" / "packets").resolve())
    assert d["planner_source"] == "argv:--out"
    d = evidence_dirs(["hs", "eval", f"--out={tmp_path}/o2"], {"HS_WR_PACKET_DIR": str(tmp_path / "x")})
    assert d["packets"] == str((tmp_path / "x").resolve()) and d["packets_source"] == "override:HS_WR_PACKET_DIR"
    assert d["planner"].endswith("o2/witnessed_rescue_evidence/planner")
    assert evidence_dirs(["pytest"], {}) == {"packets": None, "packets_source": "none", "planner": None,
                                             "planner_source": "none"}


def test_cards_are_complete_and_declare_the_planner():
    import yaml
    from hs_witnessed_rescue import motion as Mo
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hs_witnessed_rescue")
    for name in ("paper.yaml", "paper_respondoff.yaml", "paper_noplace.yaml", "paper_analytic.yaml"):
        card = yaml.safe_load(open(os.path.join(pkg, name)))
        prm = card["params"]
        assert set(prm["motion"]) == set(Mo.DEFAULTS) and set(prm["perception"]) == set(P.DEFAULTS)
        assert set(prm["triggers"]) == set(T.DEFAULTS)
        assert set(prm["planner"]) == set(PL.DEFAULTS) - {"scripted_responses"}
        assert card["budgets"]["llm_calls"] == prm["planner"]["max_calls_episode"] == 12
        assert card["budgets"]["tool_calls"] == prm["max_perception_calls"]
        (op,) = card["method_local_operators"]
        assert op["model"] == prm["planner"]["model"] == "gpt-6.1-sol" and op["backend"] == "codex"


def test_no_sibling_method_import_names_in_sources():
    import re as _re
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    text = ""
    for root, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith((".py", ".yaml", ".yml", ".json", ".toml")):
                text += open(os.path.join(root, f), errors="replace").read()
    name = "hs_" + "harness" + "_vla"          # built so this file does not contain the token itself
    assert not _re.search(rf"\b{name}\b", text)



def test_fix2_custody_prompt_describes_only_sent_images():
    t = PL.custody_prompt("put the bowl in the drawer", "bowl", ["before", "after"])
    assert "Image 1: a zoom of the fixed front scene camera BEFORE" in t and "Image 2: the SAME zoom window" in t
    assert "wrist" not in t.lower() and "Image 3" not in t
    t3 = PL.custody_prompt("put the bowl in the drawer", "bowl", ["before", "after", "wrist"])
    assert "Image 3: the wrist camera" in t3


def test_fix2_cards_declare_send_state_and_eligibility():
    import yaml
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hs_witnessed_rescue")
    for name in ("paper.yaml", "paper_respondoff.yaml", "paper_noplace.yaml", "paper_analytic.yaml"):
        card = yaml.safe_load(open(os.path.join(pkg, name)))
        assert card["benchmark_params"] == {"send_state": True}
        assert card["params"]["triggers"]["min_steps_left"] == 170 == card["params"]["min_steps_left_place"]
        assert card["params"]["custody_abstain_policy"] == "measured"
    A, S = traj([(10, OPEN, (1.0, 0.08)), (4, CLOSE, (1.0, 0.0))])
    f = T.replay(A, S, horizon=13 + 151)        # the pilot's t3i4 fire had 151 steps left -> now ineligible
    assert f and f[0]["eligible"] is False


def test_fix2_zoom_view_same_window():
    pk = P.Packet.capture(bench_obs(), 0)
    a = P.zoom_view(pk, 40, 50, {})
    assert a.shape == (320, 320, 3)
    assert np.array_equal(a, P.zoom_view(P.Packet.capture(bench_obs(), 9), 40, 50, {}))


def test_fix2_witness_prompt_carries_measured_size_only_for_located_targets():
    t = PL.witness_prompt("put the black bowl in the drawer", "black bowl", False,
                          {"height": 0.131, "extent_xy": [0.037, 0.030], "_mask": np.zeros(2)})
    assert "about 13.1 cm tall" in t and "3.7 x 3.0 cm" in t and "does not fit a black bowl" in t
    assert "cm" not in PL.witness_prompt("put the black bowl in the drawer", "black bowl", True,
                                         {"height": 0.131, "extent_xy": [0.037, 0.030]})
    assert "depth camera" not in PL.witness_prompt("x y", "bowl", False, None)



def test_fix2b_plan_prompt_says_drawers_are_never_open_top():
    for ev in ("H2_empty_close", "H1_settled_clamp"):
        t = PL.plan_prompt("put the black bowl in the bottom drawer of the cabinet and close it", ev, 512)
        assert "open_top is ALWAYS false for a drawer or any other furniture or appliance compartment" in t
        assert "microwave" in t and "caddy" in t and "even when it is open" in t
