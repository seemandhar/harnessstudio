"""Tests for hs_bcp (tier-0 evidence required by review r8 K1-c1, and fake-stack code-path evidence for K1-c2).

CPU only. Run: harnessstudio/.venv/bin/python -m pytest harnessstudio/methods/bcp/tests -q
The fake stack ignores what a prompt means (lesson 20261028T090630Z): these tests prove code paths, stream
isolation and action identity, never that a clause prompt changes real pi0-FAST actions (K1 measures that).
"""
from __future__ import annotations

import base64
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from hs_bcp import rules  # noqa: E402
from hs_bcp import ENTRIES, config_bcp, config_bcp_switch, config_bcp_switch_dot  # noqa: E402

# Public libero_10 task language in suite order (as boundary_exposure.py INSTR; vla-eval sends these strings).
INSTR = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    1: "put both the cream cheese box and the butter in the basket",
    2: "turn on the stove and put the moka pot on it",
    3: "put the black bowl in the bottom drawer of the cabinet and close it",
    4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    5: "pick up the book and place it in the back compartment of the caddy",
    6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    7: "put both the alphabet soup and the cream cheese box in the basket",
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
}
# Expected clauses = boundary_exposure.json summary.clauses (scientist's frozen output, sha256 b3b8d286...).
EXPECTED = {0: None, 1: None, 2: "put the moka pot on the stove", 3: "close the bottom drawer of the cabinet",
            4: "put the yellow and white mug on the right plate", 5: None,
            6: "put the chocolate pudding to the right of the plate", 7: None, 8: None, 9: "close the microwave"}


# ---------------------------------------------------------------- K1-c1 (a): split / pronoun rule on all 10 strings
def test_split_rule_on_all_libero10_strings():
    got = {t: rules.split_clause(s) for t, s in INSTR.items()}
    assert got == EXPECTED
    assert [t for t, c in got.items() if c is None] == [0, 1, 5, 7, 8]


def test_split_rule_matches_scientist_output_file():
    import json
    p = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtcfpp-20261101/ideation/scripts-r8-task/"
             "boundary_exposure.json")
    if not p.exists():
        pytest.skip("scientist output not present")
    clauses = json.loads(p.read_text())["summary"]["clauses"]
    assert {int(k): v for k, v in clauses.items()} == EXPECTED


def test_control_prompt_is_original_plus_space_dot():
    for s in INSTR.values():
        c = rules.control_prompt(s)
        assert c == s + " ." and c != s and c.lower().strip().replace("_", " ") != s  # survives openpi's cleaning


# ---------------------------------------------------------------- K1-c1 (b): custody tracker on scripted streams
def _state(width):
    return np.array([0, 0, 0, 0, 0, 0, width / 2, -width / 2], dtype=np.float32)


def _feed(tracker, grips, widths):
    fired = []
    for t, (g, w) in enumerate(zip(grips, widths)):
        a = np.zeros(7)
        a[6] = g
        if tracker.update(a, _state(w)):
            fired.append(t)
    return fired


def test_custody_fires_on_held_segment_then_open():
    grips = [-1] * 20 + [1] * 30 + [-1] * 10
    widths = [0.08] * 20 + [0.03] * 10 + [0.02] * 20 + [0.08] * 10
    tr = rules.CustodyTracker()
    assert _feed(tr, grips, widths) == [50] and tr.boundary == 50
    assert tr.segments[-1] == {"start": 20, "end": 50, "len": 30, "w_settled": 0.02}


def test_custody_ignores_empty_close_short_hold_and_unfinished_segment():
    # empty close (width 0.002 < 0.003), then a 14-step held segment, then a held segment never released
    grips = [1] * 20 + [-1] * 5 + [1] * 14 + [-1] * 5 + [1] * 40
    widths = [0.002] * 20 + [0.08] * 5 + [0.02] * 14 + [0.08] * 5 + [0.02] * 40
    tr = rules.CustodyTracker()
    assert _feed(tr, grips, widths) == [] and tr.boundary is None
    assert [s["len"] for s in tr.segments] == [20, 14]


def test_custody_settle_window_and_first_segment_only():
    # 15-step segment: settled rows = [a+10, b) = last 5 rows; first rows wide (closing), last rows 0.004
    grips = [1] * 15 + [-1] * 3 + [1] * 20 + [-1] * 2
    widths = [0.05] * 10 + [0.004] * 5 + [0.08] * 3 + [0.03] * 20 + [0.08] * 2
    tr = rules.CustodyTracker()
    assert _feed(tr, grips, widths) == [15]
    assert tr.segments == [{"start": 0, "end": 15, "len": 15, "w_settled": 0.004}]


def test_custody_matches_offline_features_rule_on_random_streams():
    """Online tracker == offline analysis/r8/features.py segments + boundary_exposure.py boundary()."""
    rng = np.random.default_rng(20261101)

    def offline(g, width):
        n = len(g)
        segs, cur = [], (0 if g[0] > 0 else None)
        for t in range(1, n):
            if g[t] > 0 and cur is None:
                cur = t
            if g[t] <= 0 and cur is not None:
                segs.append((cur, t))
                cur = None
        for a, b in segs:  # closed segments only (the online tracker cannot fire on an unreleased one)
            wm = float(np.median(width[min(a + 10, b - 1):b]))
            if b - a >= 15 and round(wm, 4) >= 0.003:
                return b
        return None

    for _ in range(300):
        n = 200
        runs = rng.integers(1, 40, size=40)
        g = np.concatenate([np.full(r, 1.0 if i % 2 else -1.0) for i, r in enumerate(runs)])[:n]
        if rng.random() < 0.5:
            g = -g
        width = rng.choice([0.001, 0.0025, 0.0029, 0.00296, 0.0031, 0.004, 0.02, 0.08], size=n)
        tr = rules.CustodyTracker()
        _feed(tr, g, width)
        assert tr.boundary == offline(g, width)   # incl. the 4-decimal rounding (0.00296 -> 0.003 fires)


# ---------------------------------------------------------------- measures
def test_chunk_compare_measures():
    n = np.zeros((10, 7), np.float32)
    n[:, 0] = 0.1
    n[:, 6] = 1.0
    same = n.copy()
    m = rules.chunk_compare(n, same)
    assert m["token_identical"] and m["cosine"] == pytest.approx(1.0) and not m["disagree"]
    opp = n.copy()
    opp[:, 0] = -0.1
    m = rules.chunk_compare(n, opp)
    assert not m["token_identical"] and m["cosine"] == pytest.approx(-1.0) and m["disagree"]
    flip = n.copy()
    flip[3, 6] = -1.0
    m = rules.chunk_compare(n, flip)
    assert m["gripper_flip_rows"] == [3] and m["disagree"] and m["cosine"] == pytest.approx(1.0)
    late = n.copy()
    late[7, 6] = -1.0      # row 7 is outside rows 0-4
    m = rules.chunk_compare(n, late)
    assert not m["gripper_flip"] and not m["disagree"] and not m["token_identical"]


# ---------------------------------------------------------------- registration
def test_cards_build():
    for fn, mode in ((config_bcp, "shadow_probe"), (config_bcp_switch, "switch"),
                     (config_bcp_switch_dot, "switch_control")):
        cfg = fn()
        assert cfg.null == "shadow" and cfg.budgets.llm_calls == 0 and cfg.controller is not None
        assert cfg.controller.params["mode"] == mode
        c = cfg.controller.build()
        assert c.mode == mode and c.window_decisions == 10 and c.exec_rows == 5
        assert c.tracker_args == {"hold_steps": 15, "width_min": 0.003, "settle_rows": 10, "gripper_dim": 6,
                                  "width_dims": (6, 7)}
    assert ENTRIES == ("bcp", "bcp-switch", "bcp-switch-dot")


# ---------------------------------------------------------------- fake stack (code path, K1-c2 stream isolation)
FAKE_PARAMS = {"state_key": "state", "width_dims": [0, 1], "hold_steps": 1, "width_min": -1e9, "settle_rows": 0,
               "window_decisions": 3, "log_rows": True}


def _fake_config(entry: str):
    from harnessstudio.core.card import config_from_card, load_card
    from harnessstudio.core.method import ModuleRef
    src = PKG_ROOT / "hs_bcp" / f"paper_{entry}.yaml"
    card = copy.deepcopy(load_card(src))
    card["params"].update(FAKE_PARAMS)
    card["tier1"] = {"instruction": "turn on the stove and put the moka pot on it", "horizon": 40, "chunk_len": 5}
    tmp = Path(tempfile.mkdtemp(prefix="hs-bcp-test-")) / f"paper_{entry}.yaml"
    tmp.write_text(yaml.safe_dump(card))
    return config_from_card(tmp, controller=ModuleRef("hs_bcp.controller:BCPController", dict(card["params"])))


def _runs(entry):
    from harnessstudio.core.arms import Arm
    from harnessstudio.verify import tier1
    cfg = _fake_config(entry)
    return {arm: tier1._by_case(tier1._run(cfg, arm, "fake-t1")) for arm in Arm}


def _events(rec, name):
    return [e for e in rec["events"] if e.get("t") == "event" and e.get("name") == name]


def _native(rec):
    return [(e["decision"], e["step"], e["action_sha256"]) for e in rec["events"] if e.get("t") == "native"]


def _arr(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"])


@pytest.fixture(scope="module")
def probe_runs():
    return _runs("bcp")


@pytest.fixture(scope="module")
def switch_runs():
    return _runs("bcp-switch")


def test_k1_null_arm_is_bitwise_baseline_and_native_stream_untouched(probe_runs):
    from harnessstudio.core.arms import Arm
    base, null = probe_runs[Arm.BASELINE], probe_runs[Arm.NULL]
    assert len(base) == len(null) == 4
    windows = 0
    for k, b in base.items():
        n = null[k]
        assert n["valid"] and b["valid"]
        assert n["action_sha256"] == b["action_sha256"]          # executed actions identical (E2)
        assert _native(n) == _native(b)                          # same native decisions, steps, chunks (seeds)
        assert n["accounting"]["native_policy_samples"] == b["accounting"]["native_policy_samples"]
        assert not [e for e in n["events"] if e.get("t") in ("sample", "act")]  # no call on the native session
        w = [e for e in _events(n, "bcp_decision") if "window" in e]
        windows += len(w)
        assert 1 <= len(w) <= 3 and [e["window"] for e in w] == list(range(1, len(w) + 1))
        assert all(e["step"] > e["boundary"] for e in w)
        assert n["accounting"]["shadow_policy_samples"] == len(_events(n, "bcp_decision")) + 2 * len(w)
        for e in w:   # logged chunks reproduce the logged fingerprints and measures
            assert rules.array_fingerprint(_arr(e["native"])) == e["chunk_sha"]
            assert rules.array_fingerprint(_arr(e["clause"])) == e["clause_sha"]
            assert e["clause_vs_native"] == rules.chunk_compare(_arr(e["native"]), _arr(e["clause"]))
        assert bool(_events(n, "bcp_window_done")) == (len(w) == 3)   # else the horizon cut the window
    assert windows >= 8


def test_k1_candidate_arm_would_shift_native_seeds(probe_runs):
    """Why K1 runs only as the null arm: in the candidate arm the two extra calls advance the decision counter."""
    from harnessstudio.core.arms import Arm
    cand = probe_runs[Arm.CANDIDATE]
    shifted = 0
    for rec in cand.values():
        samples = [e for e in rec["events"] if e.get("t") == "sample"]
        ndec = len(_events(rec, "bcp_decision"))
        nwin = len([e for e in _events(rec, "bcp_decision") if "window" in e])
        assert len(samples) == ndec + 2 * nwin
        if nwin:   # an episode the fake env ends before the window has nothing to shift
            shifted += 1
            assert samples[-1]["decision"] == len(samples) - 1 > ndec - 1
    assert shifted >= 2


def test_executed_gripper_rows_are_visible_to_the_controller(switch_runs, probe_runs):
    """K1-c1: the rows the controller passes to ctx.execute are the executed actions (candidate arm), so the
    custody tracker reads executed gripper commands; per-step action fingerprints match the trace."""
    from harnessstudio.core.arms import Arm
    for runs in (switch_runs, probe_runs):
        for rec in runs[Arm.CANDIDATE].values():
            rows = []
            for e in _events(rec, "bcp_decision"):
                c = _arr(e["chunk"] if "chunk" in e else e["native"])
                rows.extend(rules.array_fingerprint(np.array(c[i])) for i in range(min(5, len(c))))
            m = min(len(rows), len(rec["action_sha256"]))   # after a K1 return the episode finishes natively
            assert m >= 5 and rows[:m] == rec["action_sha256"][:m]


def test_switch_candidate_native_until_boundary_then_clause_with_aligned_seeds(switch_runs):
    from harnessstudio.core.arms import Arm
    base, cand, null = switch_runs[Arm.BASELINE], switch_runs[Arm.CANDIDATE], switch_runs[Arm.NULL]
    switched = 0
    for k, b in base.items():
        c = cand[k]
        bnd = _events(c, "bcp_boundary")
        assert len(bnd) == 1
        decs = _events(c, "bcp_decision")
        first_switch = next((e for e in decs if e["role"] == "switch"), None)
        if first_switch is None:     # the fake env ended the episode right after the boundary
            assert c["action_sha256"] == b["action_sha256"]
            continue
        switched += 1
        assert first_switch["step"] > bnd[0]["step"]
        s = first_switch["step"]
        assert c["action_sha256"][:s] == b["action_sha256"][:s]        # bitwise native before the switch
        samples = [e for e in c["events"] if e.get("t") == "sample"]
        nat = _native(b)
        assert [x["decision"] for x in samples] == list(range(len(samples)))  # one call per decision
        for smp, (dn, step, _sha) in zip(samples, nat):
            assert smp["decision"] == dn and smp["step"] == step       # same decision index and step => same seed
        assert null[k]["action_sha256"] == b["action_sha256"]
        assert all(e["role"] == "native" for e in decs if e["step"] <= bnd[0]["step"])
        assert c["action_sha256"] != b["action_sha256"]   # fake policy hashes the instruction: the switch acts
    assert switched >= 2


def test_no_split_is_fully_native():
    import asyncio

    class Ctx:
        instruction = INSTR[8]
        ended = False

        def __init__(self):
            self.ev = []

        def record(self, name, **f):
            self.ev.append((name, f))

    for entry in ENTRIES:
        cfg = {"bcp": config_bcp, "bcp-switch": config_bcp_switch, "bcp-switch-dot": config_bcp_switch_dot}[entry]()
        ctx = Ctx()
        asyncio.run(cfg.controller.build().run(ctx))
        assert [n for n, _ in ctx.ev] == ["bcp_start", "bcp_native"]


def test_verb_list_frozen_and_each_verb_splits():
    assert rules.VERBS == ("put", "close", "open", "turn", "pick", "place", "push")
    for v in rules.VERBS:
        assert rules.split_clause(f"grasp the cup and {v} the red block") == f"{v} the red block"
    assert rules.split_clause("grasp the cup and lift the red block") is None
    assert rules.split_clause("pick up the book and place it in the caddy") is None      # object pronoun refused
    assert rules.split_clause("put the cup in the box and close it") == "close the box"   # 'it' -> last in/on object
