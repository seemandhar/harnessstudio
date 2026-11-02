"""ACTIVE path through the real in-process vla-eval stack (HarnessServer + run_episode via vla-eval, seeded fake policy
server, execute_horizon 5, official 520-step horizon), playbook lesson 20261028T085458Z: tier 1 never activates PRR (the
fake contract has no 'states'), so these tests script LIBERO-like 8-d public states and a scripted policy.

Scripted benchmark: p += 0.0054 * clip(a_pos) - sag (z), R = exp(0.04 * clip(a_rot)) R (world frame), finger width 0.02
(closed command) / 0.08 (open). Scripted policy: rows depend only on the benchmark step u carried in obs['state'][0]
(so the shadow session returns the native chunk). Scenarios (task index):
  0 runaway   : rotation burst u 100-129 (+ second burst 260-289, third 360-389), closed from u 40, sag 0.0003 m/step
  1 stall     : +x motion u 150-249 then zero (route B; 50 steps earlier than in 1.0.0's test so that the 10-decision
                C4 guard admits it), closed from u 40
  2 cancel    : burst 100-124 then reverse 125-139 (confirmation fails), burst 200-239 (route A disarmed)
  3 age cap   : slow burst 0.18 units u 100-259 (no checkpoint within 65 steps)
  4 custody   : runaway with gripper toggles at u 112-116 while moving 1 unit/step in x (C2 abstain)
  5 near      : runaway with toggles at u 132-134 while the arm does not translate (allowed; latch)
  6 horizon   : runaway burst at u 400-429 (C4 abstain)
  7 quiet     : no trigger
  8-10 W1     : OSC_POSE stub plant (tests/osc_stub.py, PD to goal = current + 0.05 a / exp(0.5 a) R); rotation burst
                0.3 units u 100-129; at u 132 (after the checkpoint, before the fire) the arm is displaced by an
                unmodelled offset the reversal cannot undo: 8 = 20 cm / 20 deg, 9 = 32 cm / 25 deg, 10 = 4 cm / 10 deg
                (rotation about world z, tilt unchanged). The 1.1.0 correction must close it to W1 within 10 decisions.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from harnessstudio.contracts import load_contract
from harnessstudio.core import registry
from harnessstudio.core.arms import Arm
from harnessstudio.policy_ext.fake import FakePolicyServer
from harnessstudio.profiles import load_profile
from harnessstudio.server.harness_server import HarnessServer
from harnessstudio.verify import stack
from harnessstudio.verify.fixtures import FakeBenchmark
from vla_eval.benchmarks.base import StepResult

HORIZON = 520
N_SCEN = 11
OFFSETS = {8: ((0.20, 0.0, 0.0), 20.0), 9: ((0.25, 0.20, 0.05), 25.0), 10: ((0.04, 0.0, 0.0), 10.0)}
RECORD: dict = {}          # (label, task name) -> list of executed actions (filled by ScriptBenchmark)
LABEL = {"now": ""}


def _rot(aa):
    aa = np.asarray(aa, float); th = np.linalg.norm(aa)
    if th < 1e-12:
        return np.eye(3)
    k = aa / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _log(R):
    from hs_prr.geometry import log
    return log(R)


def script(scen: int, u: int) -> np.ndarray:
    a = np.zeros(7, np.float32)
    if scen >= 8:
        a[1] = 0.05
        a[3] = 0.3 if 100 <= u < 130 else 0.0
        a[6] = 1.0 if u >= 40 else -1.0
        return a
    a[1] = 0.3 if scen in (2, 3, 7) else (0.3 if u < 100 else 0.0)
    a[6] = 1.0 if u >= 40 else -1.0
    burst = lambda lo, hi, v=1.0: v if lo <= u < hi else 0.0
    if scen == 0:
        a[0] = 0.2 if 100 <= u < 140 else 0.0
        a[3] = burst(100, 130) + burst(260, 290) + burst(360, 390)
    elif scen == 1:
        a[0] = 0.5 if 150 <= u < 250 else 0.0
    elif scen == 2:
        a[3] = burst(100, 125) - burst(125, 140) + burst(200, 240)
    elif scen == 3:
        a[3] = burst(100, 260, 0.18)
    elif scen == 4:
        a[0] = 1.0 if 100 <= u < 140 else 0.0
        a[3] = burst(100, 130)
        if 112 <= u < 117:
            a[6] = -1.0
    elif scen == 5:
        a[0] = 1.0 if 100 <= u < 125 else 0.0
        a[3] = burst(100, 130)
        if 132 <= u < 135:
            a[6] = -1.0
    elif scen == 6:
        a[3] = burst(400, 430)
    return a


class ScriptPolicy(FakePolicyServer):
    def chunk(self, obs, seed):
        u0, scen = int(round(float(obs["state"][0]))), int(round(float(obs["state"][1])))
        return np.stack([script(scen, u0 + i) for i in range(self.chunk_len)]).astype(np.float32)


class ScriptBenchmark(FakeBenchmark):
    def reset(self, task):
        out = super().reset(task)
        self._scen = int(task["name"].rsplit("_", 1)[1])
        self._p3 = np.array([0.0, 0.0, 1.0])
        self._R = _rot([np.pi, 0.0, 0.0])
        self._w = 0.08
        self._sag = 0.0003 if self._scen == 0 else 0.0
        self._osc = None
        if self._scen >= 8:
            from osc_stub import OSCStub
            self._osc = OSCStub(self._p3, self._R)
        RECORD[(LABEL["now"], task["name"])] = []
        return out

    def step(self, action):
        a = np.asarray(action["actions"], dtype=np.float64).reshape(-1)
        RECORD[(LABEL["now"], self._task_name)].append(np.asarray(action["actions"]).copy())
        if self._osc is not None:
            if self._t == 132:
                off, deg = OFFSETS[self._scen]
                self._osc.p = self._osc.p + np.asarray(off)
                self._osc.R = _rot([0.0, 0.0, np.radians(deg)]) @ self._osc.R
            self._osc.step(a)
            self._p3, self._R = self._osc.p.copy(), self._osc.R.copy()
            self._w = 0.02 if a[6] >= 0 else 0.08
            self._t += 1
            return StepResult(obs={"pos": self._pos.copy()}, reward=0.0, done=self._t >= self.horizon, info={"success": False})
        self._p3 = self._p3 + 0.0054 * np.clip(a[:3], -1, 1) - np.array([0, 0, self._sag])
        self._R = _rot(0.04 * np.clip(a[3:6], -1, 1)) @ self._R
        self._w = 0.02 if a[6] >= 0 else 0.08
        self._t += 1
        return StepResult(obs={"pos": self._pos.copy()}, reward=0.0, done=self._t >= self.horizon, info={"success": False})

    def make_obs(self, raw_obs, task):
        obs = super().make_obs(raw_obs, task)
        self._task_name = task["name"]
        obs["state"] = np.array([float(self._t), float(self._scen)], np.float32)
        obs["states"] = np.concatenate([self._p3, _log(self._R), [self._w / 2, -self._w / 2]])
        return obs

    def get_step_result(self, step_result):
        return {"success": False}


def run_arm(method: str, arm: Arm, label: str):
    LABEL["now"] = label
    config = registry.load(method)
    prof = Path(tempfile.mkdtemp(prefix="hs-prr-prof-")) / "fake-exec5.yaml"
    prof.write_text("capabilities: [act, chunk, seeded_sampling, isolated_streams]\nchunking: harness\nexecute_horizon: 5\n")
    profile = load_profile(str(prof))
    policy = ScriptPolicy(chunking="harness", seeded=profile.seeded, chunk_len=10)
    base = load_contract("fake")
    contract = dataclasses.replace(base, public_keys=tuple(base.public_keys) + ("states",), default_horizon=HORIZON)
    with stack.serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract, arm=arm, profile=profile)
        with stack.serve(harness) as hh:
            cfg = stack.benchmark_config(hh.url, n_tasks=N_SCEN, episodes=1, horizon=HORIZON, timeout=120.0,
                                         instruction="put the pot on the stove")
            cfg["benchmarks"][0]["benchmark"] = f"{__name__}:ScriptBenchmark"
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                cfg["output_dir"] = tmp
                stack.run(cfg, docker=False, no_save=False)
            harness.wait_idle()
    recs = {int(r["task"].rsplit("_", 1)[1]): r for r in harness.records}
    acts = {int(k[1].rsplit("_", 1)[1]): np.array(v) for k, v in RECORD.items() if k[0] == label}
    return recs, acts


def events(rec, name):
    return [e for e in rec["events"] if e.get("t") == "event" and e.get("name") == name]


@pytest.fixture(scope="module")
def arms():
    out = {}
    out["base"] = run_arm("prr-v11", Arm.BASELINE, "base")
    out["null"] = run_arm("prr-v11", Arm.NULL, "null")
    out["prr"] = run_arm("prr-v11", Arm.CANDIDATE, "prr")
    out["hold"] = run_arm("prr-v11-hold", Arm.CANDIDATE, "hold")
    out["ortho"] = run_arm("prr-v11-ortho", Arm.CANDIDATE, "ortho")
    return out


def test_all_valid_full_horizon(arms):
    for name, (recs, acts) in arms.items():
        assert len(recs) == N_SCEN, name
        for k, r in recs.items():
            assert r["valid"], (name, k, r["invalid_reason"])
            assert r["accounting"]["steps"] == HORIZON and len(acts[k]) == HORIZON, (name, k)
            if name != "base":
                assert r["accounting"]["method_policy_samples"] <= 110   # _charge counts shadow samples here too
                assert r["accounting"]["llm_calls"] == 0


def test_shadow_null_bitwise_and_plans_logged(arms):
    base, _ = arms["base"]
    null, _ = arms["null"]
    prr, _ = arms["prr"]
    for k in base:
        assert null[k]["action_sha256"] == base[k]["action_sha256"], k
    # the shadow computes the same first fire plan as the candidate on the native trajectory
    for k in (0, 1, 5, 8, 9, 10):
        fn, fc = events(null[k], "prr_restore_start")[0], events(prr[k], "prr_restore_start")[0]
        assert (fn["route"], fn["fire"], fn["checkpoint"], fn["L"], fn["latch"]) == \
               (fc["route"], fc["fire"], fc["checkpoint"], fc["L"], fc["latch"]), k


def test_no_fire_scenarios_are_bitwise_native(arms):
    base, _ = arms["base"]
    for name in ("prr", "hold", "ortho"):
        recs, _ = arms[name]
        for k, why in ((2, "prr_cancel"), (3, "no_checkpoint"), (4, "custody_window"), (6, "horizon"), (7, None)):
            r = recs[k]
            assert not events(r, "prr_restore_start"), (name, k)
            assert r["action_sha256"] == base[k]["action_sha256"], (name, k)
            if why == "prr_cancel":
                assert [e["step"] for e in events(r, "prr_onset")] == [125]          # first-onset-only: no onset at 200+
                assert events(r, "prr_cancel")[0]["step"] == 140
            elif why:
                assert any(e["abstain"] == why and e["route"] == "A" for e in events(r, "prr_fire")), (name, k)


def test_runaway_active_path_exact_reversal_latch_correction_rearm_cap(arms):
    base, bacts = arms["base"]
    recs, acts = arms["prr"]
    r, a, b = recs[0], acts[0], bacts[0]
    starts = events(r, "prr_restore_start")
    assert [(e["route"], e["fire"], e["checkpoint"], e["L"], e["latch"]) for e in starts] == \
           [("A", 140, 110, 30, 1.0), ("A", 290, 260, 30, 1.0)]
    assert r["action_sha256"][:140] == base[0]["action_sha256"][:140] and r["action_sha256"][140] != base[0]["action_sha256"][140]
    expected = np.array([np.concatenate([-b[s][:6], [1.0]]) for s in range(139, 109, -1)], np.float32)
    assert np.array_equal(a[140:170], expected)
    hb = events(r, "prr_handback")
    assert len(hb) == 2
    corr = hb[0]["correction_checks"]
    n_corr = len(corr) - 1                                                  # correction decisions executed
    assert corr[0][1] >= 0.01 and 1 <= n_corr <= 10 and corr[-1][1] < 0.01   # sag left > 1 cm; closed within budget
    h = hb[0]["step"]
    assert h == 140 + 30 + 5 * n_corr + 5 and hb[0]["steps"] == 30 + 5 * n_corr + 5 and hb[0]["w1_within_2cm_10deg"]
    assert np.all(a[170:h - 5, 6] == 1.0) and np.any(a[170:h - 5, :3] != 0)                 # correction rows, latch
    assert np.all(np.max(np.abs(a[170:h - 5, :3]), axis=1) <= 1.0 + 1e-6)
    assert np.all(a[h - 5:h, :6] == 0) and np.all(a[h - 5:h, 6] == 1.0)                       # settle
    rearm = events(r, "prr_rearm")
    assert rearm[0]["step"] == h and rearm[0]["a_search_from"] == h + 30
    # third burst (360-389) after the second hand-back: the cap of 2 restorations blocks it (no events after rearm 2)
    last = max(e["step"] for e in rearm)
    assert not [e for e in r["events"] if e.get("t") == "event" and e.get("name") in ("prr_onset", "prr_fire")
                and e.get("step", e.get("fire", 0)) > last]


def test_stall_route_b_exact_reversal_then_abstain(arms):
    base, bacts = arms["base"]
    recs, acts = arms["prr"]
    r, a, b = recs[1], acts[1], bacts[1]
    st = events(r, "prr_restore_start")
    assert [(e["route"], e["fire"], e["checkpoint"], e["L"], e["latch"]) for e in st] == [("B", 310, 235, 75, 1.0)]
    assert r["action_sha256"][:310] == base[1]["action_sha256"][:310]
    expected = np.array([np.concatenate([-b[s][:6], [1.0]]) for s in range(309, 234, -1)], np.float32)
    assert np.array_equal(a[310:385], expected)
    hb = events(r, "prr_handback")[0]
    assert hb["correction_checks"][0][1] < 0.01 and hb["steps"] == 80 and hb["step"] == 390
    fires = [e for e in events(r, "prr_fire") if e["fire"] > 390]
    assert fires and fires[0]["route"] == "B" and fires[0]["fire"] == 455 and fires[0]["abstain"] == "no_checkpoint"


def test_near_toggle_restores_with_latch(arms):
    recs, acts = arms["prr"]
    _, bacts = arms["base"]
    r, a, b = recs[5], acts[5], bacts[5]
    s0 = events(r, "prr_restore_start")[0]
    f, cp = s0["fire"], s0["checkpoint"]
    assert (f, cp) == (140, 110)
    assert np.any(b[cp:f, 6] < 0)                                    # the window had open toggles ...
    assert np.all(a[f:f + (f - cp), 6] == 1.0)                       # ... the reversal never replays them (latch)


def test_hold_zero_rows_same_trigger(arms):
    base, _ = arms["base"]
    recs, acts = arms["hold"]
    r, a = recs[0], acts[0]
    s0 = events(r, "prr_restore_start")[0]
    assert (s0["route"], s0["fire"], s0["checkpoint"], s0["L"], s0["latch"]) == ("A", 140, 110, 30, 1.0)
    assert r["action_sha256"][:140] == base[0]["action_sha256"][:140]
    hb = events(r, "prr_handback")[0]
    n = hb["steps"]
    assert n == 30 + 50 + 5                                           # pose error never closes: all 10 correction decisions
    assert np.all(a[140:140 + n, :6] == 0) and np.all(a[140:140 + n, 6] == 1.0)


def test_ortho_rotated_rows_and_public_coin(arms):
    _, bacts = arms["base"]
    recs, acts = arms["ortho"]
    precs, pacts = arms["prr"]
    for k in (0, 1, 5):
        r, a = recs[k], acts[k]
        s0 = events(r, "prr_restore_start")[0]
        p0 = events(precs[k], "prr_restore_start")[0]
        assert (s0["route"], s0["fire"], s0["checkpoint"], s0["L"]) == (p0["route"], p0["fire"], p0["checkpoint"], p0["L"])
        coin = s0["coin"]
        assert coin["sha256"] == hashlib.sha256(coin["input"].encode()).hexdigest()
        assert coin["sign"] == (1 if int(coin["sha256"], 16) % 2 == 0 else -1)
        assert coin["input"].startswith("prr-ortho|put the pot on the stove|") and coin["input"].endswith(f"|{s0['fire']}")
        f, L = s0["fire"], s0["L"]
        prr_rows = pacts[k][f:f + L].astype(np.float64)
        rows = a[f:f + L].astype(np.float64)
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]) if coin["sign"] > 0 else np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
        assert np.allclose(rows[:, :3], prr_rows[:, :3] @ rot.T, atol=1e-6)
        assert np.array_equal(rows[:, 3:], prr_rows[:, 3:])
        assert np.allclose(np.linalg.norm(rows[:, :6], axis=1), np.linalg.norm(prr_rows[:, :6], axis=1), atol=1e-6)
        hb = events(r, "prr_handback")[0]
        nc = hb["steps"] - L - 5
        assert np.all(a[f + L:f + L + nc, :3] == 0)                   # correction: rotation law only, position zero


@pytest.mark.parametrize("k", [8, 9, 10])
def test_w1_closes_unmodelled_displacement_through_real_loop(arms, k):
    """C7 / W1 through the real decision loop: the reversal leaves the injected offset (reversal-end error >= 70% of it),
    the 1.1.0 correction closes it to <= 2 cm / 10 deg at hand-back within 10 decisions."""
    base, _ = arms["base"]
    recs, acts = arms["prr"]
    r = recs[k]
    st = events(r, "prr_restore_start")
    assert st and st[0]["route"] == "A" and st[0]["fire"] > 132 >= st[0]["checkpoint"]
    assert r["action_sha256"][:st[0]["step"]] == base[k]["action_sha256"][:st[0]["step"]]
    off, deg = OFFSETS[k]
    rev = events(r, "prr_reversal_end")[0]
    assert rev["pos_err_m"] >= 0.7 * np.linalg.norm(off)
    hb = events(r, "prr_handback")[0]
    n_corr = len(hb["correction_checks"]) - 1
    assert n_corr <= 10 and hb["pos_err_m"] <= 0.02 and hb["rot_err_deg"] <= 10.0 and hb["w1_within_2cm_10deg"], hb
    a = acts[k]
    f, L = st[0]["fire"], st[0]["L"]
    corr_rows = a[f + L: hb["step"] - 5]
    assert len(corr_rows) == 5 * n_corr and np.all(np.max(np.abs(corr_rows[:, :6]), axis=1) <= 1.0 + 1e-6)
    hrec = arms["hold"][0][k]
    assert not events(hrec, "prr_handback")[0]["w1_within_2cm_10deg"]         # zero rows cannot close it (control)
