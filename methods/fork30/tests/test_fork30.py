"""fork30: cue unit tests, identity guard, F3 parity on all 100 greedy map records, and the real handler seam
(vla-eval in-process -> HarnessServer -> spy policy): unfired episodes bitwise greedy at the request level; a forced
repeat prefix forks exactly at decision 30 to the frozen SCALE ref with the CRN seed of each decision; shadow null ==
baseline."""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import fire_set  # noqa: E402

from harnessstudio.context.trace import fingerprint  # noqa: E402
from harnessstudio.contracts import load_contract  # noqa: E402
from harnessstudio.core import registry  # noqa: E402
from harnessstudio.core.arms import Arm  # noqa: E402
from harnessstudio.core.noise import crn_seed  # noqa: E402
from harnessstudio.policy_ext.fake import FakePolicyServer  # noqa: E402
from harnessstudio.profiles import load_profile  # noqa: E402
from harnessstudio.server.harness_server import HarnessServer  # noqa: E402
from harnessstudio.verify.stack import run_fake, serve  # noqa: E402
from hs_fork30.controller import RepeatCue, scale_ref  # noqa: E402

EXPECTED = ["t0i1", "t0i4", "t2i2", "t2i7", "t2i8", "t3i0", "t3i4", "t3i5", "t3i6", "t4i3", "t4i4", "t4i5", "t4i9",
            "t6i0", "t6i6", "t7i2", "t7i4", "t8i1", "t8i4", "t8i5", "t9i1", "t9i7",
            "t0i6", "t2i1", "t2i3", "t2i5", "t2i9", "t5i5", "t7i5", "t8i0", "t9i3"]


def a(x, g=-1.0):
    return np.array([x, 0, 0, 0, 0, 0, g], dtype=np.float32)


# ---- cue --------------------------------------------------------------------------------------------------------------
def test_cue_counts_exact_repeats_in_steps_1_to_29_and_decides_once():
    for reps, expect in ((6, True), (5, False)):
        cue = RepeatCue()
        acts = [a(0.1)] * (reps + 1) + [a(0.2 + 0.001 * k) for k in range(29 - reps)]
        assert len(acts) == 30
        out = [cue.step(x) for x in acts]
        assert out[:29] == [None] * 29 and out[29] is expect and cue.repeats == reps
        assert cue.step(a(0.2 + 0.001 * 28)) is None          # decided once; later repeats are not counted
    cue = RepeatCue()
    for k in range(29):
        cue.step(a(0.1 + 0.001 * k))
    cue.step(a(0.1 + 0.001 * 28))                               # step 29 repeats step 28: counted (k = 29)
    assert cue.repeats == 1 and cue.decided is False


def test_cue_uses_executed_gripper_and_non_adjacent_repeats():
    cue = RepeatCue()
    acts = [a(0.1, g=0.3), a(0.1, g=0.0)] + [a(0.1 * (k % 2) + 0.5) for k in range(28)]
    for x in acts:
        cue.step(x)
    assert cue.repeats == 1                                      # 0.3 / 0.0 both execute +1; alternating rows never repeat


def test_identity_guard():
    with pytest.raises(ValueError):
        scale_ref({"factory": "hs_scale.decoder:ScaleDecoder", "params": {"T0": 0.9}})
    with pytest.raises(ValueError):
        scale_ref({"factory": "hs_other.decoder:X", "params": {}})
    ref = registry.load("fork30").controller.params["sampler"]
    assert ref == registry.load("scale").slots["sampler"].to_json()
    assert registry.load("fork30").card["benchmark_params"] == {"send_state": False}


# ---- F3 parity --------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.isdir(str(fire_set.SHARDS)), reason="development map not on this machine")
def test_fires_exactly_on_the_31_expected_scenes():
    t = fire_set.fire_table()
    key = lambda s: (int(s[1]), int(s[3:]))  # noqa: E731
    assert t["fire_set"] == sorted(EXPECTED, key=key)
    for s, v in t["scenes"].items():
        if v["fired"]:
            assert v["fork_at"] == 30 and v["repeats"] >= 6 and v["logged"]["at"] == 30
        else:
            assert v["fork_at"] is None and (v["steps"] < 30 or v["repeats"] < 6)


# ---- seam -------------------------------------------------------------------------------------------------------------
class Spy(FakePolicyServer):
    def __init__(self, repeat_prefix: int = 0):
        super().__init__(chunking="harness", seeded=True, chunk_len=1)
        self.repeat_prefix, self.log, self.n = repeat_prefix, [], {}

    def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
        n = self.n[ctx.session_id] = self.n.get(ctx.session_id, -1) + 1
        self.log.append({"session": ctx.session_id, "seeds": list(seeds), "sampler": sampler, "obs": fingerprint(obs)})
        if self.repeat_prefix and not sampler and n <= self.repeat_prefix:
            return [(np.array([[0.3, 0, 0, 0, 0, 0, -1]], np.float32), {"seed": int(s)}) for s in seeds]
        return super().sample_chunks(obs, seeds, sampler, readouts, ctx)


def run_arm(arm, repeat_prefix=0, horizon=60, episodes=1):
    config = registry.load("fork30")
    policy = Spy(repeat_prefix)
    contract = dataclasses.replace(load_contract("fake"), default_horizon=horizon)
    with serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract, arm=arm,
                                profile=load_profile("fake-t2"), scripted_llm=True)
        with serve(harness) as hh:
            run_fake(hh.url, n_tasks=1, episodes=episodes, horizon=horizon, instruction="do {name}")
            harness.wait_idle()
    for r in harness.records:
        assert r["valid"], r["invalid_reason"]
    return harness.records, policy.log


def test_unfired_is_bitwise_greedy_at_request_level():
    base, blog = run_arm(Arm.BASELINE, episodes=2)
    cand, clog = run_arm(Arm.CANDIDATE, episodes=2)
    key = lambda r: r["episode_idx"]  # noqa: E731
    b = {key(r): r for r in base}
    for r in cand:
        assert r["action_sha256"] == b[key(r)]["action_sha256"]
        assert r["accounting"]["extra"]["fork30"]["fired"] is False
    strip = lambda log: sorted(str((e["seeds"], e["sampler"], e["obs"])) for e in log)  # noqa: E731
    assert strip(clog) == strip(blog) and all(e["sampler"] is None for e in clog)


def test_repeat_prefix_forks_at_decision_30_with_scale_and_crn_seeds():
    (rec,), log = run_arm(Arm.CANDIDATE, repeat_prefix=10)
    ref = registry.load("fork30").controller.params["sampler"]
    assert [e["sampler"] for e in log] == [None] * 30 + [ref] * (len(log) - 30) and len(log) == 60
    assert [e["seeds"] for e in log] == [[crn_seed("hs", rec["benchmark"], rec["task"], 0, d, 0)] for d in range(60)]
    ev = [e for e in rec["events"] if e["t"] == "event" and e.get("name") == "fork"]
    assert ev and ev[0]["fired"] is True and ev[0]["repeats"] == 10 and ev[0]["at"] == 30
    s = rec["accounting"]["extra"]["fork30"]
    assert s["fired"] and s["scale_decisions"] == 30 and rec["accounting"]["method_policy_samples"] == 60
    (r5,), log5 = run_arm(Arm.CANDIDATE, repeat_prefix=5)          # 5 repeats -> no fork
    assert all(e["sampler"] is None for e in log5) and r5["accounting"]["extra"]["fork30"]["repeats"] == 5


def test_shadow_null_is_baseline_and_pays_the_scale_calls():
    (base,), _ = run_arm(Arm.BASELINE, repeat_prefix=10)
    (null,), nlog = run_arm(Arm.NULL, repeat_prefix=10)
    assert null["action_sha256"] == base["action_sha256"]
    ref = registry.load("fork30").controller.params["sampler"]
    assert sum(1 for e in nlog if e["sampler"] == ref) == 30           # shadow session pays SCALE from decision 30
    assert null["accounting"]["native_policy_samples"] == null["accounting"]["steps"] == 60
