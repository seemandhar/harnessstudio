"""End-to-end tests through real vla-eval (in-process): fake benchmark -> harness server -> fake policy."""
from __future__ import annotations

import pytest

from harnessstudio.context.policy import PolicyProfile
from harnessstudio.contracts import load_contract
from harnessstudio.core.arms import Arm
from harnessstudio.core.registry import load
from harnessstudio.policy_ext.fake import FakePolicyServer
from harnessstudio.server.harness_server import HarnessServer
from harnessstudio.verify import testing_methods as tm
from harnessstudio.verify.stack import episodes, run_fake, serve

T0_SERVER = PolicyProfile(capabilities=frozenset({"act"}), chunking="server")
T0_HARNESS = PolicyProfile(capabilities=frozenset({"act", "chunk"}), chunking="harness")
T1 = PolicyProfile(capabilities=frozenset({"act", "chunk", "seeded_sampling", "isolated_streams"}),
                   chunking="harness")


def run_arm(method, arm, profile, policy_kwargs, **bench):
    policy = FakePolicyServer(**policy_kwargs)
    with serve(policy) as ph:
        harness = HarnessServer(method, policy_url=ph.url, contract=load_contract("fake"), arm=arm, profile=profile)
        with serve(harness) as hh:
            results = run_fake(hh.url, **bench)
    return episodes(results), harness.records, policy


def direct(policy_kwargs, **bench):
    policy = FakePolicyServer(**policy_kwargs)
    with serve(policy) as ph:
        results = run_fake(ph.url, **bench)
    return episodes(results), policy


def outcomes(eps):
    return [(e["task"], e["episode_id"], e["metrics"]["success"], e["steps"]) for e in eps]


def test_identity_is_transparent_with_server_chunking():
    kwargs = {"chunking": "server", "seeded": False}
    d_eps, d_pol = direct(kwargs)
    h_eps, recs, h_pol = run_arm(load("identity"), Arm.CANDIDATE, T0_SERVER, kwargs)
    assert outcomes(h_eps) == outcomes(d_eps)
    # the policy received byte-identical observations, in the same order
    assert [r["sha256"] for r in h_pol.received] == [r["sha256"] for r in d_pol.received]
    assert all(r["valid"] for r in recs) and len(recs) == 4


def test_harness_chunking_reproduces_server_chunking():
    d_eps, _ = direct({"chunking": "server", "seeded": False})
    h_eps, recs, pol = run_arm(load("identity"), Arm.BASELINE, T0_HARNESS, {"chunking": "harness", "seeded": False})
    assert outcomes(h_eps) == outcomes(d_eps)


def test_policy_gets_raw_obs_method_gets_public():
    eps, recs, pol = run_arm(tm.zero_controller(), Arm.CANDIDATE, T1, {"chunking": "harness"})
    assert all(r["valid"] for r in recs), [r["invalid_reason"] for r in recs]
    assert all("secret_goal" in r["keys"] for r in pol.received)


def test_derived_null_matches_baseline_and_pays_compute():
    kw = {"chunking": "harness"}
    b_eps, b_recs, _ = run_arm(tm.pick_last_k4(), Arm.BASELINE, T1, kw)
    n_eps, n_recs, _ = run_arm(tm.pick_last_k4(), Arm.NULL, T1, kw)
    c_eps, c_recs, _ = run_arm(tm.pick_last_k4(), Arm.CANDIDATE, T1, kw)
    key = lambda r: (r["task"], r["episode_idx"])  # noqa: E731
    b = {key(r): r for r in b_recs}
    for r in n_recs:
        assert r["action_sha256"] == b[key(r)]["action_sha256"]
        n_sample_calls = sum(1 for e in r["events"] if e["t"] == "sample")
        assert r["accounting"]["method_policy_samples"] == 4 * n_sample_calls > 0
    assert any(r["action_sha256"] != b[key(r)]["action_sha256"] for r in c_recs)


def test_null_ignores_executor_cuts():
    kw = {"chunking": "harness", "seeded": False}
    b_eps, b_recs, _ = run_arm(tm.cut_after_2(), Arm.BASELINE, T0_HARNESS, kw)
    n_eps, n_recs, _ = run_arm(tm.cut_after_2(), Arm.NULL, T0_HARNESS, kw)
    c_eps, c_recs, _ = run_arm(tm.cut_after_2(), Arm.CANDIDATE, T0_HARNESS, kw)
    assert outcomes(n_eps) == outcomes(b_eps)
    assert max(r["accounting"]["decisions"] for r in c_recs) > max(r["accounting"]["decisions"] for r in n_recs)


def test_budget_exhaustion_finishes_natively():
    eps, recs, _ = run_arm(tm.budget_3(), Arm.CANDIDATE, T1, {"chunking": "harness"})
    assert all(r["valid"] for r in recs)
    assert all(r["accounting"]["termination"] == "budget_exhausted" for r in recs)
    assert all(r["accounting"]["method_policy_samples"] == 3 for r in recs)


def test_shadow_null_executes_native_and_pays():
    kw = {"chunking": "harness"}
    b_eps, b_recs, _ = run_arm(tm.zero_controller(), Arm.BASELINE, T1, kw)
    n_eps, n_recs, _ = run_arm(tm.zero_controller(), Arm.NULL, T1, kw)
    b = {(r["task"], r["episode_idx"]): r for r in b_recs}
    for r in n_recs:
        assert r["action_sha256"] == b[(r["task"], r["episode_idx"])]["action_sha256"]
        assert r["accounting"]["shadow_policy_samples"] >= 2


def test_method_error_marks_episode_invalid_and_run_continues():
    eps, recs, _ = run_arm(tm.crash_controller(), Arm.CANDIDATE, T0_HARNESS, {"chunking": "harness", "seeded": False})
    assert len(recs) == 4
    assert not any(r["valid"] for r in recs)
    assert all("deliberate method failure" in r["invalid_reason"] for r in recs)


def test_capability_mismatch_is_refused():
    from harnessstudio.core.errors import RegistrationError
    with pytest.raises(RegistrationError):
        HarnessServer(tm.pick_last_k4(), policy_url="ws://unused", contract=load_contract("fake"),
                      arm=Arm.CANDIDATE, profile=T0_HARNESS)


def test_llm_controller_tier1_and_shadow_pays_llm():
    from harnessstudio.verify import tier1
    report = tier1.run(tm.llm_planner())
    assert report.passed, report.lines()
    recs = tier1._run(tm.llm_planner(), Arm.NULL, "fake-t0")
    assert all(r["accounting"]["llm_calls"] == 1 for r in recs)
