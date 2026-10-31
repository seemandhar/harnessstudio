"""Regression tests for the 2026-10-31 core review (one per finding that broke a guarantee)."""
from __future__ import annotations

import logging

from harnessstudio.contracts import load_contract
from harnessstudio.core.arms import Arm
from harnessstudio.policy_ext.fake import FakePolicyServer
from harnessstudio.profiles import load_profile
from harnessstudio.server.harness_server import HarnessServer
from harnessstudio.verify import testing_methods as tm
from harnessstudio.verify import tier1
from harnessstudio.verify.stack import episodes, run_fake, serve


def _run(config, arm, profile, **server_kwargs):
    policy = FakePolicyServer(chunking="harness", seeded=load_profile(profile).seeded)
    with serve(policy) as ph:
        hs = HarnessServer(config, policy_url=ph.url, contract=load_contract("fake"), arm=arm,
                           profile=load_profile(profile), **server_kwargs)
        with serve(hs) as hh:
            res = run_fake(hh.url, n_tasks=2, episodes=3, horizon=12, timeout=server_kwargs.pop("timeout", 60.0)
                           if "timeout" in server_kwargs else 60.0)
            hs.wait_idle()
    return {(r["task"], r["episode_idx"]): r for r in hs.records}, episodes(res)


def test_refused_case_does_not_shift_later_episodes():
    clean, _ = _run(tm.pick_last_k4(), Arm.BASELINE, "fake-t1")
    logging.disable(logging.CRITICAL)
    try:
        dirty, _ = _run(tm.pick_last_k4(), Arm.BASELINE, "fake-t1", forbidden_cases={("fake", "fake_task_0", 0)})
    finally:
        logging.disable(logging.NOTSET)
    assert not dirty[("fake_task_0", 0)]["valid"]
    for key, rec in clean.items():
        if key == ("fake_task_0", 0):
            continue
        assert dirty[key]["valid"], dirty[key]["invalid_reason"]
        assert dirty[key]["obs_sha256"] == rec["obs_sha256"], key  # same observations as a clean run


def test_shadow_null_matches_baseline_with_recording_on():
    # stack.run_fake runs with vla-eval recording on; the controller makes paid T0 calls each decision
    base = tier1._by_case(tier1._run(tm.act_probe(), Arm.BASELINE, "fake-t0"))
    null = tier1._by_case(tier1._run(tm.act_probe(), Arm.NULL, "fake-t0"))
    assert all(null[k]["action_sha256"] == base[k]["action_sha256"] for k in base)


def test_timeout_reconnect_does_not_corrupt_other_episodes():
    tm._SLOW["episodes"] = 0
    policy = FakePolicyServer(chunking="harness", seeded=False)
    logging.disable(logging.CRITICAL)
    try:
        with serve(policy) as ph:
            hs = HarnessServer(tm.slow_second(), policy_url=ph.url, contract=load_contract("fake"),
                               arm=Arm.CANDIDATE, profile=load_profile("fake-t0"))
            with serve(hs) as hh:
                run_fake(hh.url, n_tasks=1, episodes=5, horizon=8, timeout=1.0)
                hs.wait_idle(timeout=20)
    finally:
        logging.disable(logging.NOTSET)
    keys = [(r["task"], r["episode_idx"]) for r in hs.records]
    assert len(keys) == len(set(keys)), "an episode was recorded twice"
    valid = {r["episode_idx"] for r in hs.records if r["valid"]}
    assert {0, 2, 3, 4} <= valid, sorted((r["episode_idx"], r["valid"], r["invalid_reason"]) for r in hs.records)


def test_seeds_do_not_reach_methods():
    recs = tier1._run(tm.extras_probe(), Arm.CANDIDATE, "fake-t1")
    seen = {k for r in recs for e in r["events"] if e.get("t") == "decision"
            for k in (e.get("diagnostics") or {}).get("selector", {}).get("extras_keys", [])}
    assert recs and "seed" not in seen, seen
