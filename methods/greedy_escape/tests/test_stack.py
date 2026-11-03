"""greedy-escape through the real handler seam: vla-eval (in-process) -> HarnessServer -> spy fake policy server.

The spy records every seeded request as it arrives at the policy server (session, seeds, sampler, observation hash),
so the tests check what greedy-escape actually sends, in every arm:
- no fire: the candidate's requests and executed actions equal the baseline's bitwise (same seeds, sampler None,
  same observations), so untriggered episodes are native greedy (GE3, GE5);
- forced stalls: the greedy decode returns an identical near-zero action; the fixed-point fire after decision 14
  opens a window: decisions 15..54 carry exactly the frozen escape sampler (== the SCALE method's sampler_ref) with
  seed crn_seed(ns, benchmark, task, episode, d, 0), the seed the SCALE slot method sends at the same decision d;
  then greedy again until the next fire (GE1-GE3);
- the shadow null executes the baseline's actions bitwise while paying the controller's window calls on its own
  session (GE6), and candidate and null records carry accounting.extra['greedy_escape'] (baseline is native);
- budget: one policy sample per decision.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from harnessstudio.context.trace import fingerprint
from harnessstudio.contracts import load_contract
from harnessstudio.core import registry
from harnessstudio.core.arms import Arm
from harnessstudio.core.noise import crn_seed
from harnessstudio.policy_ext.fake import FakePolicyServer
from harnessstudio.profiles import load_profile
from harnessstudio.server.harness_server import HarnessServer
from harnessstudio.verify.stack import run_fake, serve

STALL = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)


class SpyPolicy(FakePolicyServer):
    """Fake seeded policy (one action per request). stall=True: greedy (sampler None) returns STALL."""

    def __init__(self, stall: bool = False) -> None:
        super().__init__(chunking="harness", seeded=True, chunk_len=1)
        self.stall = stall
        self.log: list[dict] = []

    def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
        self.log.append({"session": ctx.session_id, "seeds": list(seeds), "sampler": sampler,
                         "obs": fingerprint(obs)})
        if self.stall and not sampler:
            return [(STALL.copy(), {"seed": int(s)}) for s in seeds]
        out = super().sample_chunks(obs, seeds, sampler, readouts, ctx)
        if self.stall:   # window actions: small random moves, open gripper (never a fixed point)
            out = [(np.concatenate([0.2 * c[:, :6], -np.ones((1, 1), np.float32)], axis=1), e) for c, e in out]
        return out


def run_arm(method: str, arm: Arm, stall: bool, horizon: int = 80, episodes: int = 1):
    config = registry.load(method)
    policy = SpyPolicy(stall=stall)
    contract = dataclasses.replace(load_contract("fake"), default_horizon=horizon)
    with serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract, arm=arm,
                                profile=load_profile("fake-t2"), scripted_llm=True)
        with serve(harness) as hh:
            run_fake(hh.url, n_tasks=1, episodes=episodes, horizon=horizon, instruction="do {name}")
            harness.wait_idle()
    return harness.records, policy.log


def by_session(log):
    out: dict = {}
    for e in log:
        out.setdefault(e["session"], []).append(e)
    return out


def test_card_escape_equals_the_scale_methods_sampler():
    ge = registry.load("greedy-escape")
    scale = registry.load("scale")
    assert ge.controller.params["escape"] == scale.slots["sampler"].to_json()
    assert ge.null == "shadow" and ge.budgets.policy_calls == 520
    assert ge.controller.params["fixed_point_steps"] == 15 and ge.controller.params["fixed_point_eps"] == 0.01
    assert ge.controller.params["empty_closes"] == 3 and ge.controller.params["window_decisions"] == 40
    assert ge.controller.params["max_windows"] == 4 and ge.controller.params["empty_width_m"] == 0.004


def test_no_fire_is_bitwise_baseline_at_the_request_level():
    base, blog = run_arm("greedy-escape", Arm.BASELINE, stall=False, horizon=30, episodes=2)
    cand, clog = run_arm("greedy-escape", Arm.CANDIDATE, stall=False, horizon=30, episodes=2)
    assert all(r["valid"] for r in base + cand)
    key = lambda r: (r["task"], r["episode_idx"])  # noqa: E731
    b = {key(r): r for r in base}
    for r in cand:
        assert r["action_sha256"] == b[key(r)]["action_sha256"]
        assert r["accounting"]["extra"]["greedy_escape"]["windows"] == 0
        assert r["accounting"]["method_policy_samples"] == r["accounting"]["steps"]
    strip = lambda log: [(e["seeds"], e["sampler"], e["obs"]) for e in log]  # noqa: E731
    assert sorted(map(str, strip(clog))) == sorted(map(str, strip(blog)))
    assert all(e["sampler"] is None for e in clog)


def test_stall_fires_window_with_escape_sampler_and_scale_seeds():
    cand, clog = run_arm("greedy-escape", Arm.CANDIDATE, stall=True, horizon=80)
    (rec,) = cand
    assert rec["valid"], rec["invalid_reason"]
    escape = registry.load("greedy-escape").controller.params["escape"]
    reqs = next(iter(by_session(clog).values()))
    steps = rec["accounting"]["steps"]
    assert len(reqs) == steps == rec["accounting"]["method_policy_samples"]
    summ = rec["accounting"]["extra"]["greedy_escape"]
    starts = summ["window_starts"]
    assert starts[0] == 15
    in_window = set()
    for s in starts:
        in_window |= set(range(s, min(s + 40, steps)))
    for d, e in enumerate(reqs):
        assert e["sampler"] == (escape if d in in_window else None), d
        assert e["seeds"] == [crn_seed("hs", rec["benchmark"], rec["task"], rec["episode_idx"], d, 0)]
    fires = [ev for ev in rec["events"] if ev["t"] == "event" and ev.get("name") == "greedy_escape"
             and ev.get("phase") == "fire"]
    assert fires[0]["fire_step"] == 14 and fires[0]["reasons"] == ["fixed_point"] and fires[0]["window"] is True
    assert fires[0]["step"] == 15            # the framework's own step field (actions executed) is kept
    ends = [ev for ev in rec["events"] if ev["t"] == "event" and ev.get("phase") == "window_end"]
    assert ends[0]["last_action"] == 54 and ends[0]["windows"] == 1
    # after the 40-decision window the stalled greedy decode needs 15 new identical actions: next fire after 69
    if steps > 70:
        assert starts[1] == 70
    # the SCALE slot method sends the same (seed, sampler) at the same decision
    scale, slog = run_arm("scale", Arm.CANDIDATE, stall=True, horizon=80)
    sreqs = next(iter(by_session(slog).values()))
    for d in sorted(in_window):
        if d < len(sreqs):
            assert (reqs[d]["seeds"], reqs[d]["sampler"]) == (sreqs[d]["seeds"], sreqs[d]["sampler"])
    # (observations differ: SCALE decodes from step 0 and leaves the stall; only the request contract is compared)


def test_shadow_null_executes_baseline_actions_and_pays_window_calls():
    base, _ = run_arm("greedy-escape", Arm.BASELINE, stall=True, horizon=80)
    null, nlog = run_arm("greedy-escape", Arm.NULL, stall=True, horizon=80)
    (b,), (n,) = base, null
    assert n["valid"], n["invalid_reason"]
    assert n["action_sha256"] == b["action_sha256"]
    escape = registry.load("greedy-escape").controller.params["escape"]
    sessions = by_session(nlog)
    assert len(sessions) == 2                              # native session + shadow session
    shadow = [s for s in sessions.values() if any(e["sampler"] for e in s)]
    native = [s for s in sessions.values() if not any(e["sampler"] for e in s)]
    assert len(shadow) == 1 and len(native) == 1
    summ = n["accounting"]["extra"]["greedy_escape"]
    # windows fire on the (stalled) native trajectory: 15, 70 within 80 steps
    assert summ["window_starts"][:2] == [15, 70]
    sh = shadow[0]
    assert [d for d, e in enumerate(sh) if e["sampler"] == escape][:3] == [15, 16, 17]
    acc = n["accounting"]
    assert acc["native_policy_samples"] == acc["steps"] and acc["shadow_policy_samples"] == acc["steps"]


def test_budget_is_one_sample_per_decision_at_libero_10_horizon():
    cand, _ = run_arm("greedy-escape", Arm.CANDIDATE, stall=True, horizon=520)
    (rec,) = cand
    assert rec["valid"], rec["invalid_reason"]
    assert rec["accounting"]["method_policy_samples"] == rec["accounting"]["steps"] <= 520
    assert rec["accounting"]["termination"] != "budget_exhausted"
    assert rec["accounting"]["extra"]["greedy_escape"]["windows"] <= 4
