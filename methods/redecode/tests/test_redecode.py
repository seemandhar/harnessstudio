"""ReDecode unit tests (CPU): C-R1 retry zero test against the sent state, C-R2 delta and token counting, C-R3 re-base,
C-R9 reuse of the frozen v1.2 op (bitwise vs v1.2 z / bz when every retry fails, vs identity / v1.2 b without zero
chunks), call accounting, override contents, cards and budgets."""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
import pytest

from harnessstudio.core import registry
from hs_redecode.controller import ReDecodeController, state_tokens
from hs_tilt_guard_v12.composite import TiltGuardV12ActionOp

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = json.load(open(os.path.join(PKG, "hs_redecode/assets/pi0_fast_libero_norm_stats.json")))["norm_stats"]
AMEAN = np.asarray(NS["actions"]["mean"])
SMEAN, SSTD = np.asarray(NS["state"]["mean"]), np.asarray(NS["state"]["std"])
SHA = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"


class _Proposal:
    def __init__(self, a):
        self.actions = np.asarray(a)
        self.extras = {}


class FakeCtx:
    """Open-loop scripted context: decision j observes states[j]; the native call returns natives[j]; a retry call
    returns retry_fn(j, k, s_sent) (k = 1, 2)."""

    def __init__(self, states, natives, retry_fn=None):
        self.states, self.natives, self.retry_fn = states, natives, retry_fn
        self.j = 0
        self.k = 0
        self.calls = []
        self.executed = []
        self.records = []
        self.memory = {}
        outer = self

        class P:
            async def sample(self, k=1, obs=None, instruction=None, sampler=None, readouts=()):
                assert k == 1 and instruction is None
                if obs is None:
                    outer.calls.append(("native", outer.j, None))
                    outer.k = 0
                    return [_Proposal(outer.natives[outer.j])]
                assert set(obs) == {"states"}
                outer.k += 1
                outer.calls.append(("retry", outer.j, np.array(obs["states"])))
                return [_Proposal(outer.retry_fn(outer.j, outer.k, np.array(obs["states"])))]
        self.policy = P()

    @property
    def ended(self):
        return self.j >= len(self.natives)

    def observe(self):
        return {"states": np.array(self.states[self.j]), "task_description": "x"}

    async def execute(self, actions):
        self.executed.append(np.array(actions))
        self.j += 1
        return len(actions)

    def record(self, event, **fields):
        self.records.append((event, fields))


def ctrl(entry="redecode"):
    return registry.load(entry).controller.build()


def run(controller, ctx):
    asyncio.run(controller.run(ctx))
    return ctx


def zero(s, rows=10):
    z = np.tile(AMEAN.copy(), (rows, 1))
    z[:, :6] += np.asarray(s, dtype=np.float64)[:6]
    return z


def state(rng, rx=3.1):
    s = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.9, 1.2), rx,
                  rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.02, -0.02])
    return s


def chunk(rng, grip=-1.0):
    c = rng.uniform(-0.5, 0.5, size=(10, 7))
    c[:, 6] = grip
    return c


def v12_reference(entry, states, natives):
    op = registry.load(entry).slots["action_op"].build()
    return [np.asarray(op.transform(np.array(c), {"states": np.array(s)}, {}))[:5] for s, c in zip(states, natives)]


def test_delta_and_token_replica():
    c = ctrl()
    assert np.allclose(c.delta, (SSTD + 1e-6) * 2 / 256, rtol=0, atol=0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        s = SMEAN + rng.normal(size=8) * SSTD * 0.6
        toks = state_tokens(s, SMEAN, SSTD)
        z = (s - SMEAN) / (SSTD + 1e-6)
        assert np.array_equal(toks, np.digitize(z, bins=np.linspace(-1, 1, 257)[:-1]) - 1)   # openpi L70
        n = c.tokens_changed(s, c.sent_state(s, +1))
        inband = np.abs(z) < 0.99
        assert n >= int(inband.sum()) - 1     # a one-bin shift moves every in-band token (float edge tolerance)
    s = SMEAN + 5 * SSTD                       # all clipped -> no token changes
    assert c.tokens_changed(s, c.sent_state(s, +1)) == 0


def test_no_zero_chunk_is_identity_and_v12_b():
    rng = np.random.default_rng(1)
    states = [state(rng) for _ in range(12)]
    natives = [chunk(rng, grip=float(rng.choice([-1, 1]))) for _ in range(12)]
    ctx = run(ctrl("redecode"), FakeCtx(states, natives))
    assert all(np.array_equal(e, n[:5]) for e, n in zip(ctx.executed, natives))
    assert [c[0] for c in ctx.calls] == ["native"] * 12
    ctx = run(ctrl("redecode-b"), FakeCtx(states, natives))
    ref = v12_reference("tilt-guard-v12-b", states, natives)
    assert all(np.array_equal(e, r) for e, r in zip(ctx.executed, ref))


def test_cr1_failed_retry_served_relative_to_sent_state_falls_through_to_z():
    rng = np.random.default_rng(2)
    states = [state(rng) for _ in range(4)]
    natives = [chunk(rng, grip=1.0), zero(states[1]), chunk(rng), chunk(rng)]
    # every retry fails the way openpi serves it: mean + s_sent (NOT a zero chunk relative to s_true)
    ctx = run(ctrl("redecode"), FakeCtx(states, natives, retry_fn=lambda j, k, s_sent: zero(s_sent)))
    rec = [f for e, f in ctx.records if e == "redecode"][1]
    assert rec["native_zero"] and rec["selected"] == "native_zero" and len(rec["retries"]) == 2
    assert all(r["zero_for_sent_state"] and not r["zero_for_true_state"] for r in rec["retries"])   # the C-R1 trap
    ref = v12_reference("tilt-guard-v12-z", states, natives)
    assert all(np.array_equal(e, r) for e, r in zip(ctx.executed, ref))              # = v1.2 Z bitwise
    assert np.array_equal(ctx.executed[1][:, :6], natives[0][5:10, :6]) and np.all(ctx.executed[1][:, 6] == 1.0)
    assert [c[0] for c in ctx.calls] == ["native", "native", "retry", "retry", "native", "native"]
    r1, r2 = [c[2] for c in ctx.calls if c[0] == "retry"]
    c0 = ctrl()
    assert np.array_equal(r1, c0.sent_state(states[1], +1)) and np.array_equal(r2, c0.sent_state(states[1], -1))


def test_cr1_retry1_fails_retry2_valid_rebased():
    rng = np.random.default_rng(3)
    states = [state(rng) for _ in range(3)]
    natives = [chunk(rng), zero(states[1]), chunk(rng)]
    valid = chunk(rng, grip=0.3)

    def retry(j, k, s_sent):
        return zero(s_sent) if k == 1 else valid
    ctx = run(ctrl("redecode"), FakeCtx(states, natives, retry_fn=retry))
    rec = [f for e, f in ctx.records if e == "redecode"][1]
    assert rec["selected"] == "retry2" and [r["zero_for_sent_state"] for r in rec["retries"]] == [True, False]
    c0 = ctrl()
    s_sent = c0.sent_state(states[1], -1)
    exp = valid[:5].copy()
    exp[:, :6] -= (s_sent - states[1])[:6]
    assert np.allclose(ctx.executed[1], exp, atol=1e-5, rtol=0)                       # C-R3 re-base (tol 1e-5)
    assert np.array_equal(ctx.executed[1][:, 6], valid[:5, 6])                         # gripper as decoded
    assert np.allclose(np.abs(ctx.executed[1][:, :6] - valid[:5, :6]), c0.delta[:6][None, :], atol=1e-12)


def test_retry1_valid_no_second_call_and_z_not_fired():
    rng = np.random.default_rng(4)
    states = [state(rng) for _ in range(3)]
    natives = [chunk(rng), zero(states[1]), zero(states[2])]
    valid = chunk(rng)
    ctx = run(ctrl("redecode"), FakeCtx(states, natives, retry_fn=lambda j, k, s: valid))
    assert [c[0] for c in ctx.calls] == ["native", "native", "retry", "native", "retry"]
    recs = [f for e, f in ctx.records if e == "redecode"]
    assert recs[1]["selected"] == "retry1" and recs[1]["tilt_guard"]["z_mode"] is None and not recs[1]["tilt_guard"]["zero"]
    assert recs[2]["tilt_guard"]["zero_run"] == 0                                       # a valid retry resets Z's run


def test_all_retries_fail_equals_v12_z_and_bz_random_episodes():
    rng = np.random.default_rng(5)
    for entry, ref_entry in (("redecode", "tilt-guard-v12-z"), ("redecode-b", "tilt-guard-v12-bz")):
        for _ in range(40):
            n = 30
            states, natives = [], []
            for j in range(n):
                tilt_hi = rng.random() < 0.5
                s = state(rng, rx=float(rng.choice([3.1, 1.6, 2.4])) if tilt_hi else 3.1)
                s[6], s[7] = (0.02, -0.02) if rng.random() < 0.6 else (0.0005, -0.0005)
                states.append(s)
                natives.append(zero(s) if rng.random() < 0.25 else chunk(rng, grip=float(rng.choice([-1, 1, 1]))))
            ctx = run(ctrl(entry), FakeCtx(states, natives, retry_fn=lambda j, k, s_sent: zero(s_sent)))
            ref = v12_reference(ref_entry, states, natives)
            assert all(np.array_equal(e, r) for e, r in zip(ctx.executed, ref))


def test_cards_budget_and_tilt_guard_params():
    for entry, src in (("redecode", "tilt-guard-v12-z"), ("redecode-b", "tilt-guard-v12-bz")):
        cfg = registry.load(entry)
        p = cfg.card["params"]
        assert p["tilt_guard"] == registry.load(src).card["params"] and p["tilt_guard_source_entry"] == src
        assert cfg.null == "shadow" and cfg.budgets.policy_calls == 312 == 3 * 104 and cfg.budgets.llm_calls == 0
        assert cfg.fidelity.value == "component" and list(cfg.substitutions) == []
        assert (p["shift_bins"], p["bin_width"], p["max_retries"], p["zero_chunk_tol"], p["exec_rows"]) == ([1, -1], 2 / 256, 2, 1e-4, 5)
        c = cfg.controller.build()
        assert isinstance(c.op, TiltGuardV12ActionOp) and c.op.zero == "substitute"
        assert c.op.envelope == ("off" if entry == "redecode" else "bte")
    with pytest.raises(ValueError):
        ReDecodeController(norm_stats_sha256=SHA, tilt_guard={"zero": "off", "norm_stats_sha256": SHA})
    with pytest.raises(RuntimeError):
        ReDecodeController(norm_stats_sha256="0" * 64, tilt_guard={"zero": "substitute"})


def test_missing_or_bad_state_never_retries():
    rng = np.random.default_rng(6)
    natives = [zero(state(rng)) for _ in range(3)]
    for bad in (None, np.arange(7.0), np.array([np.nan] * 8)):
        class C(FakeCtx):
            def observe(self):
                return {} if bad is None else {"states": bad}
        ctx = run(ctrl(), C([None] * 3, natives))
        assert all(c[0] == "native" for c in ctx.calls)
