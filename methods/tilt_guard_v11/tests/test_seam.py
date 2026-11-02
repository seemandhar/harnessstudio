"""Real handler seam: the frozen entry points run inside harnessstudio's own DecisionLoop.decide (unmodified framework
code) with a minimal fake episode context that supplies LIBERO-style public 'states' (the tier-1 fake contract uses
'state', so tier 1 only exercises the inactive path). Checks: the candidate executes transformed[:5]; the derived null
(and decide() under the baseline arm) execute the native chunk bitwise while the op still runs (shadow);
diagnostics are logged per decision; components are built per episode."""
from __future__ import annotations

import asyncio
from collections import deque

import numpy as np

from harnessstudio.components.decision_loop import DecisionLoop
from harnessstudio.context.policy import PolicyProfile, Proposal
from harnessstudio.core import registry
from harnessstudio.core.arms import Arm

from test_tilt_guard_v10_suite import AMEAN, state, zero_chunk  # noqa: E402


class FakeAccounting:
    decisions = 0


class FakePolicy:
    def __init__(self, chunks):
        self.chunks = deque(chunks)

    async def sample(self, k, obs=None, instruction=None, sampler=None, readouts=()):
        return [Proposal.make(self.chunks.popleft())]


class FakeCtx:
    def __init__(self, arm, chunks, states, horizon):
        self._arm = arm
        self.policy = FakePolicy(chunks)
        self._profile = PolicyProfile(capabilities=frozenset({"act", "chunk", "seeded_sampling"}),
                                      chunking="harness", execute_horizon=5)
        self._native_queue = deque()
        self.accounting = FakeAccounting()
        self.memory = {}
        self.instruction = "put both moka pots on the stove"
        self.emitted = []
        self.logs = []
        self.states = list(states)
        self.horizon = horizon

    @property
    def ended(self):
        return len(self.emitted) >= self.horizon

    def observe(self):
        d = len(self.emitted) // 5
        return {"states": np.array(self.states[min(d, len(self.states) - 1)]), "task_description": self.instruction}

    async def _emit(self, a):
        self.emitted.append(np.array(a, copy=True))

    def _log(self, event, **fields):
        self.logs.append((event, fields))


def scenario():
    """4 holding close decisions, a release (boundary) at high tilt, then a zero chunk and two more decisions."""
    rng = np.random.default_rng(0)
    s_hold = state(5.0)
    s_hi = state(110.0, 30.0)
    chunks, states = [], []
    for _ in range(4):
        c = rng.uniform(-0.5, 0.5, size=(10, 7))
        c[:, 6] = 1.0
        chunks.append(c)
        states.append(s_hold)
    c = rng.uniform(-0.5, 0.5, size=(10, 7))
    c[:, 6] = -1.0
    chunks.append(c)
    states.append(s_hi)
    chunks.append(zero_chunk(s_hi))
    states.append(s_hi)
    for _ in range(2):
        c = rng.uniform(-0.5, 0.5, size=(10, 7))
        c[:, 6] = -1.0
        chunks.append(c)
        states.append(s_hi)
    return chunks, states


def run(entry, arm):
    cfg = registry.load(entry)
    chunks, states = scenario()
    loop = DecisionLoop(cfg)
    ctx = FakeCtx(arm, chunks, states, horizon=5 * len(chunks))

    async def go():
        while not ctx.ended:
            await loop.decide(ctx)
    asyncio.run(go())
    return np.array(ctx.emitted), ctx, loop, chunks


def native_trajectory(chunks):
    return np.concatenate([np.asarray(c)[:5] for c in chunks])


def test_seam_candidate_null_baseline():
    for entry in ("tilt-guard-v11-gbz", "tilt-guard-v11-gb", "tilt-guard-v11-gyaw", "tilt-guard-v11-gz", "tilt-guard-v11-zsign"):
        cand, cctx, cloop, chunks = run(entry, Arm.CANDIDATE)
        null, nctx, nloop, _ = run(entry, Arm.NULL)
        base, bctx, bloop, _ = run(entry, Arm.BASELINE)
        native = native_trajectory(chunks)
        assert np.array_equal(null, native) and np.array_equal(base, native)        # null/baseline execute native
        assert not np.array_equal(cand, native)                                       # candidate intervened
        assert np.array_equal(cand[:20], native[:20])                                 # bitwise prefix before activation
        op_c = cloop.action_op
        assert op_c.stats["decisions"] == len(chunks)
        assert nloop.action_op.stats["decisions"] == len(chunks)                      # null runs the op (shadow)
        # (the real baseline arm never reaches decide(): harness_server executes the native queue; tier 1 checks it)
        diags = [f.get("diagnostics", {}).get("action_op") for e, f in cctx.logs if e == "decision"]
        assert len(diags) == len(chunks) and all(d is not None for d in diags)
        if op_c.envelope != "off":
            assert op_c.stats["b_active"] >= 1 and op_c.detector.boundary == 4
            assert np.array_equal(cand[:, :3][:20], native[:, :3][:20])
        if op_c.zero != "off":
            assert op_c.stats["zero_detected"] == 1
            z = cand[25:30]
            assert not np.allclose(z[:, :6], (AMEAN[:6] + state(110.0, 30.0)[:6])[None, :])


def test_seam_candidate_executes_op_output_rows():
    cand, cctx, cloop, chunks = run("tilt-guard-v11-gbz", Arm.CANDIDATE)
    # replay the same op standalone on the same inputs: executed rows == transformed[:5]
    cfg = registry.load("tilt-guard-v11-gbz")
    op = cfg.slots["action_op"].build()
    _, states = scenario()
    rows = [np.asarray(op.transform(np.array(c), {"states": s}, {}))[:5] for c, s in zip(chunks, states)]
    assert np.array_equal(cand, np.concatenate(rows))
