"""Active-path test through the real in-process vla-eval stack (harness server + fake policy), with a fake benchmark
that exposes a LIBERO-like public 'states' vector whose rx leaves the band on both sides. Tier 1 itself never
activates the op (the fake contract has 'state', not 'states'), so this checks the wiring of the active path:
null == baseline bitwise, candidate == baseline up to the first activation, activations in both arms' diagnostics,
and the executed candidate action equals the op's transform of the native chunk."""
from __future__ import annotations

import dataclasses

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

HORIZON = 40


def states_of(pos):
    rx = 2.4 + 2.0 * float(pos[0])            # crosses [1.5, 3.27] on both sides as the fake robot moves
    return np.array([float(pos[0]), float(pos[1]), 1.0, rx, 0.0, -0.08, 0.03, -0.03], dtype=np.float32)


class StatesBenchmark(FakeBenchmark):
    def make_obs(self, raw_obs, task):
        obs = super().make_obs(raw_obs, task)
        obs["states"] = states_of(raw_obs["pos"])
        return obs


def run_arm(method: str, arm: Arm):
    config = registry.load(method)
    profile = load_profile("fake-t0")
    policy = FakePolicyServer(chunking="harness", seeded=profile.seeded, chunk_len=10)
    base = load_contract("fake")
    contract = dataclasses.replace(base, public_keys=tuple(base.public_keys) + ("states",), default_horizon=HORIZON)
    with stack.serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract, arm=arm, profile=profile,
                                scripted_llm=True)
        with stack.serve(harness) as hh:
            cfg = stack.benchmark_config(hh.url, n_tasks=3, episodes=2, horizon=HORIZON,
                                         instruction="pick up the block and place it in the bowl")
            cfg["benchmarks"][0]["benchmark"] = f"{__name__}:StatesBenchmark"
            import contextlib
            import io
            import tempfile
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                cfg["output_dir"] = tmp
                stack.run(cfg, docker=False, no_save=False)
            harness.wait_idle()
    return {(r["task"], r["episode_idx"]): r for r in harness.records}


def decisions(rec):
    return [e for e in rec["events"] if e.get("t") == "decision"]


@pytest.mark.parametrize("method", ["band-guard", "band-guard-signctrl"])
def test_active_path_through_harness_server(method):
    base = run_arm(method, Arm.BASELINE)
    null = run_arm(method, Arm.NULL)
    cand = run_arm(method, Arm.CANDIDATE)
    assert len(base) == len(null) == len(cand) == 6
    assert all(r["valid"] for d in (base, null, cand) for r in d.values())
    for k in base:
        assert null[k]["action_sha256"] == base[k]["action_sha256"]          # derived null: bitwise native
    n_active_cases = 0
    n_active_null = 0
    for k, rec in cand.items():
        steps, first_step = 0, None
        for ev in decisions(rec):
            diag = (ev.get("diagnostics") or {}).get("action_op") or {}
            if diag.get("active") and first_step is None:
                first_step = steps
            steps += int(ev.get("executed", 0))
        n_active_null += sum(1 for ev in decisions(null[k])
                             if ((ev.get("diagnostics") or {}).get("action_op") or {}).get("active"))
        if first_step is None:
            assert rec["action_sha256"] == base[k]["action_sha256"]      # zero activations: bitwise native
            continue
        n_active_cases += 1
        a, b = rec["action_sha256"], base[k]["action_sha256"]
        assert a[:first_step] == b[:first_step]                            # bitwise prefix up to the first activation
        assert a[first_step] != b[first_step]                              # the activation changes the executed action
    assert n_active_cases >= 2 and n_active_null >= 1


RECORDED: list = []


class RecordingOp:
    """Wraps the frozen op and records (native, returned) per call; test-only."""

    def __init__(self, **params):
        from hs_band_guard.action_op import BandGuardActionOp
        self.op = BandGuardActionOp(**params)

    def diagnostics(self):
        return self.op.diagnostics()

    def transform(self, chunk, obs, memory):
        out = self.op.transform(chunk, obs, memory)
        RECORDED.append((np.array(chunk), np.array(out), dict(self.op.diagnostics())))
        return out


def test_executed_candidate_action_equals_transform(monkeypatch):
    """The candidate executes the op's returned chunk (all rows here: fake-t0 has no execute_horizon)."""
    from harnessstudio.context.trace import fingerprint
    from harnessstudio.core.method import ModuleRef
    cfg = registry.load("band-guard")
    params = dict(cfg.slots["action_op"].params)
    rec_cfg = dataclasses.replace(cfg, slots={"action_op": ModuleRef(f"{__name__}:RecordingOp", params)})
    monkeypatch.setattr(registry, "load", lambda name: rec_cfg)
    RECORDED.clear()
    cand = run_arm("band-guard", Arm.CANDIDATE)
    assert len(cand) == 6
    calls = list(RECORDED)
    # Calls arrive in case run order (sync fake benchmark). Decisions are logged after execution; an episode that
    # ends mid-chunk drops its last decision event, whose call is matched against the remaining hashes.
    n_active = 0
    ci = 0
    for k in sorted(cand):
        rec_hashes = cand[k]["action_sha256"]
        step = 0
        for ev in decisions(cand[k]):
            native, out, diag = calls[ci]
            ci += 1
            n = int(ev.get("executed", 0))
            for i in range(n):
                assert fingerprint(np.asarray(out[i])) == rec_hashes[step + i]
            if diag.get("active"):
                n_active += 1
                assert not np.array_equal(out, native)
                assert np.array_equal(out[:, 3], ((native[:, 3].astype(np.float64) - diag["s"]) + diag["clip"]).astype(out.dtype))
            else:
                assert np.array_equal(out, native)
            step += n
        if ci < len(calls) and step < len(rec_hashes):
            native, out, diag = calls[ci]
            ci += 1
            for i in range(len(rec_hashes) - step):
                assert fingerprint(np.asarray(out[i])) == rec_hashes[step + i]
    assert n_active >= 2 and ci == len(calls)
