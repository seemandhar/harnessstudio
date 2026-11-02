"""Active path through the real in-process vla-eval stack (harness server, fake seeded policy, scripted LLM = the card's
TIER1_LLM_RESPONSES) with a fake benchmark exposing LIBERO-like public 'states' (8-d) and images.agentview (256x256).
Tier 1 itself never activates the controller (the fake contract has no 'states'); this checks: shadow null == baseline
bitwise in every case, candidate == baseline bitwise until the first re-home step, zero-re-home cases bitwise native,
and that the null pays the same kind of VLM calls (shadow)."""
from __future__ import annotations

import contextlib
import dataclasses
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

HORIZON = 260


def _rot(aa):
    aa = np.asarray(aa, float); th = np.linalg.norm(aa)
    if th < 1e-12:
        return np.eye(3)
    k = aa / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        B = (R + np.eye(3)) / 2; i = int(np.argmax(np.diag(B))); k = B[:, i] / np.sqrt(max(B[i, i], 1e-12))
        return k / np.linalg.norm(k) * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)) * th


class ArmBenchmark(FakeBenchmark):
    """3-D position + orientation integrated from the actions (OSC-like, 0.02 m / 0.1 rad per unit), finger width 0.02
    while the gripper command is close (holding) else 0.08."""

    def reset(self, task):
        out = super().reset(task)
        self._p3 = np.array([self._pos[0] * 0.2, self._pos[1] * 0.2, 1.1])
        self._R = _rot([np.pi, 0.0, 0.0])
        self._w = 0.08
        return out

    def step(self, action):
        a = np.asarray(action["actions"], dtype=np.float64).reshape(-1)
        self._p3 = self._p3 + 0.02 * np.clip(a[:3], -1, 1)
        self._R = _rot(0.1 * np.clip(a[3:6], -1, 1)) @ self._R
        self._w = 0.02 if a[6] > 0 else 0.08
        return super().step(action)

    def make_obs(self, raw_obs, task):
        obs = super().make_obs(raw_obs, task)
        w = self._w
        obs["states"] = np.concatenate([self._p3, _log(self._R), [w / 2, -w / 2]]).astype(np.float32)
        level = int(abs(self._p3[0]) * 1000) % 256
        img = np.zeros((256, 256, 3), np.uint8); img[..., 0] = level; img[..., 1] = int(self._p3[2] * 100) % 256
        obs["images"] = {"agentview": img}
        return obs


def run_arm(method: str, arm: Arm):
    config = registry.load(method)
    # same execution cadence as the real pi0fast profile (execute_horizon 5), seeded like fake-t2
    prof = Path(tempfile.mkdtemp(prefix="hs-vlm-prof-")) / "fake-exec5.yaml"
    prof.write_text("capabilities: [act, chunk, seeded_sampling, isolated_streams]\nchunking: harness\nexecute_horizon: 5\n")
    profile = load_profile(str(prof))
    policy = FakePolicyServer(chunking="harness", seeded=profile.seeded, chunk_len=10)
    base = load_contract("fake")
    contract = dataclasses.replace(base, public_keys=tuple(base.public_keys) + ("states",), default_horizon=HORIZON)
    with stack.serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract, arm=arm, profile=profile, scripted_llm=True)
        with stack.serve(harness) as hh:
            cfg = stack.benchmark_config(hh.url, n_tasks=3, episodes=3, horizon=HORIZON,
                                         instruction="put the bowl on the plate and the mug in the basket")
            cfg["benchmarks"][0]["benchmark"] = f"{__name__}:ArmBenchmark"
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                cfg["output_dir"] = tmp
                stack.run(cfg, docker=False, no_save=False)
            harness.wait_idle()
    return {(r["task"], r["episode_idx"]): r for r in harness.records}


def events(rec, name):
    return [e for e in rec["events"] if e.get("t") == "event" and e.get("name") == name]


@pytest.mark.parametrize("method", ["vlm-rehome", "vlm-rehome-tiltgate", "vlm-rehome-forced"])
def test_shadow_null_and_candidate_prefix(method):
    base = run_arm(method, Arm.BASELINE)
    null = run_arm(method, Arm.NULL)
    cand = run_arm(method, Arm.CANDIDATE)
    assert len(base) == len(null) == len(cand) == 9
    assert all(r["valid"] for d in (base, null, cand) for r in d.values()), [r["invalid_reason"] for d in (base, null, cand) for r in d.values()]
    for k in base:
        assert null[k]["action_sha256"] == base[k]["action_sha256"], k        # shadow null: bitwise native
    n_rehome, n_bound = 0, 0
    for k, rec in cand.items():
        n_bound += len(events(rec, "vpr_boundary"))
        st = events(rec, "vpr_rehome_start")
        a, b = rec["action_sha256"], base[k]["action_sha256"]
        if not st:
            assert a == b, k
            continue
        n_rehome += 1
        s = st[0]["step"]
        assert a[:s] == b[:s] and a[s:] != b[s:], k
    assert n_bound >= 3 and n_rehome >= 1, (n_bound, n_rehome)
    calls_null = sum(r["accounting"]["llm_calls"] for r in null.values())
    calls_cand = sum(r["accounting"]["llm_calls"] for r in cand.values())
    if method == "vlm-rehome-forced":
        assert calls_null == calls_cand == 0
    else:
        assert calls_null > 0 and calls_cand > 0
        assert all(r["accounting"]["llm_calls"] <= 8 for d in (null, cand) for r in d.values())
        checked = [e for r in cand.values() for e in events(r, "vpr_check") if e.get("checked")]
        assert {e["verdict"] for e in checked} <= {"normal", "abnormal", "invalid_json", "unsure"}
