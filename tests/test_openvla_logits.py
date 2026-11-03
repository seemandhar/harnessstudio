"""EXT openvla-contrast, part 1 (HarnessStudio venv, no torch): the native-decode `logits` readout.

Covers the review's U1, U2 (routing, R2), U5 (bin table, R1), U6 (fake readout), U7 (profiles), U8 (toy shadow
controller with the override positive control, R4), part of U9 (sabotage) and A8 (codec round trip).
The recorded decode itself (U3, U4, R3 self-check and its sabotage) needs torch + transformers: part 2,
tests/test_openvla_logits_server.py, run with the pinned OpenVLA env.
"""
from __future__ import annotations

import asyncio
import hashlib
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from harnessstudio.context.trace import fingerprint
from harnessstudio.contracts import load_contract
from harnessstudio.core import capabilities
from harnessstudio.core.arms import Arm
from harnessstudio.core.registry import discover, load
from harnessstudio.policy_ext import fake as fake_mod
from harnessstudio.policy_ext import protocol
from harnessstudio.policy_ext.fake import FakePolicyServer
from harnessstudio.profiles import load_profile
from harnessstudio.server.harness_server import HarnessServer
from harnessstudio.verify import tier0, tier1
from harnessstudio.verify.stack import run_fake, serve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "toys"))
import hs_ext_logits_toy as toy_mod  # noqa: E402

H = pytest.importorskip("harnessstudio.policies.openvla_hs")
from harnessstudio.policies import openvla_logits_hs as L  # noqa: E402

OPENVLA_HS_SHA256 = "e7c6a4c302fece60b338b9a6c14c7ac0cea4dea2057ff72b74c0abfa0aab174b"   # main e4ae8b8
EXT_FILES = {
    "harnessstudio/policies/openvla_logits_hs.py", "harnessstudio/profiles/openvla-libero-logits.yaml",
    "harnessstudio/profiles/fake-t3.yaml", "harnessstudio/policy_ext/fake.py", "harnessstudio/verify/tier1.py",
    "tests/test_openvla_logits.py", "tests/test_openvla_logits_server.py", "tests/toys/hs_ext_logits_toy.py",
    "docs/writing-a-method.md", "docs/extensions/README.md", "docs/extensions/openvla-contrast.md",
    "docs/superpowers/specs/2026-10-31-harnessstudio-v2-design.md",
}
# Stacked branch ext2/simpler-seeded (cut from this branch at 1fc5b10, rebased onto da817ba; declared in docs/extensions/simpler-seeded.md 3.8):
# its own files are part of the diff against main until openvla-contrast merges. Its scope is pinned separately by
# tests/test_simpler_seeded.py::test_simpler_branch_touches_only_its_files (diff against da817ba).
EXT_FILES |= {
    "harnessstudio/benchmarks/simpler_seeded.py", "harnessstudio/benchmarks/__init__.py",
    "harnessstudio/contracts/simpler_seeded.yaml", "harnessstudio/policies/openvla_simpler_hs.py",
    "harnessstudio/profiles/openvla-simpler-logits.yaml", "tests/test_simpler_seeded.py",
    "tests/test_openvla_simpler_server.py", "docs/extensions/simpler-seeded.md",
}


# ---- U1: source pins ------------------------------------------------------------------------------------------
def test_openvla_hs_is_byte_identical():
    assert hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest() == OPENVLA_HS_SHA256


EXT_BRANCH = "ext2/openvla-contrast"


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _scope_check_applies() -> tuple:
    """(applies, reason). The diff-scope check is about THIS extension's own, unmerged commit range (review
    critic-simpler-diff edit 2): it applies only while HEAD is on the extension branch (or detached at a commit of
    it) and HEAD is not yet contained in main. After the merge, on main and its later commits, and on every other
    branch (including stacked extensions) it does not apply; those diffs belong to other changes."""
    if _git("rev-parse", "--verify", "-q", "main").returncode:
        return False, "no local main to compare with"
    if _git("merge-base", "--is-ancestor", "HEAD", "main").returncode == 0:
        return False, "HEAD is contained in main (after the merge, or main itself): no unmerged extension commits to scope"
    branch = _git("symbolic-ref", "-q", "--short", "HEAD").stdout.strip()
    if branch:
        if branch != EXT_BRANCH:
            return False, f"on branch {branch!r}, not {EXT_BRANCH!r}: another change's diff"
        return True, ""
    if _git("merge-base", "--is-ancestor", "HEAD", EXT_BRANCH).returncode == 0:
        return True, ""                                 # detached at a commit of the extension branch
    return False, f"detached HEAD not on {EXT_BRANCH!r}"


def test_branch_touches_only_the_extension_files():
    applies, reason = _scope_check_applies()
    if not applies:
        pytest.skip(reason)
    base = _git("merge-base", "HEAD", "main")
    assert base.returncode == 0, base.stderr
    changed = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", base.stdout.strip()],
                             capture_output=True, text=True, check=True).stdout.split()
    assert set(changed) <= EXT_FILES, sorted(set(changed) - EXT_FILES)


# ---- a numpy stub with OpenVLA's de-tokenization constants ---------------------------------------------------
def stub_server(vocab=32000, n_bins=256, action_dim=7):
    bins = np.linspace(-1.0, 1.0, n_bins)
    rng = np.random.default_rng(0)
    q01 = rng.uniform(-1.0, -0.1, action_dim)
    q99 = rng.uniform(0.1, 1.0, action_dim)
    q01[-1], q99[-1] = 0.0, 1.0
    model = types.SimpleNamespace(
        vocab_size=vocab, config=types.SimpleNamespace(n_action_bins=n_bins),
        bin_centers=(bins[:-1] + bins[1:]) / 2.0,
        get_action_dim=lambda key=None: action_dim,
        get_action_stats=lambda key=None: {"q01": list(q01), "q99": list(q99), "mask": [True] * (action_dim - 1) + [False]})
    server = L.HarnessOpenVLALogitsServer.__new__(L.HarnessOpenVLALogitsServer)
    import threading
    server._infer_lock = threading.RLock()
    server._sampler_prefixes, server._decoders, server._states = ("hs_",), {}, {}
    server._model, server.unnorm_key = model, "stub"
    return server


def table_lookup(table, tokens, end=32000, n_bins=256):
    idx = np.clip(end - np.asarray(tokens, dtype=np.int64) - 1, 0, n_bins - 1)
    return table[idx, np.arange(table.shape[1])]


# ---- U5 (R1): bin table == _actions bitwise, boundary tokens ---------------------------------------------------
BOUNDARY = [0, 1, 31743, 31744, 31745, 31746, 31998, 31999, 32000, 32001, 32063]


def test_bin_table_equals_actions_bitwise():
    s = stub_server()
    table = s._bin_table()
    assert table.dtype == np.float32 and table.shape == (256, 7)
    assert np.array_equal(table[255], table[254])          # 255 centres: row 255 (token 31744) == row 254
    rng = np.random.default_rng(1)
    toks = rng.integers(0, 32064, (10_000, 7))
    toks[: len(BOUNDARY)] = np.array(BOUNDARY)[:, None]
    for t in toks:
        assert np.array_equal(table_lookup(table, t), s._actions(t)), t


def test_boundary_tokens_map_like_native_clip():
    s = stub_server()
    native_bin = np.clip(32000 - np.array(BOUNDARY) - 1, 0, s._model.bin_centers.shape[0] - 1)
    assert list(native_bin) == [254, 254, 254, 254, 254, 253, 1, 0, 0, 0, 0]
    assert native_bin[0] == 254                             # token 0 -> the top bin, not bin 0


def test_sabotaged_table_is_caught():
    s = stub_server()
    shifted = np.roll(s._bin_table(), 1, axis=0)
    t = np.full(7, 31900)
    assert not np.array_equal(table_lookup(shifted, t), s._actions(t))


# ---- U2 (R2): routing --------------------------------------------------------------------------------------
@pytest.mark.parametrize("readouts", [(), ("velocity",), ("tokens",), ("logit",)])
def test_requests_without_logits_run_the_parent_unchanged(readouts):
    s = stub_server()
    with mock.patch.object(H.HarnessOpenVLAServer, "sample_chunks", autospec=True, return_value=["P"]) as parent, \
            mock.patch.object(L.HarnessOpenVLALogitsServer, "_predict_recorded", side_effect=AssertionError):
        out = s.sample_chunks({"x": 1}, [3, 4], None, readouts, "CTX")
    assert out == ["P"]
    parent.assert_called_once_with(s, {"x": 1}, [3, 4], None, readouts, "CTX")


def test_parent_refusal_message_is_unchanged():
    s = stub_server()
    with pytest.raises(ValueError, match=r"^this OpenVLA server offers no readouts \(asked for \['velocity'\]\)$"):
        s.sample_chunks({}, [0], None, ("velocity",), None)


@pytest.mark.parametrize("readouts,sampler", [(("logits", "velocity"), None), (("velocity", "logits"), None),
                                              (("logits", "logits"), None),
                                              (("logits",), {"factory": "hs_x.y:Z", "params": {}})])
def test_logits_refusals(readouts, sampler):
    s = stub_server()
    with mock.patch.object(L.HarnessOpenVLALogitsServer, "_predict_recorded", side_effect=AssertionError), \
            mock.patch.object(H.HarnessOpenVLAServer, "sample_chunks", side_effect=AssertionError):
        with pytest.raises(ValueError):
            s.sample_chunks({}, [0], sampler, readouts, None)


def test_sabotaged_routing_is_caught():
    """A subclass that sends no-readout requests to the recorded path must fail the routing test above."""
    class Bad(L.HarnessOpenVLALogitsServer):
        def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
            if not sampler:
                return self._predict_recorded(obs)
            return super().sample_chunks(obs, seeds, sampler, readouts, ctx)
    s = stub_server()
    s.__class__ = Bad
    with mock.patch.object(H.HarnessOpenVLAServer, "sample_chunks", autospec=True, return_value=["P"]), \
            mock.patch.object(L.HarnessOpenVLALogitsServer, "_predict_recorded", side_effect=AssertionError):
        with pytest.raises(AssertionError):
            s.sample_chunks({"x": 1}, [3], None, (), "CTX")


def test_capabilities():
    assert L.HarnessOpenVLALogitsServer.harness_capabilities == H.HarnessOpenVLAServer.harness_capabilities | {"logits_readout"}
    assert "logits_readout" not in H.HarnessOpenVLAServer.harness_capabilities


# ---- U6: fake readout ----------------------------------------------------------------------------------------
OBS = {"images": {"cam": np.full((8, 8, 3), 7, dtype=np.uint8)}, "task_description": "t", "state": [0.0, 0.0]}


def readout_never_changes_chunks(server) -> bool:
    plain = server.sample_chunks(OBS, [1, 2], None, (), None)
    with_logits = server.sample_chunks(OBS, [1, 2], None, ("logits",), None)
    return all(np.array_equal(a, b) for (a, _), (b, _) in zip(plain, with_logits))


def test_fake_logits_readout():
    f = FakePolicyServer(chunking="harness", seeded=True)
    out = f.sample_chunks(OBS, [1, 2], None, ("logits",), None)
    (_, e1), (_, e2) = out
    assert e1["logits"].dtype == np.float32 and e1["logits"].shape == (7, fake_mod.FAKE_LOGITS_WIDTH)
    assert np.array_equal(e1["logits"], e2["logits"])                     # greedy-like: no seed dependence (A3)
    assert np.array_equal(e1["logits_tokens"], e1["logits"].argmax(-1))
    assert e1["logits_bin_actions"].shape == (fake_mod.FAKE_N_BINS, 7)
    other = dict(OBS, images={"cam": np.full((8, 8, 3), 8, dtype=np.uint8)})
    (_, e3), = f.sample_chunks(other, [1], None, ("logits",), None)
    assert not np.array_equal(e1["logits"], e3["logits"])                 # depends on the observation
    (_, plain), = f.sample_chunks(OBS, [1], None, (), None)
    assert not any(k.startswith("logits") for k in plain)
    assert readout_never_changes_chunks(f)
    assert "logits_readout" in f.harness_capabilities


def test_sabotaged_fake_readout_is_caught():
    class Bad(FakePolicyServer):
        def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
            out = super().sample_chunks(obs, seeds, sampler, readouts, ctx)
            return [(c + (0.5 if "logits" in readouts else 0.0), e) for c, e in out]
    assert not readout_never_changes_chunks(Bad(chunking="harness", seeded=True))


# ---- A8: logits survive the vla-eval wire bitwise ---------------------------------------------------------------
def test_logits_round_trip_through_vla_eval_codec():
    from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message
    rng = np.random.default_rng(5)
    extras = {"logits": rng.normal(0, 3, (7, 32064)).astype(np.float32), "logits_tokens": np.arange(7, dtype=np.int32),
              "logits_bin_actions": rng.uniform(-1, 1, (256, 7)).astype(np.float32), "logits_token_end": 32000,
              "logits_n_bins": 256}
    payload = protocol.make_response([(np.zeros((1, 7), np.float32), extras)])
    msg = unpack_message(pack_message(Message(MessageType.ACTION, payload, timestamp=0.0)))
    (_, got), = protocol.parse_response(msg.payload, 1)
    for k, v in extras.items():
        if isinstance(v, np.ndarray):
            assert got[k].dtype == v.dtype and got[k].shape == v.shape and np.array_equal(got[k], v), k
        else:
            assert got[k] == v
    assert got["logits"].tobytes() == extras["logits"].tobytes()


# ---- U7: profiles ---------------------------------------------------------------------------------------------
def _old_default_profile(config):            # tier1.default_profile on main e4ae8b8
    for name in ("fake-t0", "fake-t1", "fake-t2"):
        if set(config.required) <= set(load_profile(name).capabilities):
            return name
    return "fake-t2"


def _servable_by_old_profiles(config) -> bool:
    return any(set(config.required) <= set(load_profile(n).capabilities) for n in ("fake-t0", "fake-t1", "fake-t2"))


def _expected_new_profile(config) -> str:
    """Invariance where the old code could serve the method; logits_readout methods it could not serve -> fake-t3."""
    if _servable_by_old_profiles(config):
        return _old_default_profile(config)
    if set(config.required) <= set(load_profile("fake-t3").capabilities):
        return "fake-t3"
    return _old_default_profile(config)        # still unservable: the unchanged fallback


def test_tier1_profile_choice_unchanged_for_installed_methods():
    """Review E1 (critic-contrast-method): independent of which methods are installed (e.g. hs_pcd after the merge)."""
    checked = 0
    for name in sorted(discover()):
        try:
            config = load(name)
        except Exception:  # noqa: BLE001  (a method that cannot load fails its own tier 0 in regress)
            continue
        assert tier1.default_profile(config) == _expected_new_profile(config), name
        checked += 1
    for config in (toy_mod.toy(), toy_mod.toy_alpha0()):   # a logits_readout method is always covered
        assert not _servable_by_old_profiles(config)
        assert _old_default_profile(config) == "fake-t2" and tier1.default_profile(config) == "fake-t3"
        checked += 1
    assert checked > 0


def test_new_profiles():
    for name in ("openvla-libero-logits", "fake-t3"):
        assert "logits_readout" in capabilities.check(load_profile(name).capabilities)
    assert "logits_readout" not in load_profile("openvla-libero-hs").capabilities
    hs, lg = load_profile("openvla-libero-hs"), load_profile("openvla-libero-logits")
    assert set(lg.capabilities) - set(hs.capabilities) == {"logits_readout"}
    assert (lg.chunking, lg.execute_horizon) == (hs.chunking, hs.execute_horizon)
    assert set(load_profile("fake-t3").capabilities) - set(load_profile("fake-t2").capabilities) == {"logits_readout"}


# ---- U8 (+R4): toy shadow controller -------------------------------------------------------------------------
def test_toy_tier0_and_tier1_pass():
    config = toy_mod.toy()
    assert tier0.check(config) == []
    assert tier0.check(toy_mod.toy_alpha0()) == []          # the GPU G5 variant registers too
    assert tier1.default_profile(config) == "fake-t3"
    report = tier1.run(config)
    assert report.passed, "\n".join(report.lines())


class SpyFake(FakePolicyServer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.logits_requests = []

    def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
        if "logits" in readouts:
            self.logits_requests.append({k: (dict(v) if isinstance(v, dict) else v) for k, v in obs.items()})
        return super().sample_chunks(obs, seeds, sampler, readouts, ctx)


def run_toy(arm, alpha=0.8):
    import dataclasses
    policy = SpyFake(chunking="harness", seeded=True)
    contract = dataclasses.replace(load_contract("fake"), default_horizon=12)
    with serve(policy) as ph:
        harness = HarnessServer(toy_mod.toy(alpha), policy_url=ph.url, contract=contract, arm=arm,
                                profile=load_profile("fake-t3"))
        with serve(harness) as hh:
            run_fake(hh.url, n_tasks=1, episodes=1, horizon=12)
            harness.wait_idle()
    return harness.records, policy


def test_override_reaches_the_policy_and_changes_tokens():
    recs, policy = run_toy(Arm.CANDIDATE)
    (rec,) = recs
    assert rec["valid"], rec.get("invalid_reason")
    reqs = policy.logits_requests
    steps = rec["accounting"]["steps"]
    assert len(reqs) == 2 * steps
    received = {r["sha256"] for r in policy.received}
    for real, cf in zip(reqs[0::2], reqs[1::2]):
        key = next(iter(real["images"]))
        want = toy_mod.counterfactual(real["images"][key], 37)
        assert np.array_equal(cf["images"][key], want)                         # the override reached the server
        assert {k for k in cf if k != "images"} == {k for k in real if k != "images"}
        assert all(fingerprint(cf[k]) == fingerprint(real[k]) for k in real if k != "images")
        assert fingerprint(cf) in received and fingerprint(real) in received  # received-hash assertion (R4)
        assert fingerprint(cf) != fingerprint(real)
    stats = rec["accounting"]["extra"]["logits_toy"]
    assert stats["steps"] == steps and stats["changed_dims"] > 0 and stats["skipped"] == 0
    assert rec["accounting"]["method_policy_samples"] == 2 * steps


def test_null_executes_native_and_pays_the_contrast():
    base, _ = run_toy(Arm.BASELINE)
    null, policy = run_toy(Arm.NULL)
    assert base[0]["action_sha256"] == null[0]["action_sha256"]
    acc = null[0]["accounting"]
    assert acc["shadow_policy_samples"] == acc["method_policy_samples"] == 2 * acc["steps"]
    assert acc["native_policy_samples"] >= 1
    assert len(policy.logits_requests) == 2 * acc["steps"]


def test_alpha_zero_combine_is_the_real_argmax():
    rng = np.random.default_rng(2)
    s = rng.normal(0, 3, (7, 32064)).astype(np.float32)
    c = rng.normal(0, 3, (7, 32064)).astype(np.float32)
    tokens, info = toy_mod.pcd_code_combine(s, c, 0.0, range(6), 7)
    assert np.array_equal(tokens, s.argmax(-1)) and info["changed"] == 0
