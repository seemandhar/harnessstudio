"""hs_pcd tests (CPU, fake stack): combine fallbacks, table execution, evidence, null semantics, the HTTP perception
client against a mock of the service API, error handling. Needs a HarnessStudio with the EXT openvla-contrast
`logits` readout (fake-t3); run in the isolated test venv:

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 <hs2-test venv>/bin/python -B -m pytest -q -p no:cacheprovider tests
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from harnessstudio.context.trace import fingerprint
from harnessstudio.contracts import load_contract
from harnessstudio.core.arms import Arm
from harnessstudio.policy_ext.fake import FakePolicyServer
from harnessstudio.profiles import load_profile
from harnessstudio.server.harness_server import HarnessServer
from harnessstudio.verify import tier0, tier1
from harnessstudio.verify.stack import run_fake, serve

import hs_pcd
from hs_pcd import controller as C
from hs_pcd.combine import pcd_code_combine, pcd_code_final, table_action, token_stats

V = 384


def rows(rng, t, v=V):
    return rng.normal(0, 3, (t, v)).astype(np.float32)


# ---- combine (openvla_contrast.py:40-58) ------------------------------------------------------------------------
def test_full_streams_contrast_motion_dims_only():
    rng = np.random.default_rng(0)
    s, c = rows(rng, 7), rows(rng, 7)
    tok, info, fin = pcd_code_final(s, c, 0.8)
    want = s.copy()
    want[:-1] = (1 + 0.8) * s[:-1] - 0.8 * c[:-1]
    assert np.array_equal(fin, want) and np.array_equal(tok, want.argmax(-1))
    assert tok[6] == s[6].argmax() and info["contrasted"] and not info["a_fallback"]


def test_a_short_zero_fallback_still_contrasts_with_full_b():
    rng = np.random.default_rng(1)
    s, c = rows(rng, 4), rows(rng, 7)
    tok, info, fin = pcd_code_final(s, c, 0.8)
    assert info["a_fallback"] and info["contrasted"]
    assert np.array_equal(tok[:6], c[:6].argmin(-1))              # -0.8 * c  -> argmin c (R1b)
    assert tok[6] == 0                                             # zero row -> token 0 (-> bin 254)


@pytest.mark.parametrize("tb", [6, 5, 1])
def test_b_short_skips_the_contrast(tb):
    rng = np.random.default_rng(2)
    s, c = rows(rng, 7), rows(rng, tb)
    tok, info = pcd_code_combine(s, c, 0.8)
    assert not info["contrasted"] and np.array_equal(tok, s.argmax(-1))


def test_both_short_is_all_token_zero():
    rng = np.random.default_rng(3)
    tok, info = pcd_code_combine(rows(rng, 2), rows(rng, 3), 0.8)
    assert info["a_fallback"] and not info["contrasted"] and list(tok) == [0] * 7


def test_alpha_zero_is_native_and_plus_form_is_not_bitwise():
    rng = np.random.default_rng(4)
    s, c = rows(rng, 7, 32064), rows(rng, 7, 32064)
    tok, _, fin = pcd_code_final(s, c, 0.0)
    assert np.array_equal(tok, s.argmax(-1)) and np.array_equal(fin, s)
    _, _, fin8 = pcd_code_final(s, c, 0.8)
    plus = s[:-1] + 0.8 * (s[:-1] - c[:-1])
    assert not np.array_equal(plus, fin8[:-1])                    # R9d: the algebraic twin is a different number


def test_table_action_and_token_stats():
    table = np.arange(256 * 7, dtype=np.float32).reshape(256, 7)
    extras = {"logits_bin_actions": table, "logits_token_end": 32000, "logits_n_bins": 256}
    toks = np.array([31999, 31744, 0, 32000, 31998, 31900, 31745])
    act = table_action(extras, toks)
    idx = np.clip(32000 - toks - 1, 0, 255)
    assert np.array_equal(act, table[idx, np.arange(7)])
    st = token_stats(toks, np.array([31999, 31745, 5, 32000, 31998, 31901, 31745]), extras)
    assert st["changed_dims"] == [1, 2, 5] and st["non_action_vocab_dims"] == [2, 3]


# ---- stack: candidate / null / baseline --------------------------------------------------------------------------
class SpyFake(FakePolicyServer):
    """Records logits requests; optionally truncates the rows of stream A or B (fallback paths)."""

    def __init__(self, *a, truncate=None, **k):
        super().__init__(*a, **k)
        self.logits_requests, self.truncate, self._n = [], truncate, 0

    def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
        out = super().sample_chunks(obs, seeds, sampler, readouts, ctx)
        if "logits" in readouts:
            self.logits_requests.append({k: (dict(v) if isinstance(v, dict) else v) for k, v in obs.items()})
            stream = "A" if self._n % 2 == 0 else "B"
            self._n += 1
            if self.truncate and self.truncate[0] == stream:
                for _, e in out:
                    e["logits"] = e["logits"][: self.truncate[1]]
                    e["logits_tokens"] = e["logits_tokens"][: self.truncate[1]]
        return out


def run_pcd(arm, contract=None, truncate=None, params=None, horizon=12):
    config = hs_pcd.config()
    if params:
        config = dataclasses.replace(config, controller=dataclasses.replace(
            config.controller, params={**config.controller.params, **params}))
    policy = SpyFake(chunking="harness", seeded=True, truncate=truncate)
    contract = contract or dataclasses.replace(load_contract("fake"), default_horizon=horizon)
    with serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=contract, arm=arm, profile=load_profile("fake-t3"))
        with serve(harness) as hh:
            run_fake(hh.url, n_tasks=1, episodes=1, horizon=horizon, instruction="pick coke can")
            harness.wait_idle()
    return harness.records, policy


def events(rec, name):
    return [e for e in rec["events"] if e.get("t") == "event" and e.get("name") == name]


def test_card_tier0_and_tier1():
    config = hs_pcd.config()
    assert config.name == "pcd" and config.null == "shadow"
    assert tier0.check(config) == []
    assert tier1.default_profile(config) == "fake-t3"
    report = tier1.run(config)
    assert report.passed, "\n".join(report.lines())


def test_candidate_evidence_and_override():
    (rec,), policy = run_pcd(Arm.CANDIDATE)
    assert rec["valid"], rec.get("invalid_reason")
    steps = rec["accounting"]["steps"]
    pcd = rec["accounting"]["extra"]["pcd"]
    assert pcd["perception"] == "stub" and pcd["steps"] == steps
    assert rec["accounting"]["method_policy_samples"] == 2 * steps
    assert rec["accounting"]["tool_calls"] == steps + 1                    # reset + one /step per env step
    assert pcd["first_detection_step"] == 2 and pcd["mask_present_steps"]["stub object"] == steps - 2
    assert pcd["contrast_skipped"] == 0 and pcd["a_fallback"] == 0
    assert sum(pcd["changed_tokens_per_dim"]) > 0 and pcd["changed_tokens_per_dim"][6] == 0   # gripper = real stream
    ev = events(rec, "pcd")
    assert len(ev) == steps
    need = {"env_step", "phase", "per_target", "cf_differs", "contrast_skipped", "a_fallback", "changed_dims",
            "non_action_vocab_dims", "tokens", "native_tokens", "gap_s", "gap_f"}
    assert need <= set(ev[0]), sorted(need - set(ev[0]))
    assert [e["env_step"] for e in ev] == list(range(steps))
    reqs = policy.logits_requests
    assert len(reqs) == 2 * steps
    received = {r["sha256"] for r in policy.received}
    for i, (real, cf) in enumerate(zip(reqs[0::2], reqs[1::2])):
        key = next(iter(real["images"]))
        assert fingerprint(cf) in received
        same = np.array_equal(cf["images"][key], real["images"][key])
        assert same == (i < 2)                                             # stub masks from step 2
        assert all(fingerprint(cf[k]) == fingerprint(real[k]) for k in real if k != "images")


def test_null_runs_everything_and_executes_native():
    (base,), _ = run_pcd(Arm.BASELINE)
    (null,), policy = run_pcd(Arm.NULL)
    assert null["valid"]
    assert null["action_sha256"] == base["action_sha256"]
    acc = null["accounting"]
    assert acc["shadow_policy_samples"] == acc["method_policy_samples"] == 2 * acc["steps"]
    assert acc["native_policy_samples"] >= 1 and acc["tool_calls"] == acc["steps"] + 1
    assert acc["extra"]["pcd"]["steps"] == acc["steps"] and len(events(null, "pcd")) == acc["steps"]
    assert len(policy.logits_requests) == 2 * acc["steps"]


def test_candidate_is_deterministic():
    (a,), _ = run_pcd(Arm.CANDIDATE)
    (b,), _ = run_pcd(Arm.CANDIDATE)
    assert a["action_sha256"] == b["action_sha256"]


@pytest.mark.parametrize("truncate,field", [(("B", 6), "contrast_skipped"), (("B", 3), "contrast_skipped"),
                                            (("A", 5), "a_fallback")])
def test_fallbacks_end_to_end(truncate, field):
    (rec,), _ = run_pcd(Arm.CANDIDATE, truncate=truncate)
    assert rec["valid"], rec.get("invalid_reason")
    pcd = rec["accounting"]["extra"]["pcd"]
    assert pcd[field] == pcd["steps"] > 0
    if field == "a_fallback":
        assert all(e["tokens"][6] == 0 for e in events(rec, "pcd"))


# ---- the HTTP client and the real-service code path (mock of the service API) ----------------------------------
class MockService:
    def __init__(self, fail_step=None, reset_reply=None):
        self.calls, self.fail_step, self.steps = [], fail_step, 0
        self.payloads, self.reset_reply = [], reset_reply
        svc = self

        class H(BaseHTTPRequestHandler):
            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                self._send(200, {"ok": True, "version": "mock", "device": "cpu", "weights": {}, "sam2_C": False})

            def do_POST(self):  # noqa: N802
                p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                svc.calls.append((self.path, p.get("session"), p.get("instruction")))
                svc.payloads.append((self.path, {k: v for k, v in p.items() if k != "image_npy"}))
                if self.path == "/reset" and svc.reset_reply is not None:
                    return self._send(200, svc.reset_reply(p))
                if self.path == "/step":
                    if svc.fail_step is not None and svc.steps == svc.fail_step:
                        return self._send(500, {"error": "mock failure"})
                    img = np.load(io.BytesIO(base64.b64decode(p["image_npy"])), allow_pickle=False)
                    cf = (255 - img).astype(np.uint8)
                    buf = io.BytesIO()
                    np.save(buf, cf, allow_pickle=False)
                    svc.steps += 1
                    return self._send(200, {"cf_npy": base64.b64encode(buf.getvalue()).decode(), "phase": "track",
                                            "per_target": {"coke can": {"mask_present": True, "area_fraction": 0.1,
                                                                        "tracker_none": False}},
                                            "cf_differs": True, "timing_ms": {"total_ms": 1.0},
                                            "episode": {"first_detection_step": 0}})
                return self._send(200, {"ok": True})

            def log_message(self, *a):
                pass
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def http_contract():
    return dataclasses.replace(load_contract("fake"), default_horizon=6, control_mode="fake-http")


def test_real_client_path_with_tiles(tmp_path, monkeypatch):
    svc = MockService()
    monkeypatch.setenv(C.URL_ENV, svc.url)
    monkeypatch.setenv(C.TILE_ENV, str(tmp_path))
    try:
        (rec,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), horizon=6)
    finally:
        svc.close()
    assert rec["valid"], rec.get("invalid_reason")
    pcd = rec["accounting"]["extra"]["pcd"]
    assert pcd["perception"] == "http" and pcd["tiles"] >= 1
    paths = [c[0] for c in svc.calls]
    assert paths[0] == "/reset" and paths[-1] == "/close" and paths.count("/step") == rec["accounting"]["steps"]
    sessions = {c[1] for c in svc.calls}
    assert len(sessions) == 1 and next(iter(sessions)).startswith("pcd-")   # method-generated, opaque
    assert {c[2] for c in svc.calls if c[0] != "/close"} == {"pick coke can"}
    tiles = list(Path(tmp_path, pcd["session"]).glob("step*_real-cf.png"))
    assert len(tiles) == pcd["tiles"]


def test_perception_failure_invalidates_the_episode(tmp_path, monkeypatch):
    svc = MockService(fail_step=2)
    monkeypatch.setenv(C.URL_ENV, svc.url)
    monkeypatch.setenv(C.TILE_ENV, str(tmp_path))
    try:
        (rec,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), horizon=6)
    finally:
        svc.close()
    assert not rec["valid"]


def test_real_runs_need_a_tile_dir_outside_harnessstudio(monkeypatch):
    svc = MockService()
    monkeypatch.setenv(C.URL_ENV, svc.url)
    monkeypatch.delenv(C.TILE_ENV, raising=False)
    try:
        (rec,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), horizon=6)
        assert not rec["valid"]
        monkeypatch.setenv(C.TILE_ENV, str(Path(C.__file__).resolve().parents[1] / "tiles-must-not-go-here"))
        (rec2,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), horizon=6)
        assert not rec2["valid"]
        assert not (Path(C.__file__).resolve().parents[1] / "tiles-must-not-go-here").exists()
    finally:
        svc.close()


def test_params_are_validated():
    with pytest.raises(ValueError):
        C.PCDController(parser="method")
    with pytest.raises(ValueError):
        C.PCDController(instruction_source="llm")


# ---- Addendum 2 E-4 (review M1): any-target activation, multi-target safe -------------------------------------
def test_any_target_present_union_over_targets():
    ev = {"per_target": {"apple": {"mask_present": False, "area_fraction": 0.0},
                         "top drawer": {"mask_present": True, "area_fraction": 0.02}}}
    assert C.any_target_present(ev)
    assert not C.any_target_present({"per_target": {"apple": {"area_fraction": 0.0}, "plate": {"area_fraction": 0.0}}})
    assert not C.any_target_present({"per_target": {"apple": {"mask_present": True, "area_fraction": 0.0}}})
    assert not C.any_target_present({"per_target": {}}) and not C.any_target_present({})


@pytest.mark.parametrize("active,steps,want", [(0, 0, None), (0, 10, True), (4, 10, True), (5, 10, False),
                                               (10, 10, False), (2, 5, True), (3, 5, False)])
def test_mechanism_inactive_threshold(active, steps, want):
    assert C.mechanism_inactive(active, steps) is want


def test_stack_activation_summary_and_flags():
    (rec,), _ = run_pcd(Arm.CANDIDATE)
    pcd = rec["accounting"]["extra"]["pcd"]
    ev = events(rec, "pcd")
    flags = [e["any_target_present"] for e in ev]
    assert flags == [i >= 2 for i in range(len(ev))]                 # stub masks from step 2
    assert pcd["target_active_steps"] == sum(flags)
    assert pcd["mechanism_inactive"] is (sum(flags) < 0.5 * len(ev))


class MultiTargetService(MockService):
    """Two targets present on alternating steps: per-target counts are each ~50%, the union is 100%."""

    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern
        orig = self.httpd.RequestHandlerClass.do_POST
        svc = self

        def do_post(handler):
            if handler.path != "/step":
                return orig(handler)
            n = int(handler.headers["Content-Length"])
            p = json.loads(handler.rfile.read(n))
            img = np.load(io.BytesIO(base64.b64decode(p["image_npy"])), allow_pickle=False)
            buf = io.BytesIO()
            np.save(buf, (255 - img).astype(np.uint8), allow_pickle=False)
            a, b = svc.pattern(svc.steps)
            svc.steps += 1
            svc.calls.append(("/step", p.get("session"), p.get("instruction")))
            handler._send(200, {"cf_npy": base64.b64encode(buf.getvalue()).decode(), "phase": "track",
                                "per_target": {"apple": {"mask_present": a > 0, "area_fraction": a, "tracker_none": a == 0},
                                               "top drawer": {"mask_present": b > 0, "area_fraction": b,
                                                              "tracker_none": b == 0}},
                                "cf_differs": True, "timing_ms": {"total_ms": 1.0}, "episode": {}})
        self.httpd.RequestHandlerClass.do_POST = do_post


@pytest.mark.parametrize("pattern,inactive", [(lambda i: (0.01, 0.0) if i % 2 else (0.0, 0.02), False),
                                              (lambda i: (0.0, 0.0) if i < 4 else (0.01, 0.0), True)])
def test_multi_target_union_drives_mechanism_inactive(tmp_path, monkeypatch, pattern, inactive):
    svc = MultiTargetService(pattern)
    monkeypatch.setenv(C.URL_ENV, svc.url)
    monkeypatch.setenv(C.TILE_ENV, str(tmp_path))
    try:
        (rec,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), horizon=6)
    finally:
        svc.close()
    assert rec["valid"], rec.get("invalid_reason")
    pcd = rec["accounting"]["extra"]["pcd"]
    steps = pcd["steps"]
    assert pcd["mechanism_inactive"] is inactive
    if not inactive:
        assert pcd["target_active_steps"] == steps                   # union, although each target is ~50%
        assert all(v < steps for v in pcd["mask_present_steps"].values())


# ---- SETTING.md Addendum 4: perception parser mode passed at /reset (service field `parser`) -------------------
def test_parser_param_default_and_validation():
    assert hs_pcd.config().controller.params["perception_parser"] == "simpler"
    assert C.PCDController().perception_parser == "simpler"
    b = hs_pcd.config_libero_goal()
    assert b.name == "pcd-libero-goal" and b.controller.params["perception_parser"] == "libero_goal"
    assert {k: v for k, v in b.controller.params.items() if k != "perception_parser"} == \
        {k: v for k, v in hs_pcd.config().controller.params.items() if k != "perception_parser"}
    assert tier0.check(b) == []
    with pytest.raises(ValueError):
        C.PCDController(perception_parser="libero_10")
    with pytest.raises(ValueError):
        C.PCDController(perception_parser="")


def _echo(p):
    return {"started": True, "parser": p.get("parser"), "parser_sha256": "f" * 64 if p.get("parser") == "libero_goal" else None}


@pytest.mark.parametrize("params,want", [(None, "simpler"), ({"perception_parser": "libero_goal"}, "libero_goal")])
def test_parser_is_sent_at_reset(tmp_path, monkeypatch, params, want):
    svc = MockService(reset_reply=_echo)
    monkeypatch.setenv(C.URL_ENV, svc.url)
    monkeypatch.setenv(C.TILE_ENV, str(tmp_path))
    try:
        (rec,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), params=params, horizon=6)
    finally:
        svc.close()
    assert rec["valid"], rec.get("invalid_reason")
    resets = [p for path, p in svc.payloads if path == "/reset"]
    assert len(resets) == 1 and resets[0]["parser"] == want
    assert all("parser" not in p for path, p in svc.payloads if path == "/step")
    pcd = rec["accounting"]["extra"]["pcd"]
    assert pcd["perception_parser"] == want and pcd["service_parser"] == want


@pytest.mark.parametrize("params,reply", [({"perception_parser": "libero_goal"}, lambda p: {"started": True}),
                                          ({"perception_parser": "libero_goal"}, lambda p: {"parser": "simpler"}),
                                          (None, lambda p: {"parser": "libero_goal"})])
def test_parser_echo_mismatch_invalidates(tmp_path, monkeypatch, params, reply):
    svc = MockService(reset_reply=reply)
    monkeypatch.setenv(C.URL_ENV, svc.url)
    monkeypatch.setenv(C.TILE_ENV, str(tmp_path))
    try:
        (rec,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), params=params, horizon=6)
    finally:
        svc.close()
    assert not rec["valid"]
    assert not any(path == "/step" for path, _ in svc.payloads)


def test_stub_receives_the_default_parser():
    (rec,), _ = run_pcd(Arm.CANDIDATE)
    pcd = rec["accounting"]["extra"]["pcd"]
    assert pcd["perception_parser"] == "simpler" and pcd["service_parser"] == "simpler"


# ---- conv-fixes-6: a refused episode closes its perception session and still writes extra.pcd ------------------
@pytest.mark.parametrize("params,reply,needle", [
    ({"perception_parser": "libero_goal"}, lambda p: {"started": True}, "did not confirm parser"),
    ({"perception_parser": "libero_goal"}, lambda p: {"parser": "simpler"}, "!= requested"),
    (None, lambda p: {"parser": "libero_goal"}, "!= requested")])
def test_refused_episode_closes_session_and_records(tmp_path, monkeypatch, params, reply, needle):
    svc = MockService(reset_reply=reply)
    monkeypatch.setenv(C.URL_ENV, svc.url)
    monkeypatch.setenv(C.TILE_ENV, str(tmp_path))
    try:
        (rec,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), params=params, horizon=6)
    finally:
        svc.close()
    assert not rec["valid"] and needle in (rec["invalid_reason"] or "")
    paths = [path for path, _ in svc.payloads]
    assert paths == ["/reset", "/close"]                                    # no /step; the session is closed
    sessions = {p.get("session") for _, p in svc.payloads}
    assert len(sessions) == 1
    pcd = rec["accounting"]["extra"]["pcd"]
    assert pcd["steps"] == 0 and needle in pcd["error"] and pcd["session"] in sessions
    assert pcd["perception_parser"] == (params or {}).get("perception_parser", "simpler")
    assert pcd["service_parser"] == reply({}).get("parser")
    assert not events(rec, "pcd_start") and not events(rec, "pcd")


def test_tile_dir_refusal_also_records(monkeypatch):
    svc = MockService()
    monkeypatch.setenv(C.URL_ENV, svc.url)
    monkeypatch.delenv(C.TILE_ENV, raising=False)
    try:
        (rec,), _ = run_pcd(Arm.CANDIDATE, contract=http_contract(), horizon=6)
    finally:
        svc.close()
    assert not rec["valid"]
    pcd = rec["accounting"]["extra"]["pcd"]
    assert "tile directory" in pcd["error"] and pcd["steps"] == 0
    assert [path for path, _ in svc.payloads] == ["/close"]                 # nothing was opened on the service


def test_successful_episode_has_no_error():
    (rec,), _ = run_pcd(Arm.CANDIDATE)
    assert rec["valid"] and rec["accounting"]["extra"]["pcd"]["error"] is None
