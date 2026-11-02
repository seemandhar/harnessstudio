"""tube-steer-vfb r2 tests: visual-row provenance against the S2 index and the recorded S2 descriptors (streamed),
per-episode encoder health check with a stub service (pinned fields, pinned files, unavailable), visual-branch
passthrough identity and the visual ortho cap, and a tampered-provenance memory refusing to load."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from harnessstudio.core import registry
from hs_tube_steer_vfb import action_op as A
from test_tube_steer import chunk, state

RUN = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtcfpp-20261101"
PREP = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtcf-20261031/eval/s2/PREP"
T8 = "put both moka pots on the stove"
IMG = np.zeros((256, 256, 3), np.uint8)


def params():
    return dict(registry.load("tube-steer-vfb").card["params"])


def test_visual_rows_are_s2_successes_init_10_29_and_match_recorded_descriptors():
    prep = json.load(open(PREP + "/prepared.json"))
    pins = {a["name"]: a["sha256"] for a in prep["artifacts"]}
    p = params()
    assert pins["memory"] == p["visual_memory_npz_sha256"] and pins["memory_meta"] == p["visual_memory_json_sha256"]
    for name, sha in (("visual_memory.npz", pins["memory"]), ("visual_memory.json", pins["memory_meta"])):
        assert hashlib.sha256((A.PKG / "assets" / name).read_bytes()).hexdigest() == sha
    idx = {e["key"]: e for e in json.load(open(RUN + "/analysis/s2_index.json"))}
    vmeta = json.load(open(A.PKG / "assets/visual_memory.json"))
    vz = np.load(A.PKG / "assets/visual_memory.npz")
    off = vz["offsets"]
    s2 = np.load(RUN + "/ideation/scripts-r5-memory/s2_desc.npz")
    inits = set()
    for i, e in enumerate(vmeta["memories"]):
        t = int(e["task"].rsplit("#", 1)[1])
        k = f"t{t}_i{int(e['episode_idx'])}"
        s = idx[k]
        assert s["success"] is True and s["valid"] is True and 10 <= int(s["init"]) <= 29 and t != 8
        assert s["task_name"] == e["task"]
        inits.add(int(s["init"]))
        rows = vz["descriptors"][off[i]:off[i + 1]]
        rec = s2[k + "_desc"]
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        assert rows.shape == rec.shape and np.allclose(rows, rec, atol=1e-6), k
    assert len(vmeta["memories"]) == 127 == sum(e["success"] for e in idx.values())
    assert inits <= set(range(10, 30))


def test_tampered_provenance_refuses(tmp_path):
    vm = json.load(open(A.PKG / "assets/visual_memory.json"))
    vm["memories"][0]["episode_idx"] = 35
    bad = tmp_path / "visual_memory.json"
    bad.write_text(json.dumps(vm))
    npz = tmp_path / "visual_memory.npz"
    shutil.copy(A.PKG / "assets/visual_memory.npz", npz)
    p = params()
    p.update(visual_memory_json=str(bad), visual_memory_json_sha256=hashlib.sha256(bad.read_bytes()).hexdigest(),
             visual_memory_npz=str(npz))
    with pytest.raises(RuntimeError, match="init 10-29"):
        A.TubeSteerActionOp(**p)


class _Stub(BaseHTTPRequestHandler):
    health: dict = {}
    hits = {"health": 0, "embed": 0}

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        _Stub.hits["health"] += 1
        self._send(_Stub.health)

    def do_POST(self):
        _Stub.hits["embed"] += 1
        self.rfile.read(int(self.headers["Content-Length"]))
        import base64
        v = np.zeros(1152, "<f4")
        v[0] = 1.0
        h = _Stub.health
        self._send({"embedding_f32_b64": base64.b64encode(v.tobytes()).decode(), "model": h.get("model"),
                    "revision": h.get("revision"), "pooling": h.get("pooling")})


@pytest.fixture()
def stub():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    p = params()
    _Stub.health = {"status": "ok", "model": p["encoder_model"], "revision": p["encoder_revision"],
                    "pooling": p["encoder_pooling"], "weights_sha256": p["encoder_weights_sha256"],
                    "pinned_files": dict(p["encoder_pinned_files"])}
    _Stub.hits = {"health": 0, "embed": 0}
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _run(url, n=3, **over):
    op = A.TubeSteerActionOp(**{**params(), "encoder_url": url, **over})
    for _ in range(n):
        op.transform(chunk(np.random.default_rng(0)), {"states": state(30.0), "task_description": T8,
                                                       "images": {"agentview": IMG}}, {})
    return op


def test_health_checked_once_per_episode_instance(stub):
    _run(stub)
    assert _Stub.hits == {"health": 1, "embed": 3}
    op = _run(stub)                                   # a new episode builds a new op -> a new health check
    assert _Stub.hits == {"health": 2, "embed": 6} and op.diagnostics()["visual"]["encoder_health_checks"] == 1


@pytest.mark.parametrize("field,value", [("weights_sha256", "0" * 64), ("revision", "main"), ("pooling", "mean"),
                                         ("model", "other"), ("pinned_files", {"config.json": "x"})])
def test_health_mismatch_raises(stub, field, value):
    _Stub.health[field] = value
    with pytest.raises(RuntimeError):
        _run(stub, n=1)
    assert _Stub.hits["embed"] == 0


def test_unavailable_encoder_raises_not_abstains():
    with pytest.raises(Exception):
        _run("http://127.0.0.1:9", n=1, encoder_timeout=2.0)


def test_visual_branch_passthrough_identity_and_ortho_cap():
    for entry, cap in (("tube-steer-vfb", None), ("tube-steer-vfb-ortho", 6)):
        op = registry.load(entry).slots["action_op"].build()
        vm = op.visual
        r0 = int(np.where(vm["task"] == 2)[0][40])
        op.descriptor = lambda img, _r=r0: vm["desc"][_r]
        sim = vm["desc"] @ vm["desc"][r0]
        top = np.argsort(-sim, kind="stable")[:8]
        zt = vm["z"][top]
        rng = np.random.default_rng(5)
        steered = 0
        for q in range(40):
            tilt = 3.0 if q < 3 else 120.0
            c = chunk(rng)
            out = op.transform(c, {"states": state(tilt), "task_description": T8, "images": {"agentview": IMG}}, {})
            d = op.diagnostics()
            assert d["tube_source"] == "visual"
            exp = float(np.degrees(np.arccos(np.clip(zt @ A.zaxis(state(tilt)[3:6]), -1, 1))).min())
            assert abs(d["D"] - exp) < 1e-9
            if d.get("steer"):
                steered += 1
                assert out is not c and np.array_equal(out[:, [0, 1, 2, 6]], c[:, [0, 1, 2, 6]])
                assert np.array_equal(out[5:], c[5:])
            else:
                assert out is c
        assert steered > 0 and op.stats["visual_decisions"] == 40
        if cap:
            assert steered <= cap and op.stats["arm_events"] == 1
