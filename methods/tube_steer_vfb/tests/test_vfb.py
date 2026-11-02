"""tube-steer-vfb specific tests: visual memory provenance and pinning, visual retrieval branch (injected descriptors),
fail-loud encoder, live encoder health on the pinned service, TV == T bitwise for instructions with a tube."""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from harnessstudio.core import registry
from hs_tube_steer_vfb import action_op as A
from test_tube_steer import chunk, state

T8 = "put both moka pots on the stove"
INSTR = {0: "put both the alphabet soup and the tomato sauce in the basket", 2: "turn on the stove and put the moka pot on it",
         3: "put the black bowl in the bottom drawer of the cabinet and close it", 5: "pick up the book and place it in the back compartment of the caddy",
         9: "put the yellow and white mug in the microwave and close it"}


def tv(mode="steer"):
    return registry.load("tube-steer-vfb-ortho" if mode == "ortho" else "tube-steer-vfb").slots["action_op"].build()


def test_visual_memory_provenance_and_pin():
    op = tv()
    vm = op.visual
    assert vm["desc"].shape == (6176, 1152) and len(vm["z"]) == 6176 and 8 not in set(vm["task"].tolist())
    keys = sorted(set(vm["key"]))
    assert len(keys) == 127 and all(10 <= int(k.split("_i")[1]) <= 29 for k in keys)
    tm = json.load(open(A.PKG / "assets/tube_memory.json"))
    assert sorted(e["key"] for e in tm["episodes"]) == keys
    p = registry.load("tube-steer-vfb").card["params"]
    assert p["visual_memory_npz_sha256"] == "ccebf9e0d3d80c709c116101835eea6c199fdac2a87fe7746cc2c98c732209a5"
    assert p["visual_k"] == 8 and p["encoder_revision"].startswith("d04cf29f") and p["encoder_weights_sha256"].startswith("30a00c26")
    A.VisualMemory._cache.clear()
    with pytest.raises(RuntimeError):
        A.TubeSteerActionOp(**{**p, "visual_memory_npz_sha256": "0" * 64})


def test_visual_branch_retrieval_and_arming_with_injected_descriptor():
    op = tv()
    vm = op.visual
    rows = np.where(vm["task"] == 2)[0]
    r0 = int(rows[40])
    op.descriptor = lambda img: vm["desc"][r0]                    # injected (the replay does the same with recorded descriptors)
    img = np.zeros((256, 256, 3), np.uint8)
    rng = np.random.default_rng(0)
    outs = []
    for tilt in (3.0, 80.0, 80.0):
        c = chunk(rng)
        outs.append((c, op.transform(c, {"states": state(tilt), "task_description": T8, "images": {"agentview": img}}, {}),
                     op.diagnostics()))
    d = outs[0][2]
    assert d["tube_source"] == "visual" and d["visual"]["top_tasks"][0] == 2 and len(d["visual"]["top_tasks"]) == 8
    sim = vm["desc"] @ vm["desc"][r0]
    top = np.argsort(-sim, kind="stable")[:8]
    z = A.zaxis(state(80.0)[3:6])
    exp = float(np.degrees(np.arccos(np.clip(vm["z"][top] @ z, -1, 1))).min())
    assert abs(outs[1][2]["D"] - exp) < 1e-9
    assert outs[1][2]["armed"] == (outs[0][2]["D"] > 15 and exp > 15)
    if outs[2][2]["armed"]:
        assert outs[2][2]["steer"] and outs[2][1] is not outs[2][0]


def test_encoder_unavailable_is_infrastructure_failure():
    p = registry.load("tube-steer-vfb").card["params"]
    op = A.TubeSteerActionOp(**{**p, "encoder_url": "http://127.0.0.1:9", "encoder_timeout": 2.0})
    img = np.zeros((256, 256, 3), np.uint8)
    with pytest.raises(Exception):
        op.transform(chunk(np.random.default_rng(1)), {"states": state(30.0), "task_description": T8,
                                                         "images": {"agentview": img}}, {})


def _service_up():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8149/health", timeout=3) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _service_up(), reason="pinned SigLIP service :8149 not reachable")
def test_live_encoder_health_and_wrong_pin():
    op = tv()
    img = (np.arange(256 * 256 * 3) % 251).astype(np.uint8).reshape(256, 256, 3)
    out = op.transform(chunk(np.random.default_rng(2)), {"states": state(30.0), "task_description": T8,
                                                         "images": {"agentview": img}}, {})
    d = op.diagnostics()
    assert op.encoder.healthy and d["visual"]["encoder_calls"] == 1 and out is not None
    p = registry.load("tube-steer-vfb").card["params"]
    bad = A.TubeSteerActionOp(**{**p, "encoder_weights_sha256": "0" * 64})
    with pytest.raises(RuntimeError):
        bad.transform(chunk(np.random.default_rng(3)), {"states": state(30.0), "task_description": T8,
                                                       "images": {"agentview": img}}, {})


def test_tv_equals_t_bitwise_with_instruction_tube():
    for mode, (te, ve) in (("steer", ("tube-steer", "tube-steer-vfb")), ("ortho", ("tube-steer-ortho", "tube-steer-vfb-ortho"))):
        rng = np.random.default_rng(7)
        for task, instr in INSTR.items():
            opT, opV = registry.load(te).slots["action_op"].build(), registry.load(ve).slots["action_op"].build()
            tilt = 3.0
            for q in range(70):
                tilt = float(np.clip(tilt + rng.normal(0, 6), 0, 170))
                s, c = state(tilt, float(rng.uniform(0, 360))), chunk(rng)
                a = opT.transform(np.array(c), {"states": s, "task_description": instr}, {})
                b = opV.transform(np.array(c), {"states": s, "task_description": instr, "images": {}}, {})
                assert np.array_equal(np.asarray(a), np.asarray(b))
            assert opT.stats == opV.stats
