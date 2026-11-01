"""Tests for the evaluation-side scripts: summarize (stopping rule, statistics), check_split freeze checks,
the encoder health pin, and the LOO script's Long-only guard."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(run: Path, arm: str, seed: int, tasks: int, eps: int, succ):
    d = run / arm / "traces"
    d.mkdir(parents=True, exist_ok=True)
    for t in range(tasks):
        for e in range(eps):
            ev = [{"t": "decision", "diagnostics": {"action_op": {"mode": "eval", "applied": True, "i": 0, "b": b,
                   "enc_ms": 10.0 + b, "pma_ms": 1.0, "op_ms": 12.0, "clipped_frac": 0.1}}} for b in range(3)]
            rec = {"benchmark": f"libero@seed{seed}", "task": f"libero_10/t{t}#{t}", "episode_idx": e, "valid": True,
                   "invalid_reason": None, "success": bool(succ(t, e)), "events": ev if arm != "baseline" else [],
                   "action_sha256": [arm]}
            (d / f"{t}_{e}.json").write_text(json.dumps(rec))


def test_summarize_pools_only_complete_seeds(tmp_path):
    for arm, f in (("baseline", lambda t, e: e < 5), ("candidate", lambda t, e: e < 6)):
        _write(tmp_path / "s101", arm, 101, 10, 10, f)
    _write(tmp_path / "s102", "baseline", 102, 10, 10, lambda t, e: True)
    _write(tmp_path / "s102", "candidate", 102, 3, 10, lambda t, e: True)          # partial seed
    meta = tmp_path / "memory.json"
    meta.write_text(json.dumps({"memories": [{"task": "libero_10/t0#0"}]}))
    out = tmp_path / "s.json"
    runs = f"{tmp_path / 's101'},{tmp_path / 's102'}"
    subprocess.run([sys.executable, str(SCRIPTS / "summarize.py"), "--arm", f"baseline={runs}",
                    "--arm", f"candidate={runs}", "--memory-meta", str(meta), "--out", str(out)],
                   check=True, capture_output=True)
    s = json.loads(out.read_text())
    assert s["headline_seeds"] == ["seed101"] and s["partial_seeds"] == ["seed102"]
    assert s["arms"]["baseline"]["episodes"] == 100 and s["arms"]["baseline"]["successes"] == 50
    assert s["arms"]["candidate"]["successes"] == 60
    p = s["pairs"]["candidate_vs_baseline"]
    assert (p["paired"], p["wins"], p["losses"]) == (100, 10, 0)
    assert s["seeds"]["seed102"]["status"] == "partial" and s["seeds"]["seed102"]["pair"]["paired"] == 30
    c = s["arms"]["candidate"]
    assert c["latency_ms"]["encoder"]["median"] == 11.0 and c["encoder_calls_per_logged_decision"] == 1.0
    assert c["b_star_advance_distribution"] == {"1": 200}
    assert 0.0 < c["same_task_retrieval_fraction_posthoc"] < 1.0      # only task 0 matches memory 0
    assert s["vs_paper"]["baseline"]["paper"] == 61.6 and s["vs_paper"]["baseline"]["ours_n"] == 100


def test_clustered_single_cluster_is_percent():
    mod = _load("summarize")
    assert mod.clustered({("a", 0): [1, 1, 0, 0]})["mean"] == 50.0


def test_check_split_freeze(tmp_path):
    mod = _load("check_split")
    mem = tmp_path / "memory.npz"
    mem.write_bytes(b"abc")
    run = tmp_path / "run"
    (run / "candidate").mkdir(parents=True)
    (run / "candidate" / "manifest.json").write_text(json.dumps(
        {"arm": "candidate", "method_fingerprint": "f1", "artifacts": {"memory": str(mem)}}))
    import hashlib
    ok, bad = [], []
    mod.check_freeze(run, "f1", hashlib.sha256(b"abc").hexdigest(), ok)
    mod.check_freeze(run, "f2", "0" * 64, bad)
    assert not ok and len(bad) == 2


def test_health_pin(monkeypatch):
    from hs_rtcf import encoder as enc
    client = enc.SigLIPClient("http://127.0.0.1:1", "m", "rev", "p", weights_sha256="w" * 64)

    class Resp:
        def __init__(self, body):
            self.body = body
        def read(self):
            return json.dumps(self.body).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(enc.urllib.request, "urlopen", lambda *a, **k: Resp({"weights_sha256": "x", "revision": "rev"}))
    with pytest.raises(RuntimeError):
        client.check_health()
    monkeypatch.setattr(enc.urllib.request, "urlopen", lambda *a, **k: Resp({"weights_sha256": "w" * 64, "revision": "rev"}))
    client.check_health()
    assert ("http://127.0.0.1:1", "w" * 64) in enc._HEALTH_OK


def test_loo_refuses_non_long_memory(tmp_path):
    from hs_rtcf.memory import MemoryBank
    d = np.eye(4, dtype=np.float32)
    MemoryBank(d, np.array([0, 2, 4]), np.zeros((4, 10, 7), np.int64), np.zeros(4, bool),
               {"memories": [{"task": "libero_10/a#0"}, {"task": "libero_spatial/b#1"}]}).save(
        tmp_path / "memory.npz", tmp_path / "memory.json")
    r = subprocess.run([sys.executable, str(SCRIPTS / "select_pma_params.py"), str(tmp_path), "--out",
                        str(tmp_path / "o.json")], capture_output=True, text=True)
    assert r.returncode != 0 and "non-LIBERO-Long" in (r.stderr + r.stdout)
