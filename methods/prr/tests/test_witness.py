"""Evaluator hooks: Q-recur scoring on synthetic traces, outcome-blind loader, determinism witness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import witness  # noqa: E402
from hs_prr.geometry import exp, log  # noqa: E402

BATCH = ["t2i34", "t3i32", "t3i34", "t5i32", "t8i30", "t9i31"]


def state(tilt):
    aa = log(exp([np.radians(tilt), 0, 0]) @ exp([np.pi, 0, 0]))
    return [0.0, 0.0, 1.0, *map(float, aa), 0.04, -0.04]


def write(dirp: Path, key: str, recur: bool, w1_ok: bool, success: bool):
    t, i = key[1:].split("i")
    ev = [{"t": "event", "name": "prr_restore_start", "step": 100, "route": "A", "fire": 100, "checkpoint": 70, "L": 30,
           "latch": -1.0},
          {"t": "event", "name": "prr_handback", "step": 140, "pos_err_m": 0.01 if w1_ok else 0.05, "rot_err_deg": 3.0}]
    for d in range(140, 205, 5):
        ev.append({"t": "event", "name": "prr_d", "step": d, "state": state(60.0 if (recur and d == 190) else 20.0)})
    rec = {"task": f"libero_10/x#{t}", "episode_idx": int(i), "valid": True, "success": success,
           "metrics": {"success": success}, "action_sha256": ["a"] * 200, "events": ev}
    (dirp / "traces").mkdir(parents=True, exist_ok=True)
    (dirp / "traces" / f"libero__x__{i}_{t}.json").write_text(json.dumps(rec))


@pytest.mark.parametrize("n_recur,decision", [(4, "STOP_family_closed"), (3, "continue")])
def test_q_recur(tmp_path, n_recur, decision):
    d = tmp_path / "prr"
    for j, k in enumerate(BATCH):
        write(d, k, recur=j < n_recur, w1_ok=j != 0, success=True)
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"lists": {"q_recur_batch_C6": BATCH},
                                 "first_restorations_dev80": {k: {"route": "A", "fire": 100, "checkpoint": 70, "L": 30,
                                                                  "latch": -1.0} for k in BATCH}}))
    out = tmp_path / "q.json"
    witness.main(["q-recur", "--prr", str(d), "--table", str(table), "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["complete"] and res["recur_count"] == n_recur and res["decision"] == decision
    assert res["w1"]["within"] == 5 and res["w1"]["of"] == 6 and res["w1"]["pass"] is True
    assert all(r["witness"]["matches_table"] for r in res["rows"])
    assert "success" not in out.read_text()                    # no outcome value ever read


def test_loader_refuses_outcome_named_files(tmp_path):
    for bad in ("traces/x_success.json", "vla_eval/episodes/a/traces/b.json", "traces/records.jsonl"):
        with pytest.raises(ValueError):
            witness.load_trace(str(tmp_path / bad))
