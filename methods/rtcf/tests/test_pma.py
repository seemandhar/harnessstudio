"""Unit tests for Progressive Memory Alignment (Eq. 5-9), the memory bank, the Preparer and the action_op."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from hs_rtcf import pma, spectral
from hs_rtcf.action_op import RTCFActionOp
from hs_rtcf.encoder import f32_b64, tier1_pixel_descriptor, Encoder
from hs_rtcf.memory import MemoryBank
from hs_rtcf.prepare import Preparer, decision_rows


def unit_rows(rng, n, d):
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def brute_frontier(query, mem, v_max, gamma):
    """Plain loops over Eq. 5-8 for one memory (positions 0-based); recomputed from scratch."""
    L = len(mem)
    J = [1 - np.clip(mem[q] @ query[0], -1, 1) for q in range(L)]
    for t in range(1, len(query)):
        new = []
        for q in range(L):
            c = 1 - np.clip(mem[q] @ query[t], -1, 1)
            best = np.inf
            for delta in range(v_max + 1):
                if q - delta >= 0:
                    best = min(best, J[q - delta] + gamma * abs(delta - 1))
            new.append(c + best)
        J = new
    J = np.array(J)
    return J, int(np.argmin(J)), float(J.min() / len(query))


@pytest.mark.parametrize("seed", range(30))
def test_incremental_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 9))
    lengths = rng.integers(1, 12, int(rng.integers(1, 6)))
    mems = [unit_rows(rng, int(L), d) for L in lengths]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    v_max, gamma = int(rng.integers(0, 4)), float(rng.choice([0.0, 0.05, 0.3]))
    obj = pma.ProgressiveMemoryAlignment(np.concatenate(mems), offsets, v_max, gamma)
    query = unit_rows(rng, int(rng.integers(1, 15)), d)
    for t in range(len(query)):
        i, b, dstar = obj.update(query[t])
        ref = [brute_frontier(query[: t + 1], m, v_max, gamma) for m in mems]
        scores = np.array([r[2] for r in ref])
        assert i == int(np.argmin(scores))
        assert b == ref[i][1]
        assert abs(dstar - scores[i]) < 1e-12
        for k, m in enumerate(mems):
            assert np.allclose(obj.frontier[offsets[k]:offsets[k + 1]], ref[k][0], atol=1e-12)


def test_init_is_open_begin():
    z = np.eye(3)
    obj = pma.ProgressiveMemoryAlignment(z, np.array([0, 3]), 2, 0.05)
    i, b, d = obj.update(z[2])
    assert (i, b) == (0, 2) and abs(d) < 1e-12


def test_v_max_zero_never_advances():
    """v_max = 0: only 'stay' is allowed, so every position accumulates its own costs (one fixed hypothesis each)."""
    z = np.eye(4)
    obj = pma.ProgressiveMemoryAlignment(z, np.array([0, 4]), v_max=0, gamma=0.0)
    query = [z[1], z[2], z[3], z[3]]
    for q in query:
        i, b, d = obj.update(q)
    expect = np.array([sum(1 - z[p] @ q for q in query) for p in range(4)])
    assert np.allclose(obj.frontier, expect) and b == 3 and d == pytest.approx(expect[3] / 4)


def test_no_transition_across_memory_boundary():
    z = np.eye(4)
    # memory 0 = [e0, e1], memory 1 = [e2, e3]; the query e1 then e2 must not chain memory 0 -> memory 1
    obj = pma.ProgressiveMemoryAlignment(np.stack([z[0], z[1], z[2], z[3]]), np.array([0, 2, 4]), 2, 0.0)
    obj.update(z[1])
    obj.update(z[2])
    seg1 = obj.frontier[2:4]
    # memory 1 pos 0 at t=2: c(e2, e2) = 0 plus its only valid predecessor (stay at pos 0): c(e1, e2) = 1
    assert seg1[0] == pytest.approx(1.0)
    # memory 1 pos 1 at t=2: c(e2, e3) = 1 + min(stay J(pos1) = 1, advance from pos0 J = 1) = 2
    assert seg1[1] == pytest.approx(2.0)
    assert np.isfinite(obj.frontier).all()


def test_ties_go_to_lowest_index():
    z = np.eye(2)
    obj = pma.ProgressiveMemoryAlignment(np.stack([z[0], z[0], z[0]]), np.array([0, 1, 3]), 1, 0.0)
    i, b, _ = obj.update(z[0])
    assert (i, b) == (0, 0)


def test_cosine_clip():
    c = pma.local_cost(np.array([1.0, 0.0]), np.array([[1.0000001, 0.0], [-1.0000001, 0.0]]))
    assert np.allclose(c, [0.0, 2.0])


def test_frame_nn():
    z = np.eye(3)
    obj = pma.ProgressiveMemoryAlignment(np.stack([z[0], z[1], z[2]]), np.array([0, 1, 3]), retrieval="frame_nn")
    assert obj.update(z[2])[:2] == (1, 1)
    assert obj.update(z[0])[:2] == (0, 0)


# -- memory, preparer, action_op ---------------------------------------------------------------
class _Emb:
    def dims(self, *names):
        table = {"position": [0, 1, 2], "rotation": [3, 4, 5], "gripper": [6]}
        return [d for n in names for d in table[n]]


def fake_obs(level):
    return {"images": {"cam": np.full((8, 8, 3), level, np.uint8)}, "state": np.zeros(2, np.float32),
            "task_description": "x"}


def collect_record(rng, levels, success=True, task="t"):
    events = []
    for k, lv in enumerate(levels):
        ch = rng.uniform(-1, 1, (10, 7)).astype(np.float32)
        events.append({"t": "decision", "diagnostics": {"action_op": {
            "mode": "collect", "encoder": "tier1-pixels", "descriptor_f32": f32_b64(tier1_pixel_descriptor(fake_obs(lv)["images"]["cam"])),
            "chunk_f32": f32_b64(ch), "chunk_shape": [10, 7]}}})
    return {"benchmark": "fake", "task": task, "episode_idx": 0, "valid": True, "success": success, "events": events}


def test_preparer_builds_bank(tmp_path):
    rng = np.random.default_rng(0)
    recs = [collect_record(rng, [10, 60, 120]), collect_record(rng, [200, 220], success=False, task="u"),
            collect_record(rng, [30, 90], task="v")]
    names = Preparer().build(recs, tmp_path)
    bank = MemoryBank.load(tmp_path / names["memory"], tmp_path / names["memory_meta"])
    assert bank.num_memories == 2 and list(bank.offsets) == [0, 3, 5]
    meta = json.loads((tmp_path / "memory.json").read_text())
    assert meta["encoder"] == "tier1-pixels" and meta["num_positions"] == 5


def test_decision_rows_rejects_eval_diagnostics():
    rec = {"task": "t", "episode_idx": 0, "events": [{"t": "decision", "diagnostics": {"action_op": {"mode": "eval"}}}]}
    with pytest.raises(ValueError):
        decision_rows(rec)


def make_op(tmp_path=None, recs=None, **kw):
    op = RTCFActionOp(**kw)
    op.bind(_Emb())
    if recs is not None:
        names = Preparer().build(recs, tmp_path)
        op.load_artifacts({k: str(tmp_path / v) for k, v in names.items()})
    return op


def test_collection_mode_returns_native_and_records():
    op = make_op()
    ch = np.random.default_rng(1).uniform(-1, 1, (10, 7)).astype(np.float32)
    out = op.transform(ch, fake_obs(50), {})
    assert out is ch
    d = op.diagnostics()
    assert d["mode"] == "collect" and "descriptor_f32" in d and "png_b64" in d
    assert np.frombuffer(base64.b64decode(d["chunk_f32"]), "<f4").reshape(10, 7).tobytes() == ch.tobytes()


def test_eval_mode_corrects_and_skips(tmp_path):
    rng = np.random.default_rng(2)
    op = make_op(tmp_path, [collect_record(rng, [10, 60, 120])])
    ch = (0.3 * rng.uniform(-1, 1, (10, 7))).astype(np.float32)
    out = op.transform(ch, fake_obs(60), {})
    d = op.diagnostics()
    assert d["applied"] and d["i"] == 0
    assert out[:, 6].tobytes() == ch[:, 6].tobytes() and out.tobytes() != ch.tobytes()
    # read past the memory end -> native
    (tmp_path / "b").mkdir()
    op2 = make_op(tmp_path / "b", [collect_record(rng, [10])], read_offset=1)
    out2 = op2.transform(ch, fake_obs(10), {})
    assert out2.tobytes() == ch.tobytes() and op2.diagnostics()["skip"] == "past_memory_end"


def test_empty_bank_skips(tmp_path):
    rng = np.random.default_rng(3)
    op = make_op(tmp_path, [collect_record(rng, [10], success=False)])
    ch = rng.uniform(-1, 1, (10, 7)).astype(np.float32)
    assert op.transform(ch, fake_obs(10), {}).tobytes() == ch.tobytes()
    assert op.diagnostics()["skip"] == "empty_bank"


def test_decode_failed_memory_skips(tmp_path):
    from hs_rtcf import spectral as sp
    from hs_rtcf.action_op import resolve
    center = sp.norm_center(resolve("assets/pi0_fast_libero_norm_stats.json"), "zscore")
    state = np.array([0.05, -0.1, 1.0, 3.1, 0.2, -0.4, 0.03, -0.03], np.float32)
    rec = collect_record(np.random.default_rng(4), [10])
    zero = np.tile(sp.openpi_zero_chunk_row(state, center=center), (10, 1)).astype(np.float32)
    d = rec["events"][0]["diagnostics"]["action_op"]
    d["chunk_f32"], d["state_f32"] = f32_b64(zero), f32_b64(state)
    op = make_op(tmp_path, [rec])
    assert op.bank.decode_failed.tolist() == [True]
    ch = np.random.default_rng(5).uniform(-1, 1, (10, 7)).astype(np.float32)
    assert op.transform(ch, fake_obs(10), {}).tobytes() == ch.tobytes()
    assert op.diagnostics()["skip"] == "memory_decode_failed"


def test_unusable_memory_chunk_falls_back(tmp_path):
    rng = np.random.default_rng(8)
    op = make_op(tmp_path, [collect_record(rng, [10, 60])])
    ch = rng.uniform(-1, 1, (10, 7)).astype(np.float32)
    op._memory_coef = lambda i, q: np.full((10, 7), np.nan)        # non-finite aligned chunk
    assert op.transform(ch, fake_obs(10), {}).tobytes() == ch.tobytes()
    assert op.diagnostics()["skip"] == "memory_chunk_unusable"
    op._memory_coef = lambda i, q: np.zeros((8, 7))                # wrong shape
    assert op.transform(ch, fake_obs(60), {}).tobytes() == ch.tobytes()
    op._memory_coef = lambda i, q: None                            # missing
    assert op.transform(ch, fake_obs(60), {}).tobytes() == ch.tobytes()
    assert op.stats["skipped_unusable"] == 3


def test_local_cost_independent_of_row_position():
    rng = np.random.default_rng(9)
    z = unit_rows(rng, 1, 1152)[0]
    bank = unit_rows(rng, 3000, 1152).astype(np.float32)
    row = bank[17].copy()
    for pos in (0, 1, 999, 1023, 1024, 2999):
        b = bank.copy(); b[pos] = row
        assert pma.local_cost(z, b)[pos] == pma.local_cost(z, bank)[17]


def test_lambda_zero_op_is_native(tmp_path):
    rng = np.random.default_rng(6)
    op = make_op(tmp_path, [collect_record(rng, [10, 60, 120])], lambda0=0.0)
    for lv in (10, 60, 120, 200):
        ch = rng.uniform(-1, 1, (10, 7)).astype(np.float32)
        assert op.transform(ch, fake_obs(lv), {}).tobytes() == ch.tobytes()


def test_encoder_refuses_missing_image_on_real_obs():
    enc = Encoder("agentview", "http://127.0.0.1:1", "m", "r", "p")
    with pytest.raises(KeyError):
        enc.image({"images": {"wrist": np.zeros((256, 256, 3), np.uint8)}})
    with pytest.raises(KeyError):
        enc.image({"images": {"cam": np.zeros((16, 16, 3), np.uint8)}})


def test_decision_pre_events_preferred(tmp_path):
    """With the proposed decision-pre-diagnostics extension, the complete pre-execution rows are used."""
    rng = np.random.default_rng(7)
    rec = collect_record(rng, [10, 60])                       # 2 end-of-decision rows (last decision lost)
    pre = collect_record(rng, [10, 60, 120])["events"]        # 3 pre-execution rows
    for ev in pre:
        ev["t"] = "decision_pre"
    rec["events"] = rec["events"] + pre
    assert len(decision_rows(rec)) == 3
    names = Preparer().build([rec], tmp_path)
    assert MemoryBank.load(tmp_path / names["memory"]).num_positions == 3



def test_off_grid_proposal_fails_loudly(tmp_path):
    """norm_mode must match the server: a real (non-tier-1) proposal off the FAST token grid raises."""
    from hs_rtcf import spectral as sp
    from hs_rtcf.action_op import resolve
    ns = resolve("assets/pi0_fast_libero_norm_stats.json")
    op = make_op()
    op.encoder.encode = lambda obs: (tier1_pixel_descriptor(np.zeros((8, 8, 3), np.uint8)), "siglip-mock", np.zeros((8, 8, 3), np.uint8))
    rng = np.random.default_rng(10)
    q = rng.integers(-30, 30, (10, 7))
    for mode, ok in (("zscore", True), ("quantile", False)):
        s = sp.load_scale(ns, mode=mode)
        a = ((sp.dct_matrix(10).T @ (q / 10.0)) / s[None, :]).astype(np.float32)
        obs = {"images": {"agentview": np.zeros((256, 256, 3), np.uint8)}, "states": np.zeros(8, np.float32)}
        if ok:
            assert op.transform(a, obs, {}).tobytes() == a.tobytes()
        else:
            with pytest.raises(RuntimeError, match="off the FAST token grid"):
                op.transform(a, obs, {})
