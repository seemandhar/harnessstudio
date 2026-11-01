"""Development stage: build the frozen success-memory bank from `hs prepare` records.

collect_arm = "candidate": with no memory artifact the RTCF action_op is in collection mode and returns the native
chunk unchanged (verified by tier 1 here and by the native-equivalence check on LIBERO), so development episodes
are native pi0-FAST rollouts ("successful trajectories ... collected with memory seed 7", v2 §4.1). Each decision
event's diagnostics carry the descriptor and the full proposed chunk; successful episodes become memories.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from . import spectral
from .action_op import resolve
from .encoder import from_f32_b64
from .memory import MemoryBank


def decision_rows(record: dict) -> list[dict]:
    """The action_op collection diagnostics of one record, in decision order.

    HarnessStudio logs component diagnostics at the END of a decision, after its actions ran. When the benchmark
    ends an episode mid-chunk (every LIBERO success does), the method task is cancelled and that last decision's
    diagnostics are never logged, so the memory loses each successful episode's final invocation. If the
    framework logs pre-execution diagnostics (proposed extension `decision-pre-diagnostics`: event "decision_pre"
    from the optional `pre_diagnostics()` hook), those complete rows are used instead."""
    events = record.get("events") or []
    kind = "decision_pre" if any(ev.get("t") == "decision_pre" for ev in events) else "decision"
    rows = []
    for ev in events:
        if ev.get("t") != kind:
            continue
        diag = ((ev.get("diagnostics") or {}).get("action_op")) or {}
        if diag.get("mode") != "collect":
            raise ValueError(f"development record {record.get('task')}#{record.get('episode_idx')} has a decision "
                             "without collection diagnostics (was memory loaded during development?)")
        rows.append(diag)
    return rows


class Preparer:
    collect_arm = "candidate"
    record = ()          # nothing per step: everything comes from the per-decision diagnostics

    def __init__(self, only_success: bool = True, norm_stats: str = "assets/pi0_fast_libero_norm_stats.json",
                 norm_stats_sha256: Optional[str] = None, token_scale: float = 10.0,
                 zero_chunk_tol: float = 1e-4, norm_mode: str = "zscore") -> None:
        self.only_success = bool(only_success)
        self.norm_stats_path = resolve(norm_stats)
        self.norm_mode = norm_mode
        self.scale = spectral.load_scale(self.norm_stats_path, norm_stats_sha256, mode=norm_mode)
        self.center = spectral.norm_center(self.norm_stats_path, norm_mode, norm_stats_sha256)
        self.token_scale = float(token_scale)
        self.zero_chunk_tol = float(zero_chunk_tol)

    def build(self, records, out_dir) -> dict:
        out_dir = Path(out_dir)
        descs, coefs, failed, offsets, cases = [], [], [], [0], []
        encoders, shape, missing_state = set(), None, 0
        ordered = sorted(records, key=lambda r: (str(r["benchmark"]), str(r["task"]), int(r["episode_idx"])))
        for r in ordered:
            if self.only_success and not r.get("success"):
                continue
            rows = decision_rows(r)
            if not rows:
                continue
            for d in rows:
                h, a = d["chunk_shape"]
                if shape is None:
                    shape = (h, a)
                elif shape != (h, a):
                    raise ValueError(f"chunk shape changed {shape} -> {(h, a)}")
                chunk = from_f32_b64(d["chunk_f32"], (h, a))
                c = spectral.coefficients(chunk, self.scale)
                descs.append(from_f32_b64(d["descriptor_f32"]))
                coefs.append(spectral.quantize(c, self.token_scale))
                state = from_f32_b64(d["state_f32"]).astype(np.float64) if "state_f32" in d else None
                missing_state += int(state is None)
                failed.append(spectral.is_openpi_zero_chunk(chunk, state, tol=self.zero_chunk_tol, center=self.center))
                encoders.add(d["encoder"])
            offsets.append(offsets[-1] + len(rows))
            cases.append({"benchmark": r["benchmark"], "task": r["task"], "episode_idx": int(r["episode_idx"]),
                          "length": len(rows), "success": bool(r.get("success"))})
        if len(encoders) > 1:
            raise ValueError(f"development records used several encoders: {sorted(encoders)}")
        dims = {len(x) for x in descs}
        if len(dims) > 1:
            raise ValueError(f"descriptor dimensions differ: {sorted(dims)}")
        meta = {"format": "rtcf-memory-v1", "encoder": sorted(encoders)[0] if encoders else None,
                "norm_stats_sha256": hashlib.sha256(self.norm_stats_path.read_bytes()).hexdigest(),
                "token_scale": self.token_scale, "norm_mode": self.norm_mode, "chunk_shape": list(shape) if shape else None,
                "only_success": self.only_success, "memories": cases, "num_memories": len(cases),
                "num_positions": offsets[-1], "decode_failed_positions": int(sum(failed)),
                "positions_without_state": missing_state, "decode_failure_rule": "openpi zero-chunk exact signature",
                "development_records": len(records),
                "development_successes": sum(bool(r.get("success")) for r in records)}
        if descs:
            bank = MemoryBank(np.stack(descs).astype(np.float32), np.asarray(offsets, np.int64),
                              np.stack(coefs).astype(np.int64), np.asarray(failed, bool), meta)
        else:
            bank = MemoryBank.empty(1, shape[0] if shape else 1, shape[1] if shape else 1)
            bank.meta = meta
        bank.save(out_dir / "memory.npz", out_dir / "memory.json")
        return {"memory": "memory.npz", "memory_meta": "memory.json"}
