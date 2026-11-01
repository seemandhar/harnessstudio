"""RTCF as a DecisionLoop action_op (Alg. 1): at every policy invocation, encode o_t, extend the PMA frontiers,
read the aligned memory chunk, apply the bounded low-frequency residual, and return the chunk for execution.

Modes:
- memory loaded (evaluation): retrieve + correct; if the bank is empty, the aligned record is past the end of its
  memory (read_offset > 0) or it was a failed FAST decode, the native chunk is returned unchanged.
- no memory artifact (development / `hs prepare`, collection mode): the native chunk is returned unchanged, so the
  candidate arm is the native policy, and diagnostics carry what the Preparer needs (descriptor, full chunk,
  optionally the encoded image as PNG).
The derived null runs this op too (same encoder and PMA compute) and executes the native chunk.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import spectral
from .encoder import TIER1_ENCODER, Encoder, f32_b64, png_b64
from .memory import MemoryBank
from .pma import ProgressiveMemoryAlignment

PKG = Path(__file__).resolve().parent


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PKG / p


class RTCFActionOp:
    def __init__(self, image_key: str = "agentview", encoder_url: str = "http://127.0.0.1:8149",
                 encoder_model: str = "google/siglip-so400m-patch14-224", encoder_revision: str = "UNPINNED",
                 pooling: str = "pooler_output", encoder_timeout: float = 30.0,
                 norm_stats: str = "assets/pi0_fast_libero_norm_stats.json",
                 norm_stats_sha256: Optional[str] = None, token_scale: float = 10.0,
                 lambda0: float = 0.1, f_cut: int = 3, delta_max: float = 1.0, v_max: int = 2, gamma: float = 0.05,
                 read_offset: int = 0, motion: Sequence[str] = ("position", "rotation"),
                 action_limit: Optional[float] = 1.0, retrieval: str = "pma", correction: str = "spectral",
                 store_png: bool = True, state_key: str = "states", zero_chunk_tol: float = 1e-4,
                 encoder_weights_sha256: str = "", norm_mode: str = "zscore", token_grid_tol: float = 0.02) -> None:
        self.encoder = Encoder(image_key, encoder_url, encoder_model, encoder_revision, pooling, encoder_timeout,
                               encoder_weights_sha256)
        self.norm_mode = norm_mode
        self.scale = spectral.load_scale(resolve(norm_stats), norm_stats_sha256, mode=norm_mode)
        self.center = spectral.norm_center(resolve(norm_stats), norm_mode, norm_stats_sha256)
        self.token_grid_tol = float(token_grid_tol)
        self.state_key, self.zero_chunk_tol = state_key, float(zero_chunk_tol)
        self.token_scale = float(token_scale)
        self.lambda0, self.f_cut, self.delta_max = float(lambda0), int(f_cut), float(delta_max)
        self.v_max, self.gamma, self.read_offset = int(v_max), float(gamma), int(read_offset)
        self.motion_names = tuple(motion)
        self.motion_dims: list = []
        self.action_limit = None if action_limit is None else float(action_limit)
        if retrieval not in ("pma", "frame_nn") or correction not in ("spectral", "time_domain"):
            raise ValueError(f"unknown retrieval/correction {retrieval!r}/{correction!r}")
        self.retrieval, self.correction, self.store_png = retrieval, correction, bool(store_png)
        self.bank: Optional[MemoryBank] = None
        self.pma: Optional[ProgressiveMemoryAlignment] = None
        self._diag: dict = {}
        self.stats = {"decisions": 0, "applied": 0, "skipped_empty": 0, "skipped_end": 0,
                      "skipped_decode_failed": 0, "skipped_unusable": 0, "native_decode_failed": 0}

    # -- framework hooks ----------------------------------------------------------------
    def bind(self, embodiment) -> None:
        self.motion_dims = list(embodiment.dims(*self.motion_names))

    def load_artifacts(self, artifacts) -> None:
        if "memory" in artifacts:
            self.bank = MemoryBank.load(artifacts["memory"], artifacts.get("memory_meta"))
            if self.bank.num_memories:
                self.pma = ProgressiveMemoryAlignment(self.bank.descriptors, self.bank.offsets, self.v_max,
                                                      self.gamma, self.retrieval)

    def diagnostics(self) -> dict:
        return dict(self._diag)

    def pre_diagnostics(self) -> dict:
        """Collection-mode record for a framework that logs diagnostics BEFORE executing the chunk (proposed
        extension `decision-pre-diagnostics`); not called by HarnessStudio main as of f4c9a5d. Empty otherwise."""
        return dict(self._diag) if self._diag.get("mode") == "collect" else {}

    def _memory_coef(self, i: int, q: int):
        try:
            return spectral.dequantize(self.bank.coef_q[self.bank.row(i, q)], self.token_scale)
        except Exception:  # noqa: BLE001  (Alg. 1: an unavailable aligned chunk means no correction)
            return None

    @staticmethod
    def _usable(c_mem, native: np.ndarray) -> bool:
        """Alg. 1 fallback: the aligned chunk must exist, match the proposal's shape and be finite (as must A^pi)."""
        return (c_mem is not None and np.shape(c_mem) == native.shape and bool(np.all(np.isfinite(c_mem)))
                and bool(np.all(np.isfinite(native))))

    # -- one policy invocation ------------------------------------------------------------
    def transform(self, chunk, obs, memory):
        t0 = time.monotonic()
        native = np.asarray(chunk)
        if not self.motion_dims:
            raise RuntimeError("RTCFActionOp was not bound to an embodiment")
        self.stats["decisions"] += 1
        z, enc_name, image = self.encoder.encode(obs)
        t_enc = time.monotonic() - t0
        state = obs.get(self.state_key)
        state = None if state is None else np.asarray(state, dtype=np.float64).reshape(-1)
        native_failed = spectral.is_openpi_zero_chunk(native, state, tol=self.zero_chunk_tol, center=self.center)
        if enc_name != TIER1_ENCODER and self.token_grid_tol > 0:
            # The policy decoded this chunk from integer FAST tokens: in the server's own normalization its f > 0
            # coefficients sit on the 1/10 grid. Off-grid means norm_mode does not match the server: fail loudly.
            grid = spectral.token_grid_error(native, self.scale, self.token_scale)
            if grid > self.token_grid_tol:
                raise RuntimeError(f"proposal is off the FAST token grid in norm_mode={self.norm_mode!r} "
                                   f"(max error {grid:.3f} > {self.token_grid_tol}); the policy server normalizes "
                                   "actions differently")
        self.stats["native_decode_failed"] += int(native_failed)
        diag: dict = {"t": self.stats["decisions"], "encoder": enc_name, "enc_ms": round(1000 * t_enc, 2),
                      "native_decode_failed": native_failed}
        if self.bank is None:  # collection mode (development): native chunk, record what memory needs
            diag.update({"mode": "collect", "descriptor_f32": f32_b64(z), "chunk_f32": f32_b64(native),
                         "chunk_shape": list(native.shape)})
            if state is not None:   # needed to recognise a failed FAST decode in the memory (exact signature)
                diag["state_f32"] = f32_b64(state)
            if self.store_png:
                diag["png_b64"] = png_b64(image)
            self._diag = diag
            return native
        out = native
        if self.pma is None:
            self.stats["skipped_empty"] += 1
            diag.update({"mode": "eval", "applied": False, "skip": "empty_bank"})
        else:
            t1 = time.monotonic()
            i, b, d = self.pma.update(z)
            q = b + self.read_offset
            diag.update({"mode": "eval", "i": i, "b": b, "d": round(d, 6), "L": self.bank.length(i),
                         "pma_ms": round(1000 * (time.monotonic() - t1), 3)})
            if q >= self.bank.length(i):
                self.stats["skipped_end"] += 1
                diag.update({"applied": False, "skip": "past_memory_end"})
            elif bool(self.bank.decode_failed[self.bank.row(i, q)]):
                self.stats["skipped_decode_failed"] += 1
                diag.update({"applied": False, "skip": "memory_decode_failed"})
            elif not self._usable(c_mem := self._memory_coef(i, q), native):
                self.stats["skipped_unusable"] += 1
                diag.update({"applied": False, "skip": "memory_chunk_unusable"})
            else:
                out, info = spectral.rtcf_correct(native, c_mem, self.scale, self.motion_dims, self.f_cut,
                                                  self.delta_max, self.lambda0, self.action_limit, self.correction)
                self.stats["applied"] += 1
                diag.update({"applied": True, **info})
        diag["stats"] = dict(self.stats)
        diag["op_ms"] = round(1000 * (time.monotonic() - t0), 2)
        self._diag = diag
        return out
