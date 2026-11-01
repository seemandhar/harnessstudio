#!/usr/bin/env python3
"""Live check that rtcf's coefficient coordinates are pi0-FAST's own: on real server chunks, 10 * s_d * DCT(A)
for f > 0 must be (numerically) integers, because the policy output is the inverse DCT of integer FAST tokens
after an affine per-dimension unnormalization and a per-dimension state offset (which only touches f = 0).

    .venv/bin/python methods/rtcf/scripts/probe_fast_integrality.py RUN_DIR [--tol 1e-3] [--out R.json]

RUN_DIR: an `hs prepare` development run (collection mode records every full proposed chunk in the decision
diagnostics). Reports the max |10 s C - round(10 s C)| over f > 0, per dimension, and the number of chunks with the
exact openpi zero-chunk signature (critic B4; those are excluded from the integrality maximum).
A sabotaged scale (--sabotage) must fail, proving the probe can detect a wrong normalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from hs_rtcf import spectral
from hs_rtcf.action_op import resolve
from hs_rtcf.encoder import from_f32_b64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--min-chunks", type=int, default=200)
    ap.add_argument("--sabotage", action="store_true", help="perturb the scale by 3%% (must fail)")
    ap.add_argument("--out")
    a = ap.parse_args()
    scale = spectral.load_scale(resolve("assets/pi0_fast_libero_norm_stats.json"))
    q01, q99 = spectral.load_norm_bounds(resolve("assets/pi0_fast_libero_norm_stats.json"))
    if a.sabotage:
        scale = scale * 1.03
    errs, failed, n, no_state = [], 0, 0, 0
    for p in sorted(Path(a.run).glob("*/traces/*.json")):
        r = json.loads(p.read_text())
        for ev in r.get("events") or []:
            d = ((ev.get("diagnostics") or {}).get("action_op")) or {}
            if "chunk_f32" not in d:
                continue
            a_chunk = from_f32_b64(d["chunk_f32"], tuple(d["chunk_shape"]))
            c = spectral.coefficients(a_chunk, scale)
            n += 1
            state = from_f32_b64(d["state_f32"]) if "state_f32" in d else None
            no_state += int(state is None)
            if spectral.is_openpi_zero_chunk(a_chunk, state, q01, q99):     # B4 exact signature hits
                failed += 1
                continue
            x = 10.0 * c[1:]
            errs.append(np.abs(x - np.round(x)).max(axis=0))
    per_dim = np.max(np.stack(errs), axis=0).tolist() if errs else []
    worst = max(per_dim) if per_dim else None
    report = {"chunks": n, "decode_failed_exact_signature": failed, "chunks_without_state": no_state, "max_abs_err_per_dim": per_dim, "max_abs_err": worst,
              "tol": a.tol, "sabotage": a.sabotage, "ok": bool(errs) and n >= a.min_chunks and worst <= a.tol}
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
