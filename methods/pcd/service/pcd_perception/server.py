"""HTTP front end (stdlib only), in the pattern of methods/roboharness/scripts/rh_tool_server.py.

GET  /health                        -> {"ok", "version", "device", "weights", "sam2_C", "sessions", "torch_flags"}
POST /reset {"session", "instruction", "parser"?}  -> episode summary (state cleared; tracker rebuilt lazily);
            "parser": "simpler" (default, the authors' table) | "libero_goal" (frozen Setting-B table); sticky per session
POST /step  {"session", "instruction", "image_png" | "image_npy", "return_masks"?}
            -> {"cf_png" | "cf_npy", per-step evidence, "episode": summary, "timing_ms", "calls"}
POST /close {"session"}
Images: base64 PNG (lossless) or base64 .npy (uint8 HxWx3). The counterfactual has the input's resolution.
One lock serialises model use (the models are shared; each session has its own tracker state).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from . import SERVICE_VERSION, UPSTREAM_PIN

LOG = logging.getLogger("pcd_perception")


def decode_image(payload: dict) -> np.ndarray:
    if payload.get("image_npy"):
        arr = np.load(io.BytesIO(base64.b64decode(payload["image_npy"])), allow_pickle=False)
    elif payload.get("image_png"):
        arr = np.array(Image.open(io.BytesIO(base64.b64decode(payload["image_png"]))).convert("RGB"))
    else:
        raise ValueError("image_png or image_npy required")
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3, got {arr.dtype} {arr.shape}")
    return arr


def encode_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def encode_npy(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


class Service:
    def __init__(self, engine_factory, device: str, weights: dict, sam2_c: bool):
        self.engine_factory = engine_factory
        self.base = engine_factory()      # loads the frozen models once
        self.sessions = {}
        self.lock = threading.Lock()
        self.info = {"ok": True, "version": SERVICE_VERSION, "upstream": UPSTREAM_PIN, "device": device,
                     "weights": weights, "sam2_C": sam2_c}
        self.latency = {"steps": 0, "total_ms": 0.0}

    def _engine(self, session: str):
        if session not in self.sessions:
            self.sessions[session] = self.base.__class__(self.base.detector, self.base.tracker_factory,
                                                         self.base.inpainter, classnames=self.base.classnames,
                                                         get_all_parts=self.base.get_all_parts)
        return self.sessions[session]

    def handle(self, path: str, payload: dict) -> dict:
        session = str(payload.get("session", "default"))
        with self.lock:
            if path == "/reset":
                return self._engine(session).reset(payload.get("instruction"), parser=payload.get("parser"))
            if path == "/close":
                self.sessions.pop(session, None)
                return {"closed": session}
            if path == "/step":
                image = decode_image(payload)
                eng = self._engine(session)
                out = eng.step(image, payload["instruction"])
                cf = out.pop("_cf")
                dbg = out.pop("_masks")
                if payload.get("image_npy"):
                    out["cf_npy"] = encode_npy(cf)
                else:
                    out["cf_png"] = encode_png(cf)
                if payload.get("return_masks"):
                    out["masks_npy"] = {k: (encode_npy(np.asarray(v)) if v is not None else None)
                                        for k, v in dbg.items() if k != "targets"}
                    out["masks_npy"]["targets"] = [encode_npy(m) if m is not None else None
                                                   for m in dbg["targets"]]
                self.latency["steps"] += 1
                self.latency["total_ms"] += out["timing_ms"]["total_ms"]
                out["calls"] = dict(eng.calls)
                return out
        raise KeyError(path)


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                from .runtime import torch_flags
                info = dict(service.info)
                info["torch_flags"] = torch_flags()
                info["sessions"] = len(service.sessions)
                info["latency"] = dict(service.latency)
                self._send(200, info)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, service.handle(self.path, payload))
            except KeyError as e:
                self._send(404, {"error": f"unknown endpoint or field {e}"})
            except Exception as e:  # reported to the caller; the method package treats it as infrastructure
                LOG.exception("request failed")
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            LOG.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def main(argv=None) -> None:
    from .runtime import DEFAULT_MANIFEST, WeightPaths, build_real_engine, sam2_cuda_extension_available

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--device", default="cpu", help="cpu | cuda:0 (CUDA_VISIBLE_DEVICES selects the GPU)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .runtime import set_authors_torch_flags
    set_authors_torch_flags()   # before any model is built (also done inside build_real_engine)
    weights = WeightPaths.from_manifest(args.manifest)
    t = time.time()
    service = Service(lambda: build_real_engine(args.device, weights), args.device, weights.__dict__,
                      sam2_cuda_extension_available())
    LOG.info("models loaded in %.1f s on %s", time.time() - t, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    LOG.info("listening on http://%s:%d", args.host, args.port)
    server.serve_forever()
