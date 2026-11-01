# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.5.1",
#     "transformers==4.46.3",
#     "safetensors>=0.4",
#     "pillow>=10",
#     "numpy>=1.24,<2.3",
#     "sentencepiece",
# ]
# ///
"""Frozen SigLIP descriptor service for the rtcf method (one image -> pooled final-layer features).

    CUDA_VISIBLE_DEVICES=G HF_HUB_OFFLINE=1 uv run methods/rtcf/scripts/siglip_service.py \
        --snapshot /path/to/models--google--siglip-so400m-patch14-224/snapshots/d04cf29fca7b6374f74d8bea1969314492266b5e \
        --revision d04cf29fca7b6374f74d8bea1969314492266b5e --port 8149 [--device cuda|cpu]

GET /health -> {"status": "ok", model, revision, pooling, dim, device, weights_sha256}
POST /embed {"png_b64": "..."} -> {"embedding_f32_b64", "dim", "model", "revision", "pooling", "ms"}

Model: SiglipVisionModel (vision tower of google/siglip-so400m-patch14-224), fp32, eval mode, no grad;
pooling = pooler_output (the MAP-head output of the final layer, 1152-d). Preprocessing = the checkpoint's own
SiglipImageProcessor (resize to 224, rescale 1/255, normalize mean/std 0.5). The image is decoded from lossless PNG,
so memory and query descriptors see identical pixels. Deterministic flags are set; requests are serialized.
Training-free: no parameter is updated; the weights' sha256 is checked at start against --weights-sha256.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

MODEL_ID = "google/siglip-so400m-patch14-224"
WEIGHTS_SHA256 = "30a00c26a045c84a0bd14adc94278d7f1cc941c8bd67e5fa50f7bed34c34e60d"   # model.safetensors @ d04cf29
PINNED_FILES = {   # critic B5: pin the processor config with the weights (bicubic resize 224, mean/std 0.5)
    "preprocessor_config.json": "bec6ca696ddd6c2050ce0ea9fcb7581b052d267059a3341b6c1456cab54b1d07",
    "config.json": "28c6981582dbd0dc6e1c2eeae21a240e6952019bca3cdffbc22b9da2c747cb6b",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--weights-sha256", default=WEIGHTS_SHA256)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8149)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel

    snap = Path(a.snapshot)
    digest = sha256_file(snap / "model.safetensors")
    if digest != a.weights_sha256:
        raise SystemExit(f"model.safetensors sha256 {digest} != pinned {a.weights_sha256}")
    for name, want in PINNED_FILES.items():
        got = sha256_file(snap / name)
        if got != want:
            raise SystemExit(f"{name} sha256 {got} != pinned {want}")
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    processor = SiglipImageProcessor.from_pretrained(snap)
    model = SiglipVisionModel.from_pretrained(snap, torch_dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lock = threading.Lock()
    info = {"status": "ok", "model": MODEL_ID, "revision": a.revision, "pooling": "pooler_output",
            "dim": int(model.config.hidden_size), "device": str(device), "weights_sha256": digest, "pinned_files": PINNED_FILES,
            "output_dtype": "float32 (memory and query descriptors identical, critic B5)"}

    def embed(png_b64: str) -> dict:
        img = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
        t0 = time.monotonic()
        with lock, torch.no_grad():
            px = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
            vec = model(pixel_values=px).pooler_output[0].float().cpu().numpy()
        return {"embedding_f32_b64": base64.b64encode(np.asarray(vec, "<f4").tobytes()).decode("ascii"),
                "dim": int(vec.shape[0]), "model": MODEL_ID, "revision": a.revision, "pooling": "pooler_output",
                "ms": round(1000 * (time.monotonic() - t0), 2)}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self._send(200, info) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/embed":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, embed(req["png_b64"]))
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args):  # quiet
            pass

    print(json.dumps(info), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
