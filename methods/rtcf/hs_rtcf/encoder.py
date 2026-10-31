"""Visual descriptors z_t: frozen SigLIP final-layer (pooled) features, l2-normalized, no PCA (training-free
substitute for the paper's fitted SigLIP-PCA projection).

SigLIPClient talks to the method-local service `scripts/siglip_service.py` (its own venv, pinned weights) over
HTTP: POST {url}/embed {"png_b64": ...} -> {"embedding_f32_b64", "dim", "model", "revision", "pooling"}. The
reply's model/revision/pooling must equal the pin, otherwise the episode fails (no silent fallback).

Tier-1 seam: HarnessStudio's synthetic benchmark sends one 8x8 'cam' image of uniform grey level. Only for that
exact signature (no configured image key, images == {'cam': uint8 [8, 8, 3]}) the deterministic
`tier1_pixel_descriptor` is used and reported as encoder 'tier1-pixels'. Anything else without the configured
image key raises.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request
from typing import Any, Mapping

import numpy as np

TIER1_ENCODER = "tier1-pixels"


def png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(np.asarray(image, dtype=np.uint8))).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def png_b64(image: np.ndarray) -> str:
    return base64.b64encode(png_bytes(image)).decode("ascii")


def f32_b64(vec: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(vec, dtype="<f4")).tobytes()).decode("ascii")


def from_f32_b64(text: str, shape=None) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float32)
    return arr.reshape(shape) if shape is not None else arr


def unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("descriptor has zero or non-finite norm")
    return (v / n).astype(np.float32)


def is_tier1_fake(images: Mapping[str, Any], image_key: str) -> bool:
    if image_key in images or set(images) != {"cam"}:
        return False
    a = np.asarray(images["cam"])
    return a.dtype == np.uint8 and a.shape == (8, 8, 3)


def tier1_pixel_descriptor(image: np.ndarray, bins: int = 16, width: float = 0.06) -> np.ndarray:
    """Deterministic descriptor of the fake benchmark's uniform grey image: Gaussian bumps over its mean level."""
    level = float(np.asarray(image, dtype=np.float64).mean()) / 255.0
    centers = np.linspace(0.0, 1.0, bins)
    return unit(np.exp(-0.5 * ((level - centers) / width) ** 2) + 1e-3)


class SigLIPClient:
    def __init__(self, url: str, model: str, revision: str, pooling: str, timeout: float = 30.0) -> None:
        self.url = os.environ.get("HS_RTCF_ENCODER_URL", url).rstrip("/")
        self.model, self.revision, self.pooling, self.timeout = model, revision, pooling, float(timeout)
        self.calls, self.seconds = 0, 0.0

    def encode_png(self, png_b64_text: str) -> np.ndarray:
        body = json.dumps({"png_b64": png_b64_text}).encode()
        req = urllib.request.Request(self.url + "/embed", data=body, headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            reply = json.loads(resp.read().decode())
        self.seconds += time.monotonic() - t0
        self.calls += 1
        for key, want in (("model", self.model), ("revision", self.revision), ("pooling", self.pooling)):
            if str(reply.get(key)) != str(want):
                raise RuntimeError(f"SigLIP service answered {key}={reply.get(key)!r}, pinned {want!r}")
        return unit(from_f32_b64(reply["embedding_f32_b64"]))


class Encoder:
    """Picks the pinned SigLIP service, or the tier-1 seam for the synthetic benchmark only."""

    def __init__(self, image_key: str, url: str, model: str, revision: str, pooling: str, timeout: float = 30.0):
        self.image_key = image_key
        self.client = SigLIPClient(url, model, revision, pooling, timeout)
        self.name = f"{model}@{revision}:{pooling}"

    def image(self, obs: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
        images = obs.get("images") or {}
        if self.image_key in images:
            return np.asarray(images[self.image_key], dtype=np.uint8), False
        if is_tier1_fake(images, self.image_key):
            return np.asarray(images["cam"], dtype=np.uint8), True
        raise KeyError(f"observation has no image {self.image_key!r} (images: {sorted(images)})")

    def encode(self, obs: Mapping[str, Any]) -> tuple[np.ndarray, str, np.ndarray]:
        """(unit descriptor float32, encoder name, the image that was encoded)."""
        img, fake = self.image(obs)
        if fake:
            return tier1_pixel_descriptor(img), TIER1_ENCODER, img
        return self.client.encode_png(png_b64(img)), self.name, img
