"""Minimal client for the method package (stdlib only)."""
from __future__ import annotations

import base64
import io
import json
import urllib.request
from typing import Optional

import numpy as np


def _npy(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.uint8), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


class PerceptionClient:
    def __init__(self, url: str = "http://127.0.0.1:8141", timeout: float = 300.0, session: str = "default"):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = session

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def health(self) -> dict:
        with urllib.request.urlopen(self.url + "/health", timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def reset(self, instruction: Optional[str] = None, parser: Optional[str] = None) -> dict:
        payload = {"session": self.session, "instruction": instruction}
        if parser is not None:
            payload["parser"] = parser
        return self._post("/reset", payload)

    def step(self, image: np.ndarray, instruction: str, return_masks: bool = False):
        out = self._post("/step", {"session": self.session, "instruction": instruction,
                                   "image_npy": _npy(image), "return_masks": return_masks})
        cf = np.load(io.BytesIO(base64.b64decode(out.pop("cf_npy"))), allow_pickle=False)
        return cf, out

    def close(self) -> dict:
        return self._post("/close", {"session": self.session})
