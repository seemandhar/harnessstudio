"""Pinned, budgeted runtime LLM/VLM operators (spec 5.5; ported from v1 core/llm.py).

- One exact model per operator on an OpenAI-compatible endpoint; the first call checks /models;
  there is no fallback model. Astra models are refused.
- The API key comes only from the environment variable the paper card names; it is never logged.
- Requests are logged by hash (images replaced by their hashes), responses in full.
- Tier 1 never touches the network: controllers provide TIER1_LLM_RESPONSES and the harness server
  runs with scripted operators.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import time
import urllib.error
import urllib.request
import zlib
from typing import Any, Mapping, Optional, Sequence

import anyio
import numpy as np

from ..core.errors import BudgetExceeded, RegistrationError

PRESETS = {"nrp": "https://ellm.nrp-nautilus.io/v1"}
FORBIDDEN_MODEL_WORDS = ("astra",)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def text_only(messages: Sequence[Mapping[str, Any]]) -> list:
    """Messages with image payloads replaced by hashes (for logging)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [{"type": "image_url", "sha256": _sha(p.get("image_url", {}).get("url", ""))}
                       if p.get("type") == "image_url" else dict(p) for p in content]
        out.append({"role": m.get("role"), "content": content})
    return out


def png_bytes(array: Any) -> bytes:
    """Minimal PNG encoder (stdlib + NumPy): grayscale HxW or RGB HxWx3, uint8."""
    a = np.asarray(array)
    while a.ndim > 3 or (a.ndim == 3 and a.shape[-1] not in (1, 3) and a.shape[0] == 1):
        a = a[0]
    a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3):
        raise ValueError(f"Cannot encode an image of shape {a.shape}")
    color = 2 if a.ndim == 3 else 0
    height, width = a.shape[:2]
    raw = b"".join(b"\x00" + a[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def image_data_url(array: Any) -> str:
    """PNG data URL for vision-capable models."""
    return "data:image/png;base64," + base64.b64encode(png_bytes(array)).decode()


class Operator:
    """Blocking completion; LLMTool runs it in a worker thread."""

    def complete(self, messages: list, max_tokens: int) -> tuple[str, dict]:
        raise NotImplementedError


class ScriptedOperator(Operator):
    def __init__(self, responses: Sequence[str]) -> None:
        if not responses:
            raise RegistrationError("Scripted LLM needs at least one response")
        self.responses, self.i = list(responses), 0

    def complete(self, messages, max_tokens):
        text = str(self.responses[self.i % len(self.responses)])
        self.i += 1
        return text, {"prompt_tokens": 0, "completion_tokens": 0}


class OpenAICompatibleOperator(Operator):
    _verified: dict = {}

    def __init__(self, model: str, base_url: str, api_key_env: str, temperature: float = 0.0,
                 timeout: float = 120.0, seed: Optional[int] = None, extra_body: Optional[Mapping] = None) -> None:
        if any(w in model.lower() for w in FORBIDDEN_MODEL_WORDS):
            raise RegistrationError(f"Model {model!r} is not permitted")
        self.model = model
        self.base_url = PRESETS.get(base_url, base_url).rstrip("/")
        if not (self.base_url.startswith("https://") or self.base_url.startswith(("http://127.0.0.1", "http://localhost"))):
            raise RegistrationError("base_url must be https, a local server, or a preset: " + ", ".join(PRESETS))
        self.api_key_env, self.temperature, self.timeout, self.seed = api_key_env, temperature, timeout, seed
        self.extra_body = dict(extra_body or {})

    def _request(self, path: str, payload: Optional[dict] = None) -> dict:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"Environment variable {self.api_key_env} is not set")
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method="GET" if data is None else "POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:  # never echo request headers
            raise RuntimeError(f"LLM provider returned HTTP {exc.code} for {path}") from None

    def verify(self) -> None:
        if (self.base_url, self.model) in self._verified:
            return
        listed = {m.get("id") for m in self._request("/models").get("data", [])}
        if self.model not in listed:
            raise RuntimeError(f"Pinned model {self.model!r} is not listed by {self.base_url}/models; no fallback")
        self._verified[(self.base_url, self.model)] = True

    def complete(self, messages, max_tokens):
        self.verify()
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": self.temperature}
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        for k, v in self.extra_body.items():
            payload.setdefault(k, v)
        body = self._request("/chat/completions", payload)
        if any(w in str(body.get("model", "")).lower() for w in FORBIDDEN_MODEL_WORDS):
            raise RuntimeError("Provider answered with a forbidden model")
        return body["choices"][0]["message"].get("content") or "", body.get("usage") or {}


class LLMTool:
    """ctx.tools['llm']: `await ctx.tools['llm'](messages, operator=None, max_tokens=None) -> str`."""

    def __init__(self, ctx, operators: Mapping[str, Operator], max_calls: int, max_tokens: int) -> None:
        self._ctx, self._ops, self.max_calls, self.max_tokens = ctx, dict(operators), max_calls, max_tokens
        self.log: list = []

    async def __call__(self, messages: Sequence[Mapping[str, Any]], operator: Optional[str] = None,
                       max_tokens: Optional[int] = None) -> str:
        acc = self._ctx.accounting
        if acc.llm_calls >= self.max_calls:
            raise BudgetExceeded(f"llm budget of {self.max_calls} calls per episode is spent")
        name = operator or next(iter(self._ops))
        if name not in self._ops:
            raise KeyError(f"Undeclared operator {name!r}; declared: {sorted(self._ops)}")
        limit = self.max_tokens if max_tokens is None else min(int(max_tokens), self.max_tokens)
        started = time.monotonic()
        text, usage = await anyio.to_thread.run_sync(self._ops[name].complete, list(messages), limit)
        elapsed = time.monotonic() - started
        acc.llm_calls += 1
        acc.llm_tokens += int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        request = json.dumps(text_only(messages), sort_keys=True, ensure_ascii=False)
        self._ctx._log("llm", operator=name, request_sha256=_sha(request), response=text,
                       seconds=round(elapsed, 3), usage=dict(usage))
        return text


def build_operators(pins, scripted: Optional[Sequence[str]] = None) -> dict[str, Operator]:
    """Operators from OperatorPins; scripted=responses replaces every operator (tier 1)."""
    if scripted is not None:
        return {pin.name: ScriptedOperator(scripted) for pin in pins} or {"llm": ScriptedOperator(scripted)}
    return {pin.name: OpenAICompatibleOperator(pin.model, pin.base_url, pin.api_key_env, pin.temperature,
                                               seed=pin.seed, extra_body=pin.extra_body) for pin in pins}
