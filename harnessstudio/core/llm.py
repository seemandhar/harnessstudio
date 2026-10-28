"""Budgeted runtime LLM operators for controller methods (milestone 2).

An operator is built from a controller's frozen ``llm`` params and lives in the episode worker.
Credentials come only from the environment variable the method names (``api_key_env``); the
value is never written to manifests, traces or logs. The model is pinned: the first call checks
that it is listed by the provider's ``/models`` endpoint, and there is no fallback model.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional, Sequence

from .errors import MethodContractError, RegistrationError

FORBIDDEN_MODEL_WORDS = ("astra",)
PROVIDERS = ("openai_compatible", "fake")
PRESETS = {"nrp": "https://ellm.nrp-nautilus.io/v1"}


class LLMBudgetExhausted(Exception):
    """Raised by tools.llm() when the declared per-episode call or token budget is spent."""


def validate_llm_params(params: Mapping[str, Any]) -> None:
    provider = params.get("provider")
    if provider not in PROVIDERS:
        raise RegistrationError(f"llm.provider must be one of {PROVIDERS}, got {provider!r}")
    if type(params.get("max_calls")) is not int or params["max_calls"] < 1:
        raise RegistrationError("llm.max_calls must be an integer >= 1 (per episode)")
    if type(params.get("max_tokens")) is not int or params["max_tokens"] < 1:
        raise RegistrationError("llm.max_tokens must be an integer >= 1 (per call)")
    if provider == "openai_compatible":
        model = params.get("model")
        if not isinstance(model, str) or not model.strip():
            raise RegistrationError("llm.model must be pinned to one exact model id")
        if any(word in model.lower() for word in FORBIDDEN_MODEL_WORDS):
            raise RegistrationError(f"llm.model {model!r} is not permitted")
        base = params.get("base_url")
        local = isinstance(base, str) and base.startswith(("http://127.0.0.1", "http://localhost"))
        if not isinstance(base, str) or not (base.startswith("https://") or local or base in PRESETS):
            raise RegistrationError("llm.base_url must be an https URL, a local http server or a preset name "
                                    + str(sorted(PRESETS)))
        key_env = params.get("api_key_env")
        if not isinstance(key_env, str) or not key_env.isidentifier():
            raise RegistrationError("llm.api_key_env must name an environment variable")
    if params.get("extra_body") is not None and not isinstance(params["extra_body"], Mapping):
        raise RegistrationError("llm.extra_body must be a mapping of provider options")
    if provider == "fake" and not isinstance(params.get("responses"), list):
        raise RegistrationError("llm.provider 'fake' needs a 'responses' list (tests only)")


def credential_names(method_json: Mapping[str, Any]) -> list:
    """Environment-variable NAMES a method's controller needs in the episode worker."""
    ref = (method_json.get("slots") or {}).get("controller") or {}
    llm = (ref.get("params") or {}).get("llm") or {}
    name = llm.get("api_key_env")
    return [name] if isinstance(name, str) and name else []


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Operator:
    """Common accounting. Subclasses implement _complete(messages, max_tokens) -> (text, usage)."""

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)
        self.max_calls, self.max_tokens = params["max_calls"], params["max_tokens"]
        self.calls, self.prompt_tokens, self.completion_tokens, self.seconds = 0, 0, 0, 0.0
        self.log: list = []

    def __call__(self, messages: Sequence[Mapping[str, Any]], *, max_tokens: Optional[int] = None) -> str:
        if self.calls >= self.max_calls:
            raise LLMBudgetExhausted(f"llm budget of {self.max_calls} calls per episode is spent")
        limit = self.max_tokens if max_tokens is None else min(int(max_tokens), self.max_tokens)
        started = time.monotonic()
        text, usage = self._complete(list(messages), limit)
        elapsed = time.monotonic() - started
        if not isinstance(text, str):
            raise MethodContractError("LLM operator returned a non-text completion")
        self.calls += 1
        self.seconds += elapsed
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        request = json.dumps(_text_only(messages), sort_keys=True, ensure_ascii=False)
        self.log.append({"call": self.calls, "request_sha256": _sha(request), "response": text,
                         "seconds": elapsed, "usage": dict(usage)})
        return text

    def summary(self) -> dict:
        return {"provider": self.params["provider"], "model": self.params.get("model"),
                "calls": self.calls, "max_calls": self.max_calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": self.seconds}


def _text_only(messages) -> list:
    """Messages with image payloads replaced by their hashes (for logging)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    parts.append({"type": "image_url", "sha256": _sha(url)})
                else:
                    parts.append(dict(part))
            content = parts
        out.append({"role": m.get("role"), "content": content})
    return out


class FakeOperator(Operator):
    """Scripted responses for CPU tests. Cycles through params['responses']."""

    def _complete(self, messages, max_tokens):
        responses = self.params["responses"]
        return str(responses[self.calls % len(responses)]), {"prompt_tokens": 0, "completion_tokens": 0}


class OpenAICompatibleOperator(Operator):
    """POST {base_url}/chat/completions with a pinned model. Standard library only."""
    _verified: dict = {}

    def __init__(self, params):
        super().__init__(params)
        base = params["base_url"]
        self.base_url = PRESETS.get(base, base).rstrip("/")
        self.model = params["model"]
        self.timeout = float(params.get("timeout_seconds", 60))
        self.temperature = float(params.get("temperature", 0.0))
        self.seed = params.get("seed")
        self.extra_body = dict(params.get("extra_body") or {})

    def _key(self) -> str:
        value = os.environ.get(self.params["api_key_env"], "")
        if not value:
            raise RuntimeError(f"Environment variable {self.params['api_key_env']} is not set in the episode worker")
        return value

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method="GET" if data is None else "POST",
                                     headers={"Authorization": "Bearer " + self._key(),
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:  # never echo request headers
            raise RuntimeError(f"LLM provider returned HTTP {exc.code} for {path}") from None

    def verify_model(self) -> None:
        key = (self.base_url, self.model)
        if key in self._verified:
            return
        listed = {m.get("id") for m in self._request("/models").get("data", [])}
        if self.model not in listed:
            raise RuntimeError(f"Pinned model {self.model!r} is not listed by {self.base_url}/models; no fallback")
        self._verified[key] = True

    def _complete(self, messages, max_tokens):
        self.verify_model()
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": self.temperature}
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        for key, value in self.extra_body.items():  # frozen provider options, e.g. disabling hidden reasoning
            payload.setdefault(key, value)
        body = self._request("/chat/completions", payload)
        if body.get("model") and any(w in str(body["model"]).lower() for w in FORBIDDEN_MODEL_WORDS):
            raise RuntimeError("Provider answered with a forbidden model")
        message = body["choices"][0]["message"]
        return message.get("content") or "", body.get("usage") or {}


def build_operator(params: Optional[Mapping[str, Any]]) -> Optional[Operator]:
    if not params:
        return None
    validate_llm_params(params)
    return FakeOperator(params) if params["provider"] == "fake" else OpenAICompatibleOperator(params)
