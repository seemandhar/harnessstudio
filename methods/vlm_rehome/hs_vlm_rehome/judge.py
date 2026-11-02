"""Frozen VLM judge for vlm-rehome (r4-perception P1, VPR): one prompt, one agent-view image, strict JSON verdict.

This module is the single source of every byte the operator receives, for BOTH the Q1 qualification harness
(scripts/q1_qualify.py) and the deployed controller (critic C-V2: delivered-image identity):

- ``image_png(array)``: the agent-view uint8 HxWx3 array exactly as the public observation holds it (vla-eval LIBERO
  ``obs['images']['agentview']``, 256x256, upright), PNG-encoded with Pillow ``compress_level=1`` (the encoder the RTCF S2
  traces used for their stored ``png_b64`` frames, so re-encoding a decoded S2 frame reproduces its stored bytes).
- ``build_messages(instruction, png)``: the frozen system/user messages (text + one ``image_url`` data URL).
- ``build_payload(messages)``: the exact JSON body harnessstudio's ``OpenAICompatibleOperator.complete`` sends for the
  pinned operator (key order model, messages, max_tokens, temperature, then the pin's extra_body), so a direct Q1 call
  and a deployed ``ctx.tools['llm']`` call put the same body bytes on the wire.
- ``parse_verdict(text)`` and ``classify(...)``: strict parse; semantic verdicts {abnormal, normal, unsure} are kept
  apart from transport abstentions {invalid_json, timeout, http_error, error} (critic C-V4).

Decision rule (frozen): flag iff the reply arrived within ``TIMEOUT_S`` and parses as exactly one JSON object whose
``state`` is ``"abnormal"``. Everything else abstains (native). No retries, no fallback model, no paraphrases.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from typing import Any, Mapping, Optional

import numpy as np

# ---- operator pin (critic C-V1; `python -m harnessstudio.endpoints pin nrp qwen3-vl --name judge`) -------------------
PIN = {
    "name": "judge",
    "model": "qwen3-vl",
    "base_url": "nrp",
    "api_key_env": "NRP_API_KEY",
    "original_model": "",
    "temperature": 0.0,
    # chat_template_kwargs: the registry's pin (thinking off). response_format: structured output constrained to the
    # verdict schema, added BEFORE freeze after a 1-call transport smoke on a non-Q1 frame (step-0 frame of S2 t0_i10)
    # returned 64 tokens of prose for the plain prompt (smoke/smoke.jsonl); the prompt text itself is unchanged.
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False},
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "arm_state", "strict": True,
                       "schema": {"type": "object", "properties": {"state": {"type": "string",
                                                                             "enum": ["abnormal", "normal", "unsure"]}},
                                  "required": ["state"], "additionalProperties": False}}}},
}
BASE_URL = "https://ellm.nrp-nautilus.io/v1"
EXPECTED_SERVED_MODEL = "Qwen/Qwen3.8-Flash-Next-FP8"   # endpoints.yaml served_as; drift stops the arm (C-V1)
MAX_TOKENS = 64
TIMEOUT_S = 60.0
IMAGE_KEY = "agentview"
IMAGE_SHAPE = (256, 256, 3)

# ---- frozen prompt (r4-perception P1 prompt_plan; single prompt, no ensemble, no revision after Q1) ---------------------
SYSTEM = "You check a robot arm in a tabletop task. Answer only JSON."
TASK_PREFIX = "Task: "
QUESTION = ("The robot has just finished one part of this task and released an object. Is the arm now in an ABNORMAL "
            "configuration (gripper flipped or pointing up/sideways while lifting away, arm swinging high over "
            "objects, or knocking an object), or NORMAL (moving at table height toward the next object, gripper "
            "roughly downward, or pushing or closing something as the task needs)? "
            "Reply {\"state\": \"abnormal\"|\"normal\"|\"unsure\"}.")

SEMANTIC = ("abnormal", "normal", "unsure")
TRANSPORT = ("invalid_json", "timeout", "http_error", "error")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_png(array: Any) -> bytes:
    """PNG bytes of an agent-view frame (uint8 HxWx3, as observed; no resize, flip or colour change)."""
    from PIL import Image
    a = np.asarray(array)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"agent-view frame must be uint8 HxWx3, got {a.dtype} {a.shape}")
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def decode_png(png: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def build_messages(instruction: str, png: bytes) -> list:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": TASK_PREFIX + str(instruction)},
            {"type": "image_url", "image_url": {"url": data_url(png)}},
            {"type": "text", "text": QUESTION},
        ]},
    ]


def prompt_sha256(messages: list) -> str:
    """Hash of the messages with each image replaced by a fixed placeholder: identifies the prompt bytes independently of
    the image (the delivered PNG is hashed separately)."""
    def strip(m):
        c = m.get("content")
        if isinstance(c, list):
            c = [{"type": "image_url", "image_url": "<image>"} if p.get("type") == "image_url" else dict(p) for p in c]
        return {"role": m.get("role"), "content": c}
    return sha256(json.dumps([strip(m) for m in messages], sort_keys=True, ensure_ascii=False).encode())


def build_payload(messages: list, model: str = PIN["model"], max_tokens: int = MAX_TOKENS,
                  temperature: Optional[float] = PIN["temperature"], extra_body: Mapping = PIN["extra_body"]) -> dict:
    """Mirror of harnessstudio.tools.llm.OpenAICompatibleOperator.complete payload construction (tests compare them)."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature
    for k, v in dict(extra_body).items():
        payload.setdefault(k, v)
    return payload


def body_bytes(payload: Mapping) -> bytes:
    """The request body exactly as OpenAICompatibleOperator._request encodes it."""
    return json.dumps(payload).encode()


_FENCE = re.compile(r"\A```(?:json)?\s*\n?(.*?)\n?\s*```\Z", re.DOTALL)


def parse_verdict(text: Optional[str]) -> str:
    """'abnormal' | 'normal' | 'unsure' for a reply that is exactly one JSON object with a valid 'state' (one optional
    surrounding markdown code fence is tolerated); 'invalid_json' otherwise."""
    if not isinstance(text, str):
        return "invalid_json"
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return "invalid_json"
    if not isinstance(obj, dict):
        return "invalid_json"
    state = obj.get("state")
    if not isinstance(state, str) or state.strip().lower() not in SEMANTIC:
        return "invalid_json"
    return state.strip().lower()


def classify(text: Optional[str], seconds: Optional[float], error: Optional[str] = None) -> str:
    """Final per-call class. A reply after TIMEOUT_S counts as a timeout abstention whatever it says (the deployed
    framework operator may wait longer; the decision is the same as Q1's 60 s client timeout)."""
    if error is not None:
        return error if error in TRANSPORT else "error"
    if seconds is None or seconds > TIMEOUT_S:
        return "timeout"
    return parse_verdict(text)


def flags(verdict: str) -> bool:
    return verdict == "abnormal"
