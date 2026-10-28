"""propose / get_action_seeded payload schema (spec §5.2 + plan clarification 1)."""
from __future__ import annotations

import json
from dataclasses import dataclass

MAX_EXTRAS_BYTES = 8 * 1024 * 1024
REQUEST_KEYS = {"observation", "arm", "decision_index", "step_index", "seeds", "want", "method_manifest_sha256"}
WANTS = {"sampler"}


@dataclass(frozen=True)
class ProposeRequest:
    observation: dict
    arm: str
    decision_index: int
    step_index: int
    seeds: list
    want: list
    method_manifest_sha256: str


def parse_request(payload, *, manifest_sha256: str) -> ProposeRequest:
    if not isinstance(payload, dict) or set(payload) != REQUEST_KEYS:
        raise ValueError(f"propose payload must have exactly {sorted(REQUEST_KEYS)}")
    if payload["arm"] not in ("null", "candidate"):
        raise ValueError("propose arm must be 'null' or 'candidate' (baseline uses get_action_seeded)")
    for key in ("decision_index", "step_index"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"{key} must be a nonnegative int")
    seeds = payload["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be a nonempty list of nonnegative ints")
    if not isinstance(payload["want"], list) or set(payload["want"]) - WANTS:
        raise ValueError(f"want must be a subset of {sorted(WANTS)}")
    if payload["method_manifest_sha256"] != manifest_sha256:
        raise ValueError("method manifest mismatch between worker and server")
    return ProposeRequest(**payload)


def parse_seeded(payload) -> tuple:
    if not isinstance(payload, dict) or set(payload) != {"observation", "noise_seed"}:
        raise ValueError("get_action_seeded payload must be {observation, noise_seed}")
    if type(payload["noise_seed"]) is not int or payload["noise_seed"] < 0:
        raise ValueError("noise_seed must be a nonnegative int")
    return payload["observation"], payload["noise_seed"]


def _size(value) -> int:
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_size(k) + _size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_size(v) for v in value)
    return len(json.dumps(value, default=str).encode())


def check_extras_size(extras: dict) -> int:
    size = _size(extras)
    if size > MAX_EXTRAS_BYTES:
        raise ValueError(f"extras are {size} bytes, above the {MAX_EXTRAS_BYTES}-byte bound")
    return size
