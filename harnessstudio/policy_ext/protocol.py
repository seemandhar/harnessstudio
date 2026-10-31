"""HarnessStudio's extension of the vla-eval wire protocol (spec 4.2).

vla-eval payloads are free-form dicts, so T1+ requests ride inside a normal OBSERVATION message:

    request:  {<raw observation keys>..., "harnessstudio": {"v": 1, "seeds": [...],
               "sampler": {"factory": ..., "params": {...}} | None, "readouts": [...]}}
    response: {"harnessstudio": {"v": 1}, "proposals": [{"actions": [H, A], "extras": {...}}, ...]}

A request without the key is an ordinary vla-eval observation and gets the server's native reply.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..core.errors import PolicyError

KEY = "harnessstudio"
VERSION = 1


def as_chunk(actions: Any) -> np.ndarray:
    """A finite, non-empty [H, A] float array (a single action becomes [1, A])."""
    arr = np.asarray(actions)
    if arr.dtype == object or arr.dtype.kind not in "fiub":
        raise PolicyError("Actions must be numeric arrays")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or 0 in arr.shape:
        raise PolicyError(f"Actions must be [H, A] or [A], got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise PolicyError("Actions contain non-finite values")
    return arr


def make_request(raw_obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping] = None,
                 readouts: Sequence[str] = ()) -> dict:
    if KEY in raw_obs:
        raise PolicyError(f"Observation already carries a {KEY!r} key")
    if not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise PolicyError("Seeds must be a non-empty list of uint32 integers")
    spec = {"v": VERSION, "seeds": list(seeds), "sampler": dict(sampler) if sampler else None,
            "readouts": list(readouts)}
    return {**raw_obs, KEY: spec}


def split_request(obs: Mapping[str, Any]) -> tuple[dict, Optional[dict]]:
    """(observation without the extension, extension spec or None)."""
    if KEY not in obs:
        return dict(obs), None
    spec = obs[KEY]
    if not isinstance(spec, Mapping) or spec.get("v") != VERSION:
        raise PolicyError(f"Unsupported {KEY} request {spec!r}")
    rest = {k: v for k, v in obs.items() if k != KEY}
    return rest, dict(spec)


def make_response(proposals: Sequence[tuple[Any, Mapping[str, Any]]]) -> dict:
    return {KEY: {"v": VERSION},
            "proposals": [{"actions": np.asarray(a, dtype=np.float32), "extras": dict(e)} for a, e in proposals]}


def parse_response(payload: Mapping[str, Any], expected: int) -> list[tuple[np.ndarray, dict]]:
    if not isinstance(payload, Mapping) or KEY not in payload:
        raise PolicyError("The policy server does not speak the HarnessStudio extension "
                          "(it returned a native reply to a seeded request)")
    proposals = payload.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != expected:
        raise PolicyError(f"Expected {expected} proposals, got "
                          f"{len(proposals) if isinstance(proposals, list) else type(proposals).__name__}")
    out = []
    for item in proposals:
        out.append((as_chunk(item["actions"]), dict(item.get("extras") or {})))
    return out
