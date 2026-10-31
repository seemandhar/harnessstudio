"""Content fingerprints and per-episode trace records (spec 10)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def fingerprint(value: Any) -> str:
    """Stable content identity; arrays include dtype and shape."""
    return hashlib.sha256(json.dumps(_canon(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _canon(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be fingerprinted")
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, Mapping):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    return value


def jsonable(value: Any) -> Any:
    """Make event fields portable; arrays become fingerprints."""
    if isinstance(value, np.ndarray):
        return {"array": _canon(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def case_filename(benchmark: str, task: str, episode_idx: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task)[:120]
    return f"{benchmark}__{safe}__{int(episode_idx)}.json"


def write_record(out_dir: Path, record: Mapping[str, Any]) -> Path:
    out_dir = Path(out_dir) / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / case_filename(record["benchmark"], record["task"], record["episode_idx"])
    path.write_text(json.dumps(jsonable(record), allow_nan=False))
    return path
