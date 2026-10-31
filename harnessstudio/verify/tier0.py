"""Tier 0: static checks, no execution (spec 8.2)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..core.method import MethodConfig, validate
from ..profiles import load_profile, profile_names


def check(config: MethodConfig) -> list[str]:
    """Return a list of problems (empty = pass)."""
    problems: list[str] = []
    try:
        validate(config)
    except Exception as exc:  # noqa: BLE001
        return [f"registration: {exc}"]
    for art in config.artifacts:
        path = Path(art.path)
        if not path.exists():
            problems.append(f"artifact {art.name}: {art.path} does not exist")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if digest is not None and digest != art.sha256:
            problems.append(f"artifact {art.name}: sha256 mismatch")
    supported = [n for n in profile_names() if set(config.required) <= set(load_profile(n).capabilities)]
    if not supported:
        problems.append(f"no registered policy profile offers {sorted(config.required)}")
    for slot, ref in config.slots.items():
        try:
            ref.build()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"slot {slot}: cannot build {ref.factory}: {exc}")
    if config.controller is not None:
        try:
            config.controller.build()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"controller: cannot build {config.controller.factory}: {exc}")
    return problems
