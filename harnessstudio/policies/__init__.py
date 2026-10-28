"""Policy capability adapters."""
from __future__ import annotations

import importlib

_ADAPTERS = {"gr00t-n1.5": "harnessstudio.policies.gr00t_n15", "pi0.5": "harnessstudio.policies.pi05"}
_ALIASES = {"groot": "gr00t-n1.5", "gr00t": "gr00t-n1.5", "gr00t-n15": "gr00t-n1.5", "pi05": "pi0.5", "pi-0.5": "pi0.5"}


def names() -> list:
    return sorted(_ADAPTERS)


def canonical(name: str) -> str:
    key = name.lower().replace("_", "-")
    return _ALIASES.get(key, key)


def adapter(name: str):
    key = canonical(name)
    if key not in _ADAPTERS:
        raise ValueError(f"No HarnessStudio adapter for policy {name!r}; available: {names()}")
    return importlib.import_module(_ADAPTERS[key])
