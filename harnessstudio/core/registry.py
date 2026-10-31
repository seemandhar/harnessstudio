"""Method discovery through the `harnessstudio.methods` entry point group (spec 6.1)."""
from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from .errors import RegistrationError
from .method import MethodConfig, validate

GROUP = "harnessstudio.methods"


def discover() -> dict:
    """name -> entry point, for every installed method package."""
    return {ep.name: ep for ep in entry_points(group=GROUP)}


def load(name: str) -> MethodConfig:
    """Load a method by entry-point name, or by 'package.module:attr' for unregistered code."""
    if ":" in name:
        module, _, attr = name.partition(":")
        factory = getattr(importlib.import_module(module), attr)
    else:
        eps = discover()
        if name not in eps:
            raise RegistrationError(f"No method named {name!r}; installed: {sorted(eps)}")
        factory = eps[name].load()
    config = factory()
    if not isinstance(config, MethodConfig):
        raise RegistrationError(f"{name} must return a MethodConfig")
    validate(config)
    if ":" not in name and config.name != name:
        raise RegistrationError(f"Entry point {name!r} returned a config named {config.name!r}")
    return config
