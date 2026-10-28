"""Built-in methods plus entry-point plugins (group 'harnessstudio.methods')."""
from __future__ import annotations

import importlib
from importlib import metadata

from .errors import RegistrationError
from .method import MethodConfig, validate

ENTRY_GROUP = "harnessstudio.methods"
BUILTINS = {
    "identity": "harnessstudio.methods.identity:config",
    "chunktrust-ahs": "harnessstudio.methods.chunktrust_ahs:config",
    "medoid-bon": "harnessstudio.methods.medoid_bon:config",
    "llm-subtask-planner": "harnessstudio.methods.llm_subtask_planner:config",
    "chunk-interrupt": "harnessstudio.methods.chunk_interrupt:config",
}


def _load(ref: str) -> MethodConfig:
    module, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module), attr)()


def plugin_entry_points() -> list:
    return list(metadata.entry_points(group=ENTRY_GROUP))


def all_methods(include_plugins: bool = True) -> dict:
    found = {}
    for name, ref in BUILTINS.items():
        try:
            config = _load(ref)
        except ModuleNotFoundError as exc:
            if exc.name != ref.partition(":")[0]:
                raise  # a real missing dependency must not be hidden
            continue  # built-in not implemented yet (milestone build order)
        validate(config)
        if config.name != name:
            raise RegistrationError(f"Built-in {name} returned a config named {config.name}")
        found[name] = (config, {"kind": "builtin", "ref": ref})
    if include_plugins:
        for ep in plugin_entry_points():
            config = validate(ep.load()())
            if config.name in found:
                raise RegistrationError(f"Method name {config.name!r} is already registered")
            if config.name != ep.name:
                raise RegistrationError(f"Entry point {ep.name!r} returned a config named {config.name!r}")
            found[config.name] = (config, {"kind": "plugin", "distribution": ep.dist.metadata["Name"],
                                           "version": ep.dist.version, "ref": getattr(ep, "value", None)})
    return found


def get(name: str):
    methods = all_methods()
    if name not in methods:
        raise KeyError(f"Unknown method {name!r}; run `rh list`")
    return methods[name]
