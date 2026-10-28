"""Rebuild a MethodConfig from a frozen method manifest (inside child processes)."""
from __future__ import annotations

from ..core.capabilities import Capability
from ..core.method import ArtifactPin, Fidelity, MethodConfig, ModuleRef, validate


def method_from_manifest(manifest: dict) -> MethodConfig:
    m = manifest["method"]
    return validate(MethodConfig(
        name=m["name"], version=m["version"], paper=m["paper"], fidelity=Fidelity(m["fidelity"]),
        substitutions=tuple(m["substitutions"]), required=frozenset(Capability(c) for c in m["required"]),
        slots={k: ModuleRef(v["factory"], v["params"]) for k, v in m["slots"].items()},
        artifacts=tuple(ArtifactPin(**a) for a in m["artifacts"]),
        trainable_parameters=m["trainable_parameters"], description=m["description"]))
