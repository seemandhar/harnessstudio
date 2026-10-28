"""Method configs and registration rules (spec §4.3)."""
from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from .capabilities import Capability
from .errors import RegistrationError

SLOT_NAMES = ("sampler", "proposer", "selector", "executor", "episode_op", "controller")
SERVER_SLOTS = frozenset({"sampler"})
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Fidelity(str, Enum):
    FAITHFUL = "faithful"
    COMPONENT = "component"
    TF_VARIANT = "tf_variant"


@dataclass(frozen=True)
class ArtifactPin:
    name: str
    path: str
    sha256: str
    license: str

    def __post_init__(self):
        if not self.name or not self.path or not _HEX64.match(self.sha256) or not self.license.strip():
            raise RegistrationError(f"Artifact {self.name!r} needs a path, a sha256 and a license")


@dataclass(frozen=True)
class ModuleRef:
    factory: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        module, sep, attr = self.factory.partition(":")
        if not module or sep != ":" or not attr.isidentifier():
            raise RegistrationError(f"Module factory must be 'package.module:attr', got {self.factory!r}")
        try:
            text = json.dumps(dict(self.params), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RegistrationError(f"Module params for {self.factory} must be JSON-serializable") from exc
        object.__setattr__(self, "params", json.loads(text))

    def build(self):
        module, _, attr = self.factory.partition(":")
        return getattr(importlib.import_module(module), attr)(**self.params)

    def to_json(self) -> dict:
        return {"factory": self.factory, "params": json.loads(json.dumps(self.params, sort_keys=True))}


@dataclass(frozen=True)
class MethodConfig:
    name: str
    version: str
    paper: str
    fidelity: Fidelity
    substitutions: tuple
    required: frozenset
    slots: Mapping[str, ModuleRef]
    artifacts: tuple = ()
    trainable_parameters: int = 0
    description: str = ""

    def module(self, slot: str) -> Optional[ModuleRef]:
        return self.slots.get(slot)

    def to_json(self) -> dict:
        return {"name": self.name, "version": self.version, "paper": self.paper,
                "fidelity": Fidelity(self.fidelity).value, "substitutions": list(self.substitutions),
                "required": sorted(Capability(c).value for c in self.required),
                "slots": {k: self.slots[k].to_json() for k in sorted(self.slots)},
                "artifacts": [vars(a) for a in self.artifacts],
                "trainable_parameters": self.trainable_parameters, "description": self.description}


def validate(config: MethodConfig) -> MethodConfig:
    if not _NAME.match(config.name or ""):
        raise RegistrationError(f"Method name {config.name!r} must match {_NAME.pattern}")
    if not str(config.version).strip() or not str(config.paper).strip():
        raise RegistrationError("Method needs a version and a paper reference ('none' allowed)")
    if config.trainable_parameters != 0:
        raise RegistrationError("Training-free only: trainable_parameters must be 0")
    if Fidelity(config.fidelity) is Fidelity.TF_VARIANT and not config.substitutions:
        raise RegistrationError("tf_variant methods must name their substitutions")
    if Fidelity(config.fidelity) is not Fidelity.TF_VARIANT and config.substitutions:
        raise RegistrationError("Only tf_variant methods list substitutions")
    for cap in config.required:
        Capability(cap)
    unknown = set(config.slots) - set(SLOT_NAMES)
    if unknown:
        raise RegistrationError(f"Unknown slots {sorted(unknown)}; known: {list(SLOT_NAMES)}")
    for ref in config.slots.values():
        if not isinstance(ref, ModuleRef):
            raise RegistrationError("Every slot must be a ModuleRef")
    if "selector" in config.slots:
        proposer = config.slots.get("proposer")
        k = proposer.params.get("num_samples") if proposer else None
        if type(k) is not int or k < 2:
            raise RegistrationError("A selector needs a proposer with integer num_samples >= 2")
    if "proposer" in config.slots:
        k = config.slots["proposer"].params.get("num_samples")
        if type(k) is not int or k < 1:
            raise RegistrationError("A proposer must declare integer num_samples >= 1")
    if "executor" in config.slots:
        params = config.slots["executor"].params
        if type(params.get("min_k")) is not int or params["min_k"] < 1:
            raise RegistrationError("An executor must declare integer min_k >= 1")
        if type(params.get("per_step")) is not bool:
            raise RegistrationError("An executor must declare boolean per_step")
    if "controller" in config.slots:
        from .controller import validate_controller_params
        validate_controller_params(config.slots["controller"].params)
    for artifact in config.artifacts:
        if not isinstance(artifact, ArtifactPin):
            raise RegistrationError("Artifacts must be ArtifactPin entries")
    return config
