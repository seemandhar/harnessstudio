"""Method configs and registration rules (spec sections 5 and 6)."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional

from . import capabilities
from .categories import CATEGORIES
from .errors import RegistrationError

# Slots of the default DecisionLoop, in pipeline order (spec 5.3). "controller" is separate.
SLOT_NAMES = ("observation_op", "instruction_op", "gate", "proposer", "sampler",
              "selector", "action_op", "executor", "monitor")
NULL_RULES = ("derived", "shadow", "custom")

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Fidelity(str, Enum):
    FAITHFUL = "faithful"      # reproduces the paper's method
    COMPONENT = "component"    # ports one part of the paper
    TF_VARIANT = "tf_variant"  # replaces a trained part with a training-free one
    ADAPTED = "adapted"        # same method, substituted operator (e.g. an NRP LLM for Claude)


@dataclass(frozen=True)
class ArtifactPin:
    name: str
    path: str
    sha256: str
    license: str

    def __post_init__(self):
        if not self.name or not self.path or not _HEX64.match(self.sha256 or "") or not str(self.license).strip():
            raise RegistrationError(f"Artifact {self.name!r} needs a path, a sha256 and a license")


@dataclass(frozen=True)
class OperatorPin:
    """One runtime LLM/VLM: a single exact model on an OpenAI-compatible endpoint, no fallback.
    base_url is an https URL, a local server, or a preset name ("nrp")."""
    name: str
    model: str
    base_url: str
    api_key_env: str
    original_model: str = ""     # the paper's model, kept for provenance when substituted
    temperature: Optional[float] = 0.0   # None = omit (some models accept only their default)
    seed: Optional[int] = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    token_param: Optional[str] = None    # "max_tokens" | "max_completion_tokens"; None = auto by endpoint

    def __post_init__(self):
        if "astra" in self.model.lower():
            raise RegistrationError("Astra models are not allowed in this workspace")
        for value in (self.name, self.model, self.base_url, self.api_key_env):
            if not str(value).strip():
                raise RegistrationError("Operators need a name, model, base_url and api_key_env")
        if not str(self.api_key_env).isidentifier():
            raise RegistrationError("api_key_env must name an environment variable")
        if self.token_param not in (None, "max_tokens", "max_completion_tokens"):
            raise RegistrationError("token_param must be max_tokens or max_completion_tokens")


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
            raise RegistrationError(f"Params for {self.factory} must be JSON-serializable") from exc
        object.__setattr__(self, "params", json.loads(text))

    def build(self):
        module, _, attr = self.factory.partition(":")
        return getattr(importlib.import_module(module), attr)(**self.params)

    def to_json(self) -> dict:
        return {"factory": self.factory, "params": self.params}


@dataclass(frozen=True)
class Budgets:
    """Per-episode caps. policy_calls counts samples (K per sample() call, 1 per act()).
    None means bounded only by the horizon (allowed for DecisionLoop methods, not controllers)."""
    policy_calls: Optional[int] = None
    llm_calls: int = 0
    tool_calls: int = 0
    max_tokens: int = 0

    def __post_init__(self):
        for name in ("llm_calls", "tool_calls", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise RegistrationError(f"Budget {name} must be a non-negative integer")
        if self.policy_calls is not None and (type(self.policy_calls) is not int or self.policy_calls < 1):
            raise RegistrationError("Budget policy_calls must be a positive integer or None")


@dataclass(frozen=True)
class MethodConfig:
    name: str
    version: str
    paper: str
    fidelity: Fidelity
    substitutions: tuple = ()
    categories: tuple = ()
    required: frozenset = frozenset({"act"})
    slots: Mapping[str, ModuleRef] = field(default_factory=dict)
    controller: Optional[ModuleRef] = None
    prepare: Optional[ModuleRef] = None   # development stage (spec 7): builds frozen artifacts
    null: str = "derived"
    budgets: Budgets = Budgets()
    artifacts: tuple = ()
    operators: tuple = ()
    trainable_parameters: int = 0
    description: str = ""
    card: Mapping[str, Any] = field(default_factory=dict)  # the full paper.yaml, for provenance

    def with_params(self, slot: str, **params) -> "MethodConfig":
        ref = self.slots[slot]
        slots = dict(self.slots)
        slots[slot] = ModuleRef(ref.factory, {**ref.params, **params})
        return replace(self, slots=slots)

    def to_json(self) -> dict:
        return {
            "name": self.name, "version": self.version, "paper": self.paper,
            "fidelity": Fidelity(self.fidelity).value, "substitutions": list(self.substitutions),
            "categories": list(self.categories), "required": sorted(self.required),
            "slots": {k: self.slots[k].to_json() for k in sorted(self.slots)},
            "controller": self.controller.to_json() if self.controller else None,
            "prepare": self.prepare.to_json() if self.prepare else None,
            "null": self.null, "budgets": vars(self.budgets),
            "artifacts": [vars(a) for a in self.artifacts],
            "operators": [vars(o) for o in self.operators],
            "trainable_parameters": self.trainable_parameters,
        }

    def fingerprint(self) -> str:
        text = json.dumps(self.to_json(), sort_keys=True, allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()


def validate(config: MethodConfig) -> MethodConfig:
    if not _NAME.match(config.name or ""):
        raise RegistrationError(f"Method name {config.name!r} must match {_NAME.pattern}")
    if not str(config.version).strip() or not str(config.paper).strip():
        raise RegistrationError("A method needs a version and a paper reference ('none' allowed)")
    fidelity = Fidelity(config.fidelity)
    if config.trainable_parameters != 0:
        raise RegistrationError("Training-free only: trainable_parameters must be 0")
    if fidelity in (Fidelity.TF_VARIANT, Fidelity.ADAPTED) and not config.substitutions:
        raise RegistrationError(f"{fidelity.value} methods must name their substitutions")
    if fidelity in (Fidelity.FAITHFUL, Fidelity.COMPONENT) and config.substitutions:
        raise RegistrationError("Only tf_variant and adapted methods list substitutions")
    unknown = set(config.categories) - set(CATEGORIES)
    if unknown:
        raise RegistrationError(f"Unknown categories {sorted(unknown)}; known: {sorted(CATEGORIES)}")
    capabilities.check(config.required)
    if not set(config.slots) <= set(SLOT_NAMES):
        raise RegistrationError(f"Unknown slots {sorted(set(config.slots) - set(SLOT_NAMES))}; "
                                f"known: {list(SLOT_NAMES)}")
    for ref in list(config.slots.values()) + [r for r in (config.controller, config.prepare) if r]:
        if not isinstance(ref, ModuleRef):
            raise RegistrationError("Slots and the controller must be ModuleRefs")
    if config.null not in NULL_RULES:
        raise RegistrationError(f"null must be one of {NULL_RULES}")
    if config.controller is None and config.null != "derived":
        raise RegistrationError("Methods without a controller use the derived null")
    if config.controller is not None and config.null == "derived":
        raise RegistrationError("Controllers need a 'shadow' or 'custom' null")
    if config.controller is not None and config.budgets.policy_calls is None:
        raise RegistrationError("Controllers must cap policy_calls in their budgets")
    proposer = config.slots.get("proposer")
    k = proposer.params.get("num_samples", 1) if proposer else 1
    if proposer and (type(k) is not int or k < 1):
        raise RegistrationError("proposer num_samples must be a positive integer")
    if "selector" in config.slots and k < 2:
        raise RegistrationError("A selector needs a proposer with num_samples >= 2")
    if (k > 1 or "sampler" in config.slots) and not {"seeded_sampling", "chunk"} <= set(config.required):
        raise RegistrationError("Multiple samples or a sampler need 'chunk' and 'seeded_sampling'")
    readouts = (proposer.params.get("readouts") or ()) if proposer else ()
    needs = {"velocity": "velocity_readout", "attention": "attention_readout", "logits": "logits_readout",
             "futures": "predicted_futures"}
    for r in readouts:
        if r in needs and needs[r] not in config.required:
            raise RegistrationError(f"Readout {r!r} needs capability {needs[r]!r}")
    if "sampler" in config.slots and not {"denoise_hooks", "differentiable_denoiser",
                                           "decoding_hooks"} & set(config.required):
        raise RegistrationError("A sampler needs 'denoise_hooks', 'differentiable_denoiser' or 'decoding_hooks'")
    for a in config.artifacts:
        if not isinstance(a, ArtifactPin):
            raise RegistrationError("Artifacts must be ArtifactPins")
    for o in config.operators:
        if not isinstance(o, OperatorPin):
            raise RegistrationError("Operators must be OperatorPins")
    if config.operators and (config.budgets.llm_calls == 0 or config.budgets.max_tokens == 0):
        raise RegistrationError("A method with operators needs llm_calls and max_tokens budgets")
    return config
