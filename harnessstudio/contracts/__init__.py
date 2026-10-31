"""Benchmark contracts (spec 3.3): which observation keys a method may see, and the episode horizon.

The policy always receives the raw benchmark observation (spec 16.2); the contract only limits what
methods see.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..core.errors import ContractError

FORBIDDEN_KEYS = frozenset({"reward", "success", "seed", "episode_idx", "task_id", "sim_state", "arm"})
_HERE = Path(__file__).parent


@dataclass(frozen=True)
class Contract:
    name: str
    public_keys: tuple
    instruction_key: str
    optional_keys: tuple = ()          # public keys a benchmark may omit (depends on its params)
    image_key: str = "images"
    state_key: Optional[str] = None
    action_dim: Optional[int] = None
    control_mode: str = "unknown"
    horizons: Mapping[str, int] = field(default_factory=dict)          # by task name
    suite_horizons: Mapping[str, int] = field(default_factory=dict)    # by task["suite"]
    default_horizon: int = 1000
    # Where the end-effector position lives in the public observation: {path: [key, ...], slice: [a, b] | null}
    eef_position: Mapping[str, Any] = field(default_factory=dict)
    # Fallback action layout (name -> [start, end]) for policies whose action_spec is one raw block
    action_layout: Mapping[str, Any] = field(default_factory=dict)
    # Analytic-skill constants for eef_delta control: metres per unit position action, gripper commands
    eef_delta_scale: Optional[float] = None
    gripper_open: float = -1.0
    gripper_close: float = 1.0

    def __post_init__(self):
        if not self.public_keys:
            raise ContractError(f"Contract {self.name} declares no public keys")
        leaked = set(self.public_keys) & FORBIDDEN_KEYS
        if leaked:
            raise ContractError(f"Contract {self.name} exposes evaluator keys {sorted(leaked)}")
        if self.instruction_key not in self.public_keys or self.instruction_key in self.optional_keys:
            raise ContractError(f"Contract {self.name}: instruction_key must be a required public key")
        if not set(self.optional_keys) <= set(self.public_keys):
            raise ContractError(f"Contract {self.name}: optional_keys must be public keys")

    def filter_public(self, obs: Mapping[str, Any]) -> dict:
        """Deep copy of the public part of an observation. Missing public keys are an error."""
        missing = [k for k in self.public_keys if k not in obs and k not in self.optional_keys]
        if missing:
            raise ContractError(f"Observation lacks public keys {missing}")
        return copy.deepcopy({k: obs[k] for k in self.public_keys if k in obs})

    def horizon(self, task: Mapping[str, Any]) -> int:
        name = str(task.get("name", ""))
        if name in self.horizons:
            return int(self.horizons[name])
        suite = task.get("suite")
        if suite is not None and str(suite) in self.suite_horizons:
            return int(self.suite_horizons[str(suite)])
        return int(self.default_horizon)


def load_contract(name_or_path: str | Path) -> Contract:
    path = Path(name_or_path)
    if not path.suffix:
        path = _HERE / f"{name_or_path}.yaml"
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"Cannot read contract {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(f"Contract {path} must be a mapping")
    data["public_keys"] = tuple(data.get("public_keys") or ())
    data["optional_keys"] = tuple(data.get("optional_keys") or ())
    try:
        return Contract(**data)
    except TypeError as exc:
        raise ContractError(f"Contract {path}: {exc}") from exc
