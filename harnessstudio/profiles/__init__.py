"""Policy profiles (spec 4.1, 16.1): what the harness knows about a policy server.

A profile names the capabilities a server offers and who owns chunking. Profiles ship as YAML here;
a path to another YAML file also works.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..context.policy import PolicyProfile
from ..core import capabilities

_HERE = Path(__file__).parent


def profile_names() -> list[str]:
    return sorted(p.stem for p in _HERE.glob("*.yaml"))


def load_profile(name_or_path: str | Path) -> PolicyProfile:
    path = Path(name_or_path)
    if not path.suffix:
        path = _HERE / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    return PolicyProfile(capabilities=capabilities.check(data.get("capabilities") or ["act"]),
                         chunking=data.get("chunking", "harness"), execute_horizon=data.get("execute_horizon"))
