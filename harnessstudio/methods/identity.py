"""The native frozen policy, run through the harness (the baseline arm of every comparison)."""
from __future__ import annotations

from ..core.method import Fidelity, MethodConfig


def config() -> MethodConfig:
    return MethodConfig(name="identity", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=frozenset({"act"}),
                        description="No harness: one native policy decision per chunk.")
