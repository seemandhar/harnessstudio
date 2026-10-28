"""Where HarnessStudio reads and writes. Every location can be set with an environment variable.

HARNESSSTUDIO_WORKSPACE    run outputs go to WORKSPACE/runs/harnessstudio/...   (default: current dir)
HARNESSSTUDIO_LEASES       shared GPU lease registry directory                   (default: nearest
                           `.leases` directory at or above the workspace, else ~/.harnessstudio/leases)
HARNESSSTUDIO_POLICYBENCH  RobotPolicyBench root (policy/simulator envs, checkpoints, GPU map)
                           (default: $RMH_POLICYBENCH, else WORKSPACE/../RobotPolicyBench)
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # .../harnessstudio/harnessstudio
SOURCE_ROOT = PACKAGE_DIR.parent                       # directory that contains the package
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_TABLE = DATA_DIR / "canonical_tables" / "robocasa365_atomic_seen_dev_r2.json"


def workspace() -> Path:
    return Path(os.environ.get("HARNESSSTUDIO_WORKSPACE") or os.getcwd()).resolve()


def runs_dir() -> Path:
    return workspace() / "runs" / "harnessstudio"


def leases_dir() -> Path:
    if os.environ.get("HARNESSSTUDIO_LEASES"):
        return Path(os.environ["HARNESSSTUDIO_LEASES"]).resolve()
    for parent in (workspace(), *workspace().parents):
        if (parent / ".leases").is_dir():
            return (parent / ".leases").resolve()
    return Path.home() / ".harnessstudio" / "leases"


def policybench() -> Path:
    for key in ("HARNESSSTUDIO_POLICYBENCH", "RMH_POLICYBENCH"):
        if os.environ.get(key):
            return Path(os.environ[key]).resolve()
    return (workspace().parent / "RobotPolicyBench").resolve()


def ledger_path() -> Path:
    return runs_dir() / "gpu_ledger.json"
