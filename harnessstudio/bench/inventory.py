"""GPU id parsing and the frozen source inventory recorded in every run manifest."""
from __future__ import annotations

from pathlib import Path

from .. import config
from .common import sha256


def parse_gpu_ids(value):
    try:
        ids = [int(word) for word in value.split(',')]
    except ValueError as exc:
        raise ValueError('GPU IDs must be comma-separated integers or available') from exc
    if not ids or any(g < 0 for g in ids) or len(ids) != len(set(ids)):
        raise ValueError('GPU IDs must be nonempty, nonnegative and unique')
    return ids


def source_inventory(bench, extra):
    """sha256 of every file that can affect execution: HarnessStudio's package, RobotPolicyBench's
    scripts/GPU map/checkpoint pin, plus caller-declared extras (files or directories)."""
    bench = Path(bench)
    paths = list(config.PACKAGE_DIR.rglob('*.py')) + list(config.PACKAGE_DIR.rglob('*.json'))
    paths += list((bench / 'scripts').glob('*.py'))
    paths += [bench / 'registry/gpu-mapping.json', bench / 'registry/gr00t-n15-checkpoint-verification.json']
    for item in extra:
        item = Path(item)
        if not item.exists():
            raise ValueError('Missing source dependency: ' + str(item))
        paths.extend(item.rglob('*') if item.is_dir() else [item])
    unique = sorted({p.resolve() for p in paths if p.is_file() and '__pycache__' not in p.parts})
    return {str(p): sha256(p) for p in unique}, {}
