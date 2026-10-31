"""Development stage (spec 7): run development cases, build frozen artifacts, pin them.

A method with `prepare=ModuleRef(...)` provides a Preparer:

    class Preparer:
        collect_arm = "baseline"        # arm that runs on development cases ("candidate" for exploration)
        record = ("state",)             # public observation keys kept per step (plus "action")
        def build(self, records: list[dict], out_dir: Path) -> dict[str, str]:
            ...                         # write files into out_dir; return {artifact name: file name}

Development records include outcomes (success) and per-step experience; that is allowed because this is
development. The result is `prepared.json` (artifact sha256 pins + the development case keys). Evaluation
with `--prepared` loads the artifacts read-only (ctx.artifacts: name -> path) and refuses any case whose
key was used in development.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from ..core.errors import RegistrationError
from ..core.method import MethodConfig


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_prepared(config: MethodConfig, records: list[dict], out_dir: Path,
                   previous: Optional[dict] = None) -> dict:
    """Call the method's Preparer on development records and pin what it writes.

    previous: the metadata of an earlier prepared.json when development is chained (hs prepare --prepared):
    its artifacts are passed to Preparer.build(..., previous=...) if the Preparer accepts that argument, and
    its development cases stay forbidden."""
    import inspect
    if config.prepare is None:
        raise RegistrationError(f"{config.name} has no prepare stage")
    preparer = config.prepare.build()
    art_dir = Path(out_dir) / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in records if r.get("valid")]
    if previous is not None and "previous" in inspect.signature(preparer.build).parameters:
        prev_artifacts = {a["name"]: a["path"] for a in previous.get("artifacts", [])}
        written = preparer.build(valid, art_dir, previous=prev_artifacts)
    else:
        written = preparer.build(valid, art_dir)
    artifacts = []
    for name, filename in sorted(written.items()):
        path = (art_dir / filename).resolve()
        if not path.is_file() or art_dir.resolve() not in path.parents:
            raise RegistrationError(f"Preparer artifact {name!r} must be a file inside {art_dir}")
        artifacts.append({"name": name, "path": str(path), "sha256": _sha(path)})
    prepared = {
        "method": config.name, "method_fingerprint": config.fingerprint(), "created": time.time(),
        "artifacts": artifacts,
        "dev_cases": sorted({(r["benchmark"], r["task"], r["episode_idx"]) for r in records}
                            | {tuple(c) for c in (previous or {}).get("dev_cases", [])}),
        "chained_from": (previous or {}).get("created"),
        "dev_episodes": len(records), "dev_valid": sum(bool(r.get("valid")) for r in records),
        "dev_successes": sum(bool(r.get("success")) for r in records),
    }
    (Path(out_dir) / "prepared.json").write_text(json.dumps(prepared, indent=2))
    return prepared


def load_prepared(path: str | Path, config: Optional[MethodConfig] = None) -> tuple[dict, set, dict]:
    """(artifacts name->path, forbidden case keys, prepared metadata). Verifies every sha256 pin."""
    path = Path(path)
    if path.is_dir():
        path = path / "prepared.json"
    prepared = json.loads(path.read_text())
    if config is not None and prepared["method"] != config.name:
        raise RegistrationError(f"{path} was prepared for {prepared['method']}, not {config.name}")
    artifacts = {}
    for art in prepared["artifacts"]:
        p = Path(art["path"])
        if not p.is_file() or _sha(p) != art["sha256"]:
            raise RegistrationError(f"Prepared artifact {art['name']} is missing or changed since it was pinned")
        artifacts[art["name"]] = str(p)
    forbidden = {tuple(c) for c in prepared["dev_cases"]}
    return artifacts, forbidden, prepared


def preparer_settings(config: MethodConfig) -> tuple[str, tuple]:
    preparer = config.prepare.build() if config.prepare else None
    return (str(getattr(preparer, "collect_arm", "baseline")), tuple(getattr(preparer, "record", ())))


def merge_dev_cases(prepared: Mapping[str, Any]) -> set:
    return {tuple(c) for c in prepared.get("dev_cases", [])}
