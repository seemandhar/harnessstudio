"""Frozen method manifest and source closure (spec §3.3, §8.2)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

from .errors import RegistrationError
from .method import MethodConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # .../harnessstudio/harnessstudio
COMPILED = (".so", ".pyd", ".dylib")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _walk(directory: Path) -> list:
    files = sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if any(p.suffix in COMPILED for p in files):
        raise RegistrationError(f"Compiled extensions are not allowed in a method closure: {directory}")
    return [p for p in files if p.suffix in (".py", ".json") or p.name.endswith("LICENSE")]


def distribution_closure(dist, *, top_packages) -> list:
    direct = dist.read_text("direct_url.json")
    info = json.loads(direct) if direct else {}
    if info.get("dir_info", {}).get("editable"):
        source = Path(unquote(urlparse(info["url"]).path))
        files = []
        for top in top_packages:
            found = next((c for c in (source / top, source / "src" / top) if c.is_dir()), None)
            if found is None:
                raise RegistrationError(f"Editable plugin package {top!r} not found under {source}")
            files += _walk(found)
        return files
    names = [str(f) for f in (dist.files or [])]
    if any(n.endswith(COMPILED) for n in names):
        raise RegistrationError("Compiled extensions are not allowed in a method closure")
    return [Path(dist.locate_file(f)) for f in (dist.files or [])
            if str(f).endswith(".py") and str(f).split("/")[0] in top_packages]


def harnessstudio_closure() -> list:
    return _walk(PACKAGE_ROOT)


def method_manifest(config: MethodConfig, origin: dict, *, policy: str) -> dict:
    files = harnessstudio_closure()
    top = []
    if origin.get("kind") == "plugin":
        from importlib import metadata
        dist = metadata.distribution(origin["distribution"])
        top = sorted({origin["ref"].split(":")[0].split(".")[0]})
        files += distribution_closure(dist, top_packages=top)
    closure = {str(p): sha256_file(p) for p in sorted(set(files))}
    body = {"schema": "harnessstudio-method-manifest/1", "method": config.to_json(), "origin": origin,
            "policy": policy, "closure": closure, "plugin_top_packages": top}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def _plugin_relpath(src: Path, top_packages) -> Path:
    """plugins/<top_package>/<...> preserving the package-relative layout (nested packages work)."""
    parts = src.parts
    for top in top_packages:
        if top in parts:
            index = len(parts) - 1 - parts[::-1].index(top)
            return Path("plugins", *parts[index:])
    raise RegistrationError(f"Closure file {src} is outside the plugin packages {list(top_packages)}")


def freeze(manifest: dict, destination: Path) -> dict:
    """Copy the closure into destination so child processes import the copy.
    harnessstudio files go to harnessstudio/...; plugin files go under plugins/<top_package>/...,
    and destination/plugins is put on the child PYTHONPATH."""
    destination = Path(destination)
    top_packages = manifest.get("plugin_top_packages", [])
    out = {}
    for path, digest in manifest["closure"].items():
        src = Path(path)
        try:
            # Only files inside the harnessstudio package itself; examples/ and other in-tree
            # plugin sources go through _plugin_relpath.
            rel = Path("harnessstudio") / src.relative_to(PACKAGE_ROOT)
        except ValueError:
            rel = _plugin_relpath(src, top_packages)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        if sha256_file(target) != digest:
            raise RegistrationError(f"Closure file changed while freezing: {src}")
        out[str(rel)] = digest
    return {"root": str(destination), "files": out, "manifest_sha256": manifest["sha256"]}
