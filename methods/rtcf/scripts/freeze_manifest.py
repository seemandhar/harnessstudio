#!/usr/bin/env python3
"""Frozen-source record for rtcf (preflight review R5 / E1 / F8).

HarnessStudio v2's method fingerprint covers the config JSON only (slots, params, card; core/method.py), not the
method's source code, so a code edit with unchanged params keeps the fingerprint. This script records the sha256 of
every file of the method package and re-checks it at stage boundaries.

    .venv/bin/python methods/rtcf/scripts/freeze_manifest.py write OUT.json [--memory PREP/artifacts/memory.npz]
    .venv/bin/python methods/rtcf/scripts/freeze_manifest.py check OUT.json     # exit 1 on any difference

Recorded: per-file sha256 of methods/rtcf/{hs_rtcf,bench_ext,scripts,configs,tests} + pyproject.toml + README.md,
an aggregate tree hash, the git HEAD and `git status --porcelain methods/rtcf` of the harnessstudio repository, the
rtcf / rtcf-framenn / rtcf-timedomain method fingerprints and params, the imported hs_rtcf location (editable
install), and optionally the prepared memory sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]          # methods/rtcf
REPO = PKG_DIR.parents[1]                              # harnessstudio checkout
PARTS = ("hs_rtcf", "bench_ext", "scripts", "configs", "tests")


def file_hashes() -> dict:
    out = {}
    for part in PARTS:
        for p in sorted((PKG_DIR / part).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                out[str(p.relative_to(PKG_DIR))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("pyproject.toml", "README.md"):
        p = PKG_DIR / name
        if p.is_file():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def record(memory: str | None) -> dict:
    files = file_hashes()
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    from harnessstudio.core.registry import load
    import hs_rtcf
    methods = {}
    for name in ("rtcf", "rtcf-framenn", "rtcf-timedomain"):
        cfg = load(name)
        methods[name] = {"fingerprint": cfg.fingerprint(), "version": cfg.version,
                         "params": cfg.slots["action_op"].params}
    rec = {"package_dir": str(PKG_DIR), "imported_hs_rtcf": str(Path(hs_rtcf.__file__).resolve().parent),
           "git_head": git("rev-parse", "HEAD"), "git_status_methods_rtcf": git("status", "--porcelain", "methods/rtcf"),
           "tree_sha256": tree, "files": files, "methods": methods,
           "fingerprint_note": "HarnessStudio v2 fingerprints config only (core/method.py); code is covered by 'files'"}
    if memory:
        rec["memory_npz_sha256"] = hashlib.sha256(Path(memory).read_bytes()).hexdigest()
        rec["memory_npz"] = str(Path(memory).resolve())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write", "check"])
    ap.add_argument("path")
    ap.add_argument("--memory")
    a = ap.parse_args()
    if a.mode == "write":
        rec = record(a.memory)
        Path(a.path).write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(json.dumps({"tree_sha256": rec["tree_sha256"], "git_head": rec["git_head"],
                          "rtcf_fingerprint": rec["methods"]["rtcf"]["fingerprint"],
                          "dirty": bool(rec["git_status_methods_rtcf"])}))
        return 0
    old = json.loads(Path(a.path).read_text())
    new = record(old.get("memory_npz"))
    problems = []
    for k in sorted(set(old["files"]) | set(new["files"])):
        if old["files"].get(k) != new["files"].get(k):
            problems.append(f"file changed: {k}")
    for name, m in old["methods"].items():
        if new["methods"].get(name, {}).get("fingerprint") != m["fingerprint"]:
            problems.append(f"fingerprint changed: {name}")
    if old.get("memory_npz_sha256") and old["memory_npz_sha256"] != new.get("memory_npz_sha256"):
        problems.append("memory.npz changed")
    if old["imported_hs_rtcf"] != new["imported_hs_rtcf"]:
        problems.append(f"hs_rtcf now imported from {new['imported_hs_rtcf']}")
    print(json.dumps({"ok": not problems, "problems": problems, "tree_sha256": new["tree_sha256"]}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
