#!/usr/bin/env python3
"""Framework regression gate for HarnessStudio v2 (used before merging a framework extension) + in-flight check.

    python3 scripts/regress.py --inflight                       # running hs processes (do not edit what they import)
    python3 scripts/regress.py [--source WORKTREE] --out R.json # unit tests + tier 0 + tier 1 of every method

The gate runs the checkout given by --source (default: this repository) by putting it, and each of its
methods/<slug> folders, first on PYTHONPATH of the main venv, so a worktree's code shadows the editable installs.
Methods are discovered from the installed entry points; a method that exists only in the worktree must be
installed (uv pip install -e) before its tier 1 can run. `ok` is true only when everything passed.
Read-only: it never edits, installs or merges anything.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
HS_MARKERS = (" eval ", " prepare ", " serve ", " verify ")


def inflight() -> list:
    """Running `hs` / harnessstudio processes (any checkout), from /proc cmdlines."""
    me = os.getpid()
    out = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == me:
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        cmd = " ".join(a.decode(errors="replace") for a in argv if a)
        if not cmd:
            continue
        is_hs = ("/bin/hs " in cmd or cmd.startswith("hs ") or "harnessstudio.cli" in cmd) and \
            any(m in cmd + " " for m in HS_MARKERS)
        if is_hs and "regress.py" not in cmd:
            out.append({"pid": int(proc.name), "cmd": cmd[:240]})
    return sorted(out, key=lambda p: p["pid"])


def env_for(source: Path) -> dict:
    env = dict(os.environ)
    paths = [str(source)] + [str(p) for p in sorted((source / "methods").glob("*")) if p.is_dir()]
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def run(argv: list, source: Path, timeout: int) -> dict:
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, cwd=source, env=env_for(source), capture_output=True, text=True, timeout=timeout)
        tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
        return {"rc": p.returncode, "seconds": round(time.monotonic() - t0, 1), "tail": tail}
    except subprocess.TimeoutExpired:
        return {"rc": -1, "seconds": timeout, "tail": [f"timeout after {timeout}s"]}


def methods(source: Path) -> list:
    r = subprocess.run([str(PYTHON), "-m", "harnessstudio.cli", "list"], cwd=source, env=env_for(source),
                       capture_output=True, text=True, timeout=300)
    names = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--inflight", action="store_true", help="only list running hs processes")
    ap.add_argument("--timeout", type=int, default=900, help="per tier-1 run, seconds")
    args = ap.parse_args(argv)
    running = inflight()
    if args.inflight:
        print(json.dumps({"inflight": running}, indent=1))
        return 0
    source = args.source.resolve()
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    report: dict = {"source": str(source), "git_head": head.stdout.strip(), "python": str(PYTHON),
                    "inflight_at_start": running, "pytest": run([str(PYTHON), "-m", "pytest", "-q", "tests"],
                                                                source, 1800), "methods": {}}
    for name in methods(source):
        t0 = run([str(PYTHON), "-m", "harnessstudio.cli", "verify", name, "--tier", "0"], source, 300)
        t1 = run([str(PYTHON), "-m", "harnessstudio.cli", "verify", name], source, args.timeout)
        report["methods"][name] = {"tier0": t0, "tier1": t1,
                                   "ok": t0["rc"] == 0 and t1["rc"] == 0 and any("PASS" in x for x in t1["tail"])}
    report["ok"] = report["pytest"]["rc"] == 0 and bool(report["methods"]) and \
        all(m["ok"] for m in report["methods"].values())
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    print(json.dumps({"ok": report["ok"], "pytest_rc": report["pytest"]["rc"],
                      "methods": {k: v["ok"] for k, v in report["methods"].items()},
                      "inflight_at_start": len(running), "out": str(args.out) if args.out else None}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
