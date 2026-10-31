"""Endpoint registry for method packages: which model APIs and local services exist, and how to pin them.

Data lives in `endpoints.yaml` next to this file (human guide: docs/ENDPOINTS.md). The registry stores only
environment-variable NAMES, never credentials.

    python -m harnessstudio.endpoints list               # providers, models, local services
    python -m harnessstudio.endpoints check              # live: GET /models per provider, /health per local service
    python -m harnessstudio.endpoints pin nrp qwen3-vl --name planner   # paper.yaml operators entry

`operator_entry()` returns the dict for a paper card's `operators:` list (the fields of OperatorPin). It refuses models
the registry does not list and providers HarnessStudio cannot call yet. Astra models are confirm-before-use (user
rule 2026-10-31): they need `user_confirmation` = a record of the user's explicit yes for this use (who, when, scope);
without it the pin is refused. Never fill it in yourself: ask the user first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import yaml

REGISTRY = Path(__file__).with_name("endpoints.yaml")
CONFIRM_REQUIRED = re.compile(r"astra", re.IGNORECASE)   # ask the user before every use (2026-10-31)
CALLABLE_KINDS = {"openai_chat"}          # what harnessstudio.tools.llm can call today


def load(path: Optional[Path] = None) -> dict:
    data = yaml.safe_load(Path(path or REGISTRY).read_text())
    if data.get("schema") != "harnessstudio-endpoints-v1":
        raise ValueError("unknown endpoint registry schema")
    return data


def operator_entry(provider: str, model: str, name: str, original_model: str = "",
                   data: Optional[dict] = None, user_confirmation: str = "", **overrides: Any) -> dict:
    """A paper.yaml `operators` entry for one pinned model. Overrides (temperature, seed, extra_body, ...) win."""
    confirm = bool(CONFIRM_REQUIRED.search(model))
    if confirm and not str(user_confirmation).strip():
        raise ValueError(f"{model!r} needs the user's explicit confirmation before any use (Astra rule 2026-10-31): "
                         "ask the user, then pass user_confirmation='who/when/scope of the yes'")
    data = data or load()
    prov = (data.get("providers") or {}).get(provider)
    if prov is None:
        raise KeyError(f"unknown provider {provider!r}; known: {sorted(data.get('providers') or {})}")
    if prov.get("kind") not in CALLABLE_KINDS:
        raise ValueError(f"provider {provider!r} is {prov.get('kind')!r}: not callable by the LLM tool yet "
                         f"({prov.get('gap', 'see endpoints.yaml')})")
    if model in (prov.get("forbidden") or []):
        raise ValueError(f"{model!r} is forbidden on {provider}")
    info = (prov.get("models") or {}).get(model) or (prov.get("confirm_required") or {}).get(model)
    if info is None:
        raise KeyError(f"{model!r} is not in the registry for {provider}; probe it (endpoints check) and add it first")
    entry: dict = {"name": name, "model": model, "base_url": prov.get("preset") or prov["base_url"],
                   "api_key_env": prov["api_key_env"], "original_model": original_model}
    if info.get("recommended_extra_body"):
        entry["extra_body"] = dict(info["recommended_extra_body"])
    entry.update(info.get("pin_overrides") or {})
    entry.update(overrides)
    if confirm:
        entry["user_confirmation"] = str(user_confirmation).strip()
    return entry


def _get(url: str, key_env: Optional[str], timeout: float = 20.0) -> tuple[bool, Any]:
    headers = {}
    if key_env:
        key = os.environ.get(key_env, "")
        if not key:
            return False, f"{key_env} not set"
        headers["Authorization"] = "Bearer " + key
    if "api.anthropic.com" in url:
        from .tools.llm import ANTHROPIC_VERSION
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            body = r.read().decode(errors="replace")
        try:
            return True, json.loads(body)
        except ValueError:
            return True, body[:200]
    except urllib.error.HTTPError as exc:   # never echo request headers
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"


def check(provider: Optional[str] = None, data: Optional[dict] = None) -> dict:
    """Live availability. Reports booleans and model ids only; never prints credentials."""
    data = data or load()
    out: dict = {"providers": {}, "local_services": {}}
    for name, prov in (data.get("providers") or {}).items():
        if provider and name != provider:
            continue
        if prov.get("kind") != "openai_chat":
            out["providers"][name] = {"kind": prov.get("kind"), "status": prov.get("status", "not probed")}
            continue
        ok, body = _get(prov["base_url"].rstrip("/") + "/models", prov.get("api_key_env"))
        listed = {m.get("id") for m in body.get("data", [])} if ok and isinstance(body, dict) else set()
        registry = set((prov.get("models") or {}))
        out["providers"][name] = {"reachable": ok, "error": None if ok else body,
                                  "registry_models_listed": sorted(registry & listed),
                                  "registry_models_missing": sorted(registry - listed) if ok else [],
                                  "unregistered_listed": sorted(m for m in listed - registry
                                                                if not CONFIRM_REQUIRED.search(str(m)))[:40],
                                  "confirm_required_listed": sorted(set(prov.get("confirm_required") or {}) & listed)}
    if provider is None:
        for group, services in (data.get("local_services") or {}).items():
            for sname, svc in (services or {}).items():
                url = svc.get("health") or svc.get("url", "")
                if group == "tools" and url.startswith("http"):
                    url = url.rstrip("/") + "/health"
                if not url.startswith("http"):
                    out["local_services"][sname] = {"probed": False}
                    continue
                ok, _ = _get(url, None, timeout=3.0)
                out["local_services"][sname] = {"up": ok}
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harnessstudio.endpoints", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("check")
    c.add_argument("--provider", default=None)
    p = sub.add_parser("pin")
    p.add_argument("provider")
    p.add_argument("model")
    p.add_argument("--name", required=True)
    p.add_argument("--original-model", default="")
    p.add_argument("--user-confirmation", default="",
                   help="required for Astra models: record of the user's explicit yes (who/when/scope)")
    args = ap.parse_args(argv)
    data = load()
    if args.cmd == "list":
        summary = {name: {"kind": prov.get("kind"), "role": prov.get("role"),
                          "models": sorted(prov.get("models") or {}) or sorted(prov.get("services") or {})}
                   for name, prov in data["providers"].items()}
        summary["local_services"] = {g: sorted(s) for g, s in (data.get("local_services") or {}).items()}
        print(json.dumps(summary, indent=1))
    elif args.cmd == "check":
        print(json.dumps(check(args.provider, data), indent=1))
    else:
        print(yaml.safe_dump([operator_entry(args.provider, args.model, args.name, args.original_model, data,
                                             user_confirmation=args.user_confirmation)], sort_keys=False).rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
