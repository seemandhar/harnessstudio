"""`hs`: list, verify, serve and evaluate harness methods."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .core.categories import CATEGORIES
from .core.registry import discover, load


def _cmd_list(args) -> int:
    rows = []
    for name, ep in sorted(discover().items()):
        try:
            c = load(name)
            if args.category and args.category not in c.categories:
                continue
            rows.append((name, c.version, c.fidelity.value, ",".join(c.categories) or "-",
                         ",".join(sorted(c.required)), c.paper))
        except Exception as exc:  # noqa: BLE001
            rows.append((name, "?", "BROKEN", "-", "-", f"{type(exc).__name__}: {exc}"[:80]))
    header = ("method", "version", "fidelity", "categories", "requires", "paper")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)).rstrip())
    return 0


def _cmd_verify(args) -> int:
    if args.tier == 2:
        return _tier2(args)
    config = load(args.method)
    if args.tier == 0:
        from .verify import tier0
        problems = tier0.check(config)
        print(f"tier 0: {config.name}: {'PASS' if not problems else 'FAIL'}")
        for p in problems:
            print(f"  - {p}")
        return 0 if not problems else 1
    from .core.registry import discover as _d
    from .verify import tier1
    module = None
    ep = _d().get(args.method)
    if ep is not None:
        module = ep.value.partition(":")[0]
    report = tier1.run(config, profile=args.profile, module=module)
    print("\n".join(report.lines()))
    return 0 if report.passed else 1


def _server(args, arm: str, **extra):
    from .contracts import load_contract
    from .profiles import load_profile
    from .server.harness_server import HarnessServer
    config = load(args.method)
    out = Path(args.out) / arm if args.out else None
    kwargs = dict(case_namespace=args.case_namespace, **extra)
    if getattr(args, "prepared", None):
        from .eval.prepare import load_prepared
        artifacts, forbidden, _ = load_prepared(args.prepared, config)
        kwargs.update(artifacts=artifacts, forbidden_cases=forbidden)
    return HarnessServer(config, policy_url=args.policy_url, contract=load_contract(args.contract),
                         arm=arm, profile=load_profile(args.profile), out_dir=out, namespace=args.namespace,
                         policy_timeout=args.policy_timeout, **kwargs)


def _cmd_prepare(args) -> int:
    from vla_eval.api import serve_background
    from .eval.prepare import build_prepared, preparer_settings
    config = load(args.method)
    if config.prepare is None:
        print(f"{config.name} has no prepare stage", file=sys.stderr)
        return 1
    if not args.case_namespace:
        print("[hs prepare] no --case-namespace: development cases keep their plain keys and evaluation will "
              "refuse them (right for LIBERO, where episode_idx fixes the scene). Use a namespace only when the "
              "development run uses a different benchmark seed that really changes the scenes.", file=sys.stderr)
    out = Path(args.out)
    collect_arm, keys = preparer_settings(config)
    previous = None
    if args.prepared:  # chained development: start from earlier artifacts, keep their cases forbidden
        from .eval.prepare import load_prepared
        _, _, previous = load_prepared(args.prepared, config)
    server = _server(args, collect_arm, development=True, experience_keys=keys)
    overrides = json.loads(args.benchmark_overrides) if args.benchmark_overrides else None
    with serve_background(server, host=args.host, port=args.port) as handle:
        url = f"ws://{args.connect_host}:{handle.port}"
        print(f"[hs prepare] arm={collect_arm} harness={url} policy={args.policy_url}", flush=True)
        _run_benchmark(_with_timeout(args.config, getattr(args, "action_timeout", None)), url,
                       out / "vla_eval", overrides, args.pull, args.no_docker)
        server.wait_idle()
    (out / "dev_records.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in server.records))
    prepared = build_prepared(config, server.records, out, previous=previous)
    print(json.dumps({k: prepared[k] for k in ("artifacts", "dev_episodes", "dev_valid", "dev_successes")}, indent=2))
    return 0


def _cmd_serve(args) -> int:
    from vla_eval.model_servers.serve import serve
    serve(_server(args, args.arm), host=args.host, port=args.port)
    return 0


def _cmd_eval(args) -> int:
    summary = _run_arms(args)
    print(json.dumps(summary, indent=2))
    return 0


def _with_timeout(config, timeout):
    """The benchmark's per-action timeout (vla-eval server.timeout); slow LLM-reviewed methods need more."""
    if not timeout:
        return config
    from vla_eval.cli.config_loader import load_config
    cfg = load_config(str(config)) if not isinstance(config, dict) else dict(config)
    cfg.setdefault("server", {})["timeout"] = float(timeout)
    return cfg


def _run_benchmark(config, url: str, output_dir: Path, overrides, pull: bool, no_docker: bool) -> None:
    """One benchmark run per arm, always in a fresh process: Docker runs get their own container, and
    in-process (--no-docker) runs get a child interpreter, because some simulators' resets depend on what
    the process ran before (RoboCasa365 renders different first frames after earlier episodes)."""
    import subprocess
    if not no_docker:
        from vla_eval.api import run
        run(config, server_url=url, output_dir=output_dir, pull=pull, benchmark_overrides=overrides, docker=None)
        return
    code = ("import json, sys; from vla_eval.api import run; a = json.loads(sys.argv[1]); "
            "run(a['config'], server_url=a['url'], output_dir=a['out'], benchmark_overrides=a['overrides'], "
            "docker=False)")
    payload = json.dumps({"config": config if isinstance(config, dict) else str(config), "url": url,
                          "out": str(output_dir), "overrides": overrides})
    proc = subprocess.run([sys.executable, "-c", code, payload])
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark run failed with exit code {proc.returncode}")


def _run_arms(args) -> dict:
    from vla_eval.api import run, serve_background
    from .eval.pairing import summarize
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    records = {}
    overrides = json.loads(args.benchmark_overrides) if args.benchmark_overrides else None
    for arm in arms:
        server = _server(args, arm)
        with serve_background(server, host=args.host, port=args.port) as handle:
            url = f"ws://{args.connect_host}:{handle.port}"
            print(f"[hs eval] arm={arm} harness={url} policy={args.policy_url}", flush=True)
            _run_benchmark(_with_timeout(args.config, getattr(args, "action_timeout", None)), url,
                           out / arm / "vla_eval", overrides, args.pull, args.no_docker)
            server.wait_idle()
        records[arm] = server.records
        (out / arm / "records.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in server.records))
    summary = summarize(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _tier2(args) -> int:
    """Tier 2: all three arms on a small real case list; gate on validity and null attribution."""
    for name in ("config", "policy_url", "contract", "profile", "out"):
        if not getattr(args, name, None):
            print(f"tier 2 needs --{name.replace('_', '-')}", file=sys.stderr)
            return 2
    args.arms = "baseline,null,candidate"
    args.prepared = getattr(args, "prepared", None)
    summary = _run_arms(args)
    attr = summary.get("null_attribution", {})
    invalid = {arm: v for arm, v in summary["invalid"].items() if v}
    checks = [("all episodes valid", not invalid, json.dumps(invalid)[:200] if invalid else ""),
              ("null vs baseline has no X", attr.get("X", 1) == 0,
               f"M {attr.get('M')} / R {attr.get('R')} / X {attr.get('X')}"),
              ("at least one paired case", summary["paired_cases"] > 0, f"{summary['paired_cases']} paired")]
    ok = all(c for _, c, _ in checks)
    print(f"tier 2: {args.method}: {'PASS' if ok else 'FAIL'}")
    for name, c, detail in checks:
        print(f"  [{'ok' if c else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    for arm, v in summary["arms"].items():
        print(f"  {arm}: {v['successes']}/{v['episodes']} success, {v['policy_samples']} policy samples, "
              f"{v['wall_seconds']} s")
    return 0 if ok else 1


def _cmd_pair(args) -> int:
    """Pair arms that were run separately (e.g. on different policy servers): ARM=DIR with DIR/records.jsonl."""
    from .eval.pairing import summarize
    records = {}
    for spec in args.arm:
        arm, _, d = spec.partition("=")
        path = Path(d) / "records.jsonl" if Path(d).is_dir() else Path(d)
        # repeated --arm NAME=DIR flags are concatenated (an arm split across policy servers)
        records.setdefault(arm, []).extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    summary = summarize(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    args.dir = str(out)
    return _cmd_report(args)


def _cmd_report(args) -> int:
    path = Path(args.dir) / "summary.json"
    summary = json.loads(path.read_text())
    print(f"{args.dir}: {summary['paired_cases']} paired cases")
    for arm, v in summary["arms"].items():
        vs = v.get("vs_baseline")
        extra = f"  vs baseline: +{vs['wins']} / -{vs['losses']} (p={vs['p']})" if vs else ""
        print(f"  {arm:9s} {v['successes']:4d}/{v['episodes']:<4d} success  {v['policy_samples']:7d} samples  "
              f"{v['llm_calls']:5d} LLM calls  {v['wall_seconds']:8.1f} s{extra}")
    attr = summary.get("null_attribution")
    if attr:
        print(f"  null vs baseline: M {attr['M']} / R {attr['R']} / X {attr['X']}")
        for case, c in attr["per_case"].items():
            if c["class"] != "M":
                print(f"    {c['class']} at step {c['step']}: {case} ({c['reason']})")
    bad = {arm: v for arm, v in summary.get("invalid", {}).items() if v}
    if bad:
        print(f"  invalid episodes: {json.dumps(bad)[:300]}")
    return 0


def _cmd_new(args) -> int:
    from .scaffold import create
    base = create(args.slug, name=args.name or args.slug.replace("_", "-"), category=args.category, kind=args.kind,
                  paper=args.paper, title=args.title or args.slug, llm=args.llm, root=Path(args.root))
    print(f"created {base}\nnext:\n  uv pip install --python .venv/bin/python -e {base}\n"
          f"  hs verify {args.name or args.slug.replace('_', '-')}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hs", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="installed methods")
    p.add_argument("--category", choices=sorted(CATEGORIES))
    p.set_defaults(fn=_cmd_list)

    p = sub.add_parser("verify", help="tier 0 (static), tier 1 (fake stack) or tier 2 (real benchmark) checks")
    p.add_argument("method")
    p.add_argument("--tier", type=int, choices=(0, 1, 2), default=1)
    p.add_argument("--profile", default=None, help="tier 1: fake-t0/t1/t2 (default from needs); tier 2: real profile")
    p.add_argument("--config", default=None, help="tier 2: vla-eval benchmark config (keep it to a few cases)")
    p.add_argument("--policy-url", default=None)
    p.add_argument("--contract", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--benchmark-overrides", default=None)
    p.add_argument("--namespace", default="hs")
    p.add_argument("--case-namespace", default="")
    p.add_argument("--prepared", default=None)
    p.add_argument("--action-timeout", type=float, default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--connect-host", default="localhost")
    p.add_argument("--policy-timeout", type=float, default=600.0)
    p.add_argument("--pull", action="store_true")
    p.add_argument("--no-docker", action="store_true")
    p.set_defaults(fn=_cmd_verify)

    p = sub.add_parser("pair", help="pair separately run arms: --arm baseline=DIR --arm null=DIR ...")
    p.add_argument("--arm", action="append", required=True, help="ARM=DIR (DIR holds records.jsonl)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_pair)

    p = sub.add_parser("report", help="print the paired summary of an hs eval directory")
    p.add_argument("dir")
    p.set_defaults(fn=_cmd_report)

    def serving(p):
        p.add_argument("method")
        p.add_argument("--policy-url", required=True)
        p.add_argument("--contract", required=True)
        p.add_argument("--profile", required=True, help="policy profile name or YAML path")
        p.add_argument("--out", default=None)
        p.add_argument("--namespace", default="hs")
        p.add_argument("--host", default="0.0.0.0")
        p.add_argument("--policy-timeout", type=float, default=600.0)
        p.add_argument("--case-namespace", default="",
                       help="part of every case key; use one per benchmark seed/split (e.g. seed7)")
        p.add_argument("--prepared", default=None, help="prepared.json (or its directory) from hs prepare")
        p.add_argument("--action-timeout", type=float, default=None,
                       help="benchmark per-action timeout in s (vla-eval server.timeout); raise for LLM-heavy methods")

    p = sub.add_parser("serve", help="run one arm's harness server (blocking)")
    serving(p)
    p.add_argument("--arm", default="candidate", choices=("baseline", "null", "candidate"))
    p.add_argument("--port", type=int, default=8200)
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("eval", help="run arms against a vla-eval benchmark config and pair them")
    serving(p)
    p.add_argument("--config", required=True, help="vla-eval benchmark config YAML")
    p.add_argument("--arms", default="baseline,null,candidate")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--connect-host", default="localhost", help="host the benchmark uses to reach the harness")
    p.add_argument("--benchmark-overrides", default=None, help='JSON, e.g. {"episodes_per_task": 2}')
    p.add_argument("--pull", action="store_true")
    p.add_argument("--no-docker", action="store_true")
    p.set_defaults(fn=_cmd_eval)

    p = sub.add_parser("new", help="scaffold a method package under methods/")
    p.add_argument("slug", help="package folder name, [a-z0-9_]+")
    p.add_argument("--name", default=None, help="method (entry point) name; default: slug with dashes")
    p.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    p.add_argument("--kind", default="slot", choices=("slot", "controller"))
    p.add_argument("--paper", default="none", help="e.g. arXiv:2607.08448")
    p.add_argument("--title", default=None)
    p.add_argument("--llm", action="store_true", help="controller with a pinned runtime LLM operator")
    p.add_argument("--root", default="methods")
    p.set_defaults(fn=_cmd_new)

    p = sub.add_parser("prepare", help="development stage: run development cases and pin artifacts")
    serving(p)
    p.add_argument("--config", required=True, help="vla-eval benchmark config YAML (development cases)")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--connect-host", default="localhost")
    p.add_argument("--benchmark-overrides", default=None)
    p.add_argument("--pull", action="store_true")
    p.add_argument("--no-docker", action="store_true")
    p.set_defaults(fn=_cmd_prepare)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
