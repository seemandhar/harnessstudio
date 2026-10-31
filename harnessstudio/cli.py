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
    from vla_eval.api import run, serve_background
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
    args.prepared = None
    server = _server(args, collect_arm, development=True, experience_keys=keys)
    overrides = json.loads(args.benchmark_overrides) if args.benchmark_overrides else None
    with serve_background(server, host=args.host, port=args.port) as handle:
        url = f"ws://{args.connect_host}:{handle.port}"
        print(f"[hs prepare] arm={collect_arm} harness={url} policy={args.policy_url}", flush=True)
        run(args.config, server_url=url, output_dir=out / "vla_eval", pull=args.pull,
            benchmark_overrides=overrides, docker=None if not args.no_docker else False)
        server.wait_idle()
    (out / "dev_records.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in server.records))
    prepared = build_prepared(config, server.records, out)
    print(json.dumps({k: prepared[k] for k in ("artifacts", "dev_episodes", "dev_valid", "dev_successes")}, indent=2))
    return 0


def _cmd_serve(args) -> int:
    from vla_eval.model_servers.serve import serve
    serve(_server(args, args.arm), host=args.host, port=args.port)
    return 0


def _cmd_eval(args) -> int:
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
            run(args.config, server_url=url, output_dir=out / arm / "vla_eval", pull=args.pull,
                benchmark_overrides=overrides, docker=None if not args.no_docker else False)
            server.wait_idle()
        records[arm] = server.records
        (out / arm / "records.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in server.records))
    summary = summarize(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hs", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="installed methods")
    p.add_argument("--category", choices=sorted(CATEGORIES))
    p.set_defaults(fn=_cmd_list)

    p = sub.add_parser("verify", help="tier 0 (static) or tier 1 (fake stack) checks")
    p.add_argument("method")
    p.add_argument("--tier", type=int, choices=(0, 1), default=1)
    p.add_argument("--profile", default=None, help="fake-t0 or fake-t1 (default: from the method's needs)")
    p.set_defaults(fn=_cmd_verify)

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
