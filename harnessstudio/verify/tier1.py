"""Tier 1 (spec 8.2): the method runs through real vla-eval, in-process, on the fake benchmark and fake
policy. CPU only, seconds. Checks: valid episodes in every arm, null parity, determinism, budget
bounds, case-order independence and import closure in a clean interpreter.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

from ..contracts import load_contract
from ..core.arms import Arm
from ..core.method import MethodConfig
from ..policy_ext.fake import FakePolicyServer
from ..profiles import load_profile
from ..server.harness_server import HarnessServer
from . import tier0
from .stack import run_fake, serve


@dataclass
class Report:
    method: str
    profile: str
    checks: dict = field(default_factory=dict)   # name -> (passed: bool, detail: str)

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.checks.values())

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks[name] = (bool(ok), detail)

    def lines(self) -> list[str]:
        out = [f"tier 1: {self.method} on {self.profile}: {'PASS' if self.passed else 'FAIL'}"]
        for name, (ok, detail) in self.checks.items():
            out.append(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        return out


def _run(config: MethodConfig, arm: Arm, profile_name: str, task_order: str = "forward",
         episodes: int = 2, n_tasks: int = 2, horizon: Optional[int] = None, task_prefix: str = "fake_task_",
         server_kwargs: Optional[dict] = None) -> list[dict]:
    # A paper card may set tier1: {instruction: "...", horizon: N} to exercise multi-step behavior.
    t1 = dict((config.card or {}).get("tier1") or {})
    instruction = str(t1.get("instruction", "reach the goal of {name}"))
    horizon = int(horizon or t1.get("horizon", 12))
    profile = load_profile(profile_name)
    policy = FakePolicyServer(chunking="harness", seeded=profile.seeded)
    with serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=load_contract("fake"), arm=arm, profile=profile,
                                scripted_llm=True, **(server_kwargs or {}))
        with serve(harness) as hh:
            run_fake(hh.url, n_tasks=n_tasks, episodes=episodes, horizon=horizon, task_order=task_order,
                     instruction=instruction, task_prefix=task_prefix)
            harness.wait_idle()
    return harness.records


def _by_case(records: list[dict]) -> dict:
    return {(r["task"], r["episode_idx"]): r for r in records}


def default_profile(config: MethodConfig) -> str:
    for name in ("fake-t0", "fake-t1", "fake-t2"):
        if set(config.required) <= set(load_profile(name).capabilities):
            return name
    return "fake-t2"


def run(config: MethodConfig, profile: Optional[str] = None, module: Optional[str] = None) -> Report:
    profile = profile or default_profile(config)
    report = Report(config.name, profile)
    problems = tier0.check(config)
    report.add("tier 0 (registration, artifacts, buildable components)", not problems, "; ".join(problems))
    if problems:
        return report
    if not set(config.required) <= set(load_profile(profile).capabilities):
        report.add("capabilities", False, f"{profile} lacks {sorted(set(config.required) - set(load_profile(profile).capabilities))}")
        return report

    extra: dict = {}
    if config.prepare is not None:
        import tempfile
        from pathlib import Path
        from ..eval.prepare import build_prepared, load_prepared, preparer_settings
        collect_arm, keys = preparer_settings(config)
        dev = _run(config, Arm(collect_arm), profile, task_prefix="fake_dev_task_",
                   server_kwargs={"development": True, "experience_keys": keys})
        tmp = Path(tempfile.mkdtemp(prefix="hs-tier1-prepare-"))
        prepared = build_prepared(config, dev, tmp)
        artifacts, forbidden, _ = load_prepared(tmp, config)
        report.add("prepare stage builds pinned artifacts", bool(prepared["artifacts"]),
                   f"{len(prepared['artifacts'])} artifacts from {prepared['dev_episodes']} development episodes")
        extra = {"artifacts": artifacts, "forbidden_cases": forbidden}
        import logging
        logging.disable(logging.CRITICAL)  # the refusals below are expected errors
        try:
            refused = _run(config, Arm.CANDIDATE, profile, task_prefix="fake_dev_task_", server_kwargs=extra)
        finally:
            logging.disable(logging.NOTSET)
        report.add("development cases are refused at evaluation",
                   len(refused) == 4 and all(not r["valid"] for r in refused), f"{len(refused)} refusal records")
    arms = {arm: _run(config, arm, profile, server_kwargs=extra) for arm in Arm}
    for arm, recs in arms.items():
        bad = [r["invalid_reason"] for r in recs if not r["valid"]]
        report.add(f"{arm.value} episodes valid", not bad and len(recs) == 4, "; ".join(map(str, bad[:2])))

    base = _by_case(arms[Arm.BASELINE])
    mism = [k for k, r in _by_case(arms[Arm.NULL]).items() if r["action_sha256"] != base.get(k, {}).get("action_sha256")]
    report.add(f"null ({config.null}) reproduces baseline actions", not mism and config.null != "custom",
               f"{len(mism)} cases differ" if mism else ("custom null: compare by hand" if config.null == "custom" else ""))

    total = lambda recs: sum(r["accounting"]["method_policy_samples"] + r["accounting"]["native_policy_samples"]  # noqa: E731
                             + r["accounting"]["shadow_policy_samples"] for r in recs)
    null_cost, cand_cost = total(arms[Arm.NULL]), total(arms[Arm.CANDIDATE])
    # Informational: executors that cut chunks make the candidate re-plan more often than the null.
    report.add("policy compute (informational)", True,
               f"null {null_cost} vs candidate {cand_cost} samples"
               + (f" (ratio {cand_cost / null_cost:.2f})" if null_cost else ""))

    again = _by_case(_run(config, Arm.CANDIDATE, profile, server_kwargs=extra))
    first = _by_case(arms[Arm.CANDIDATE])
    report.add("candidate is deterministic", all(again[k]["action_sha256"] == r["action_sha256"] for k, r in first.items()))

    reordered = _by_case(_run(config, Arm.CANDIDATE, profile, task_order="reverse", server_kwargs=extra))
    report.add("case-order independent", all(reordered[k]["action_sha256"] == r["action_sha256"] for k, r in first.items()))

    cap = config.budgets.policy_calls
    if cap is not None:
        worst = max(r["accounting"]["method_policy_samples"] for r in arms[Arm.CANDIDATE])
        report.add("policy budget respected", worst <= cap, f"max {worst} of {cap}")

    if module:
        proc = subprocess.run([sys.executable, "-I", "-c", f"import {module}"], capture_output=True, text=True)
        report.add("imports in a clean interpreter", proc.returncode == 0, proc.stderr.strip()[-300:])
    return report
