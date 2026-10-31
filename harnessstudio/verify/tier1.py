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
         episodes: int = 2, n_tasks: int = 2, horizon: int = 12) -> list[dict]:
    profile = load_profile(profile_name)
    policy = FakePolicyServer(chunking="harness", seeded=profile.seeded)
    with serve(policy) as ph:
        harness = HarnessServer(config, policy_url=ph.url, contract=load_contract("fake"), arm=arm, profile=profile)
        with serve(harness) as hh:
            run_fake(hh.url, n_tasks=n_tasks, episodes=episodes, horizon=horizon, task_order=task_order)
    return harness.records


def _by_case(records: list[dict]) -> dict:
    return {(r["task"], r["episode_idx"]): r for r in records}


def default_profile(config: MethodConfig) -> str:
    return "fake-t0" if set(config.required) <= set(load_profile("fake-t0").capabilities) else "fake-t1"


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

    arms = {arm: _run(config, arm, profile) for arm in Arm}
    for arm, recs in arms.items():
        bad = [r["invalid_reason"] for r in recs if not r["valid"]]
        report.add(f"{arm.value} episodes valid", not bad and len(recs) == 4, "; ".join(map(str, bad[:2])))

    base = _by_case(arms[Arm.BASELINE])
    mism = [k for k, r in _by_case(arms[Arm.NULL]).items() if r["action_sha256"] != base.get(k, {}).get("action_sha256")]
    report.add(f"null ({config.null}) reproduces baseline actions", not mism and config.null != "custom",
               f"{len(mism)} cases differ" if mism else ("custom null: compare by hand" if config.null == "custom" else ""))

    null_cost = sum(r["accounting"]["method_policy_samples"] + r["accounting"]["shadow_policy_samples"] for r in arms[Arm.NULL])
    cand_cost = sum(r["accounting"]["method_policy_samples"] for r in arms[Arm.CANDIDATE])
    # Informational: executors that cut chunks make the candidate re-plan more often than the null.
    report.add("policy compute (informational)", True,
               f"null {null_cost} vs candidate {cand_cost} samples"
               + (f" (ratio {cand_cost / null_cost:.2f})" if null_cost else ""))

    again = _by_case(_run(config, Arm.CANDIDATE, profile))
    first = _by_case(arms[Arm.CANDIDATE])
    report.add("candidate is deterministic", all(again[k]["action_sha256"] == r["action_sha256"] for k, r in first.items()))

    reordered = _by_case(_run(config, Arm.CANDIDATE, profile, task_order="reverse"))
    report.add("case-order independent", all(reordered[k]["action_sha256"] == r["action_sha256"] for k, r in first.items()))

    cap = config.budgets.policy_calls
    if cap is not None:
        worst = max(r["accounting"]["method_policy_samples"] for r in arms[Arm.CANDIDATE])
        report.add("policy budget respected", worst <= cap, f"max {worst} of {cap}")

    if module:
        proc = subprocess.run([sys.executable, "-I", "-c", f"import {module}"], capture_output=True, text=True)
        report.add("imports in a clean interpreter", proc.returncode == 0, proc.stderr.strip()[-300:])
    return report
