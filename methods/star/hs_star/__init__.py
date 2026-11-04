"""STAR (arXiv:2610.41009v1) as a HarnessStudio v2 slot method (DRAFT, runs/star-port-20261104/method; not installed).

STAR changes the policy's forward pass inside the policy server, in one pass: a proprioception-modulated EMA over
the residual updates of the last third of the decoder layers (hs_star.star_core, ported from the authors' pi0.5
code). The harness side is SCALE-like: one sample per decision (K = 1) and the `sampler` slot naming the server-side
hook; the derived null switches it off, so baseline and null both decode natively (null == baseline bitwise; STAR
adds no policy call, only elementwise work inside the same forwards).

Needs the gated extension EXT openvla-residual-hook (audit/port_design.md section 3): a policy-server mode that
honours this sampler on the simpler_pcd_authors decode, and a SimplerEnv benchmark that sends `states`.
This module is imported by the policy server to reach hs_star.server_hook, so nothing heavy is imported here.
"""
from __future__ import annotations

from pathlib import Path

from .server_hook import DEFAULTS, StarResidualHook  # noqa: F401  (factory "hs_star.server_hook:StarResidualHook")


class SingleSample:
    """Proposer: one seeded sample per decision (STAR is single-pass)."""

    def __init__(self, num_samples: int = 1) -> None:
        if num_samples != 1:
            raise ValueError("STAR draws exactly one sample per decision")
        self.num_samples = 1


def config():
    from harnessstudio.core.card import config_from_card
    from harnessstudio.core.method import ModuleRef

    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "proposer": ModuleRef("hs_star:SingleSample", {"num_samples": 1}),
        "sampler": ModuleRef("hs_star.server_hook:StarResidualHook", dict(DEFAULTS)),
    })


def config_shadow():
    """Overhead/shadow arm (SETTING.md section 2 'null'): the same server-side STAR computation (beta, Delta, m in
    every mixed layer, same hooks) with nothing written back, so its actions must equal the baseline bitwise. Run as
    the candidate arm of this entry point and paired with baseline/STAR by case key. The framework's own derived null
    (sampler off) of `star` is byte-identical to baseline and is a determinism control only."""
    from harnessstudio.core.card import config_from_card
    from harnessstudio.core.method import ModuleRef

    params = dict(DEFAULTS, shadow=True)
    return config_from_card(Path(__file__).with_name("paper_shadow.yaml"), slots={
        "proposer": ModuleRef("hs_star:SingleSample", {"num_samples": 1}),
        "sampler": ModuleRef("hs_star.server_hook:StarResidualHook", params),
    })
