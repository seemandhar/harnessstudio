"""SCALE (arXiv:2602.04208, ICML 2026) as a HarnessStudio slot method.

SCALE changes how an autoregressive VLA decodes, inside the policy server, in one forward pass:
the per-token sampling temperature and the vision-encoder attention temperature both follow the model's
self-uncertainty (hs_scale.decoder.ScaleDecoder). The harness side is trivial: one sample per decision
(K = 1) and the `sampler` slot naming the decoder, which the derived null switches off (baseline and null
decode greedily, the OpenVLA default; the candidate decodes with SCALE).

This module is imported by the policy server to reach hs_scale.decoder, so nothing heavy is imported
at module level.
"""
from __future__ import annotations

from pathlib import Path

from .decoder import DEFAULTS, ScaleDecoder  # noqa: F401  (factory "hs_scale.decoder:ScaleDecoder")


class SingleSample:
    """Proposer: one seeded sample per decision (SCALE needs no candidates, no verifier)."""

    def __init__(self, num_samples: int = 1) -> None:
        if num_samples != 1:
            raise ValueError("SCALE draws exactly one sample per decision")
        self.num_samples = 1


def config():
    from harnessstudio.core.card import config_from_card
    from harnessstudio.core.method import ModuleRef

    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "proposer": ModuleRef("hs_scale:SingleSample", {"num_samples": 1}),
        "sampler": ModuleRef("hs_scale.decoder:ScaleDecoder", dict(DEFAULTS)),
    })
