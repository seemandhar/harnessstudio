"""scale-gtok: SCALE (arXiv:2602.04208v2) with argmax on the gripper token, as a HarnessStudio v2 slot method.

Approved candidate of run scalepp-20261102 (review runs/scalepp-20261102/reviews/proposals-r1-review.json,
DECISION_FIRST_scale_gtok; alias of DOMAIN r1d-h1-gripmode-scale). The method is SCALE's slots with one decoder
class swapped: proposer hs_scale:SingleSample (K = 1), sampler hs_scale_gtok.decoder:ScaleGripperArgmax with SCALE's
paper defaults. The derived null switches the sampler off (native greedy), exactly as for `scale`.

The policy server imports this package to reach hs_scale_gtok.decoder (factory prefix hs_), so nothing heavy is
imported at module level, and hs_scale must be importable there too (methods/scale is a referenced method root).
"""
from __future__ import annotations

from pathlib import Path

from .decoder import DEFAULTS, ScaleGripperArgmax  # noqa: F401  (factory "hs_scale_gtok.decoder:ScaleGripperArgmax")

DECODER_FACTORY = "hs_scale_gtok.decoder:ScaleGripperArgmax"
PROPOSER_FACTORY = "hs_scale:SingleSample"


def config():
    from harnessstudio.core.card import config_from_card
    from harnessstudio.core.method import ModuleRef

    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "proposer": ModuleRef(PROPOSER_FACTORY, {"num_samples": 1}),
        "sampler": ModuleRef(DECODER_FACTORY, dict(DEFAULTS)),
    })
