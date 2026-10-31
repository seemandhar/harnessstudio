"""hs-aac: Adaptive Action Chunking (AAC, arXiv:2604.04161v2, CVPR 2026) as a HarnessStudio v2 method.

N=20 seeded samples per decision (proposer); the executor measures the per-step action entropy of the
N samples and the motion of sample 0, and runs sample 0 for k = max(motion floor, entropy elbow) steps.
The algorithm lives in hs_aac.aac (numpy only); identity and provenance in paper.yaml.
"""
from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

from .aac import (AACEntropyExecutor, AACPool, NUM_SAMPLES, PARAMS, action_magnitude, bernoulli_entropy,
                  compose_rotations, entropy_breakdown, find_first_idx, gaussian_entropy, select_chunk_size)

__all__ = ["AACEntropyExecutor", "AACPool", "NUM_SAMPLES", "PARAMS", "action_magnitude", "bernoulli_entropy",
           "compose_rotations", "config", "entropy_breakdown", "find_first_idx", "gaussian_entropy",
           "select_chunk_size"]


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "proposer": ModuleRef("hs_aac:AACPool", {"num_samples": NUM_SAMPLES}),
        "executor": ModuleRef("hs_aac:AACEntropyExecutor", PARAMS),
    })
