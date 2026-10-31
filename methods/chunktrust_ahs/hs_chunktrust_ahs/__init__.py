"""hs-chunktrust-ahs: ChunkTrust adaptive horizon selection (arXiv:2609.39754v2) as a HarnessStudio v2 method.

One native sample per decision with the policy's flow-velocity readout (proposer, readouts=["velocity"]);
the executor runs the vendored ChunkTrust AHS on that readout and executes the first k in {4..16} actions.
Components in hs_chunktrust_ahs.ahs; vendored upstream numerics in chunktrust_gr00t_ahs.py and v1's episode
state in horizon.py (both byte-identical, PROVENANCE.json); identity and provenance in paper.yaml.
"""
from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

from .ahs import (AHS_PARAMS, AHS_PARAMS_SHA256, EXECUTOR_PARAMS, MIN_K, V1_COLUMN_ORDER, AHSExecutor,
                  VelocityReadoutProposer)

__all__ = ["AHS_PARAMS", "AHS_PARAMS_SHA256", "AHSExecutor", "EXECUTOR_PARAMS", "MIN_K", "V1_COLUMN_ORDER",
           "VelocityReadoutProposer", "config"]


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "proposer": ModuleRef("hs_chunktrust_ahs:VelocityReadoutProposer",
                              {"num_samples": 1, "readouts": ["velocity"]}),
        "executor": ModuleRef("hs_chunktrust_ahs:AHSExecutor", EXECUTOR_PARAMS),
    })
