"""Minimal example method package (see docs/writing-a-method.md)."""
from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef


class Pool:
    def __init__(self, num_samples: int = 2):
        self.num_samples = num_samples


class First:
    def select(self, chunks, extras, obs) -> int:
        return 0


def config():
    return config_from_card(Path(__file__).with_name("paper.yaml"), slots={
        "proposer": ModuleRef("hs_first_sample:Pool", {"num_samples": 2}),
        "selector": ModuleRef("hs_first_sample:First"),
    })
