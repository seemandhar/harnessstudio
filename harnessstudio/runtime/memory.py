"""Frozen, provenance-bound symbolic memory; no online evaluation learning."""
from dataclasses import asdict, dataclass
import re

from .contracts import canonical_hash


@dataclass(frozen=True)
class MemoryCard:
    id: str
    policy_id: str
    checkpoint_identity: str
    instruction: str
    advice: str
    evidence_sha256: str
    split: str = "development"

    def __post_init__(self):
        if self.split != "development":
            raise ValueError("Only development memory can be admitted")
        if not re.fullmatch(r"[0-9a-f]{64}", self.evidence_sha256):
            raise ValueError("Memory must bind immutable development evidence")
        if not all((self.id, self.policy_id, self.checkpoint_identity, self.instruction, self.advice)):
            raise ValueError("Memory card fields cannot be empty")


@dataclass(frozen=True, init=False)
class FrozenMemory:
    cards: tuple
    sha256: str

    def __init__(self, cards=()):
        cards = tuple(cards)
        if len({c.id for c in cards}) != len(cards):
            raise ValueError("Duplicate memory card IDs")
        object.__setattr__(self, "cards", cards)
        object.__setattr__(self, "sha256", canonical_hash([asdict(c) for c in cards]))

    def retrieve(self, instruction, policy_spec, limit=3):
        if type(limit) is not int or limit < 0:
            raise ValueError("Retrieval limit must be nonnegative")
        tokens = set(re.findall(r"\w+", instruction.lower()))
        scored = []
        for card in self.cards:
            if (card.policy_id, card.checkpoint_identity) != (policy_spec.policy_id, policy_spec.checkpoint_identity):
                continue
            other = set(re.findall(r"\w+", card.instruction.lower()))
            score = len(tokens & other) / max(1, len(tokens | other))
            if score > 0:
                scored.append((score, card.id, card))
        return tuple(x[2] for x in sorted(scored, key=lambda x: (-x[0], x[1]))[:limit])
