"""Paper cards (paper.yaml, spec 6.2) and MethodConfig construction from a card."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .errors import RegistrationError
from .method import ArtifactPin, Budgets, Fidelity, MethodConfig, ModuleRef, OperatorPin, validate

REQUIRED_FIELDS = ("name", "version", "paper", "fidelity")


def load_card(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        card = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RegistrationError(f"Cannot read paper card {path}: {exc}") from exc
    if not isinstance(card, dict):
        raise RegistrationError(f"Paper card {path} must be a mapping")
    missing = [f for f in REQUIRED_FIELDS if f not in card]
    if missing:
        raise RegistrationError(f"Paper card {path} is missing {missing}")
    return card


def _paper_ref(paper: Any) -> str:
    if isinstance(paper, str):
        return paper
    if isinstance(paper, Mapping) and paper.get("id"):
        return str(paper["id"])
    raise RegistrationError("paper must be a string or a mapping with an 'id'")


def _categories(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        primary = value.get("primary")
        secondary = value.get("secondary") or []
        return tuple(([primary] if primary else []) + list(secondary))
    return tuple(value)


def config_from_card(card_path: str | Path, *, slots: Optional[Mapping[str, ModuleRef]] = None,
                     controller: Optional[ModuleRef] = None, description: str = "") -> MethodConfig:
    """Build and validate a MethodConfig: identity and provenance from the card, wiring from code."""
    card = load_card(card_path)
    requires = card.get("requires") or {}
    budgets = card.get("budgets") or {}
    try:
        config = MethodConfig(
            name=str(card["name"]), version=str(card["version"]), paper=_paper_ref(card["paper"]),
            fidelity=Fidelity(card["fidelity"]),
            substitutions=tuple(card.get("substitutions") or ()),
            categories=_categories(card.get("categories")),
            required=frozenset(requires.get("capabilities") or ["act"]),
            slots=dict(slots or {}), controller=controller,
            null=str(card.get("null", "derived" if controller is None else "shadow")),
            budgets=Budgets(**budgets),
            artifacts=tuple(ArtifactPin(**a) for a in card.get("artifacts") or ()),
            operators=tuple(OperatorPin(**o) for o in card.get("operators") or ()),
            description=description or str(card.get("description", "")),
            card=card,
        )
    except (TypeError, ValueError) as exc:
        raise RegistrationError(f"Paper card {card_path}: {exc}") from exc
    return validate(config)
