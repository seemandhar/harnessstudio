"""Endpoint registry: loads, holds no secrets, pins build valid OperatorPins, banned/unknown/unwired refused."""
import re

import pytest

from harnessstudio import endpoints as E
from harnessstudio.core.method import OperatorPin


def test_registry_loads_and_has_core_providers():
    data = E.load()
    assert {"nrp", "openai", "anthropic"} <= set(data["providers"])
    assert "qwen3-vl" in data["providers"]["nrp"]["models"]
    assert "gpt-6.1-sol" in data["providers"]["openai"]["models"]


def test_registry_holds_no_credentials():
    text = E.REGISTRY.read_text()
    assert not re.search(r"sk-[A-Za-z0-9_-]{16,}", text)
    assert "Bearer " not in text
    for prov in E.load()["providers"].values():
        env = prov.get("api_key_env")
        assert env is None or re.fullmatch(r"[A-Z][A-Z0-9_]*", env)


def test_pins_build_operator_pins():
    qwen = E.operator_entry("nrp", "qwen3-vl", "planner", original_model="Qwen3-VL-32B-Instruct", temperature=0.01)
    pin = OperatorPin(**qwen)
    assert pin.base_url == "nrp" and pin.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    sol = E.operator_entry("openai", "gpt-6.1-sol", "reviewer")
    assert sol["temperature"] is None
    OperatorPin(**sol)


@pytest.mark.parametrize("provider,model", [("openai", "gpt-6-astra"), ("nrp", "Astra-x")])
def test_astra_refused_without_user_confirmation(provider, model):
    with pytest.raises(ValueError, match="confirmation"):
        E.operator_entry(provider, model, "x")
    with pytest.raises(ValueError, match="confirmation"):
        E.operator_entry(provider, model, "x", user_confirmation="   ")


def test_astra_pin_with_recorded_confirmation():
    entry = E.operator_entry("openai", "gpt-6-astra", "planner",
                             user_confirmation="user said yes 2026-10-31 for run X (20 calls)")
    assert entry["temperature"] is None and entry["user_confirmation"].startswith("user said yes")
    OperatorPin(**entry)   # the core accepts a confirmed Astra pin


def test_unknown_and_unwired_refused():
    with pytest.raises(KeyError):
        E.operator_entry("nrp", "not-a-model", "x")
    with pytest.raises(KeyError):
        E.operator_entry("nope", "qwen3-vl", "x")
    with pytest.raises(ValueError):   # local gateways are listed but are not harness providers
        E.operator_entry("local_gateways", "anything", "x")


def test_claude_pins_build_operator_pins():
    entry = E.operator_entry("anthropic", "claude-sonnet-5-5", "planner", original_model="claude-opus-4-8")
    assert entry["base_url"] == "anthropic" and entry["temperature"] is None
    assert entry["api_key_env"] == "HS_ANTHROPIC_API_KEY"     # never ANTHROPIC_API_KEY (Claude Code's own auth)
    OperatorPin(**entry)
