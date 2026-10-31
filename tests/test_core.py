from __future__ import annotations

import pytest

from harnessstudio.contracts import Contract, load_contract
from harnessstudio.core.errors import ContractError, RegistrationError
from harnessstudio.core.method import Budgets, Fidelity, MethodConfig, ModuleRef, OperatorPin, validate
from harnessstudio.core.noise import crn_seed
from harnessstudio.core.registry import load

T1 = frozenset({"act", "chunk", "seeded_sampling", "isolated_streams"})


def cfg(**kw):
    base = dict(name="m", version="1", paper="none", fidelity=Fidelity.FAITHFUL)
    base.update(kw)
    return MethodConfig(**base)


def test_identity_registered():
    assert load("identity").name == "identity"


@pytest.mark.parametrize("kw", [
    dict(fidelity=Fidelity.TF_VARIANT),                                   # needs substitutions
    dict(substitutions=("x -> y",)),                                      # faithful must not list them
    dict(categories=("not_a_category",)),
    dict(required=frozenset({"telepathy"})),
    dict(slots={"bogus": ModuleRef("a.b:C")}),
    dict(required=T1, slots={"proposer": ModuleRef("a.b:C", {"num_samples": 1}), "selector": ModuleRef("a.b:D")}),
    dict(slots={"proposer": ModuleRef("a.b:C", {"num_samples": 4})}),     # K>1 without seeded capability
    dict(controller=ModuleRef("a.b:C")),                                  # controller needs shadow/custom null
    dict(controller=ModuleRef("a.b:C"), null="shadow"),                   # and a policy_calls cap
    dict(null="shadow"),                                                  # shadow without controller
    dict(trainable_parameters=5),
])
def test_registration_rules(kw):
    with pytest.raises(RegistrationError):
        validate(cfg(**kw))


def test_valid_controller_and_adapted():
    validate(cfg(controller=ModuleRef("a.b:C"), null="shadow", budgets=Budgets(policy_calls=10)))
    validate(cfg(fidelity=Fidelity.ADAPTED, substitutions=("Claude Code -> NRP Qwen3",)))


def test_astra_operator_needs_user_confirmation():
    # User rule 2026-10-31: Astra is confirm-before-use (banned 2026-10-20 before that).
    with pytest.raises(RegistrationError):
        OperatorPin(name="planner", model="gpt-6-astra", base_url="nrp", api_key_env="K")
    with pytest.raises(RegistrationError):
        OperatorPin(name="planner", model="gpt-6-astra", base_url="nrp", api_key_env="K", user_confirmation="  ")
    pin = OperatorPin(name="planner", model="gpt-6-astra", base_url="openai", api_key_env="K",
                      user_confirmation="user yes 2026-10-31, run X, <=20 calls")
    from harnessstudio.tools.llm import build_operators
    assert build_operators([pin])["planner"].confirmed


def test_empty_user_confirmation_keeps_fingerprints():
    from harnessstudio.core.registry import load
    cfg = load("harness-vla")
    assert all("user_confirmation" not in o for o in cfg.to_json()["operators"])


def test_crn_seed_is_stable_and_distinct():
    base = ("hs", "libero", "task", 0, 0, 0)
    assert crn_seed(*base) == crn_seed(*base)
    variants = {crn_seed(*(base[:i] + (base[i] + 1 if isinstance(base[i], int) else base[i] + "x",) + base[i + 1:]))
                for i in range(len(base))}
    assert len(variants) == len(base) and crn_seed(*base) not in variants


def test_contract_filtering_and_horizon():
    c = load_contract("libero")
    obs = {"images": {"agentview": 1}, "task_description": "pick", "sim_state": 3}
    assert c.filter_public(obs) == {"images": {"agentview": 1}, "task_description": "pick"}
    assert c.horizon({"name": "x", "suite": "libero_spatial"}) == 220
    with pytest.raises(ContractError):
        c.filter_public({"images": {}})
    with pytest.raises(ContractError):
        Contract(name="bad", public_keys=("images", "success"), instruction_key="images")


def test_all_shipped_contracts_load():
    for name in ("fake", "libero", "libero_pro", "robocasa365"):
        assert load_contract(name).name == name


def test_libero_pro_perturbed_suites_use_base_horizon():
    c = load_contract("libero_pro")
    assert c.horizon({"suite": "libero_spatial_swap"}) == 220
    assert c.horizon({"suite": "libero_object_with_mug"}) == 280
    assert c.horizon({"suite": "libero_10_task"}) == 520
    assert c.horizon({"suite": "libero_goal"}) == 300


def test_attribution_classes():
    from harnessstudio.eval.attribution import classify
    base = {"action_sha256": list("aaaaa"), "obs_sha256": list("ooooo")}
    assert classify(base, dict(base))["class"] == "M"
    noisy = {"action_sha256": list("aaaaa"), "obs_sha256": list("ooxoo")}      # renderer noise, same actions
    assert classify(base, noisy)["class"] == "M" and classify(base, noisy)["obs_noise_steps"] == [2]
    branch = {"action_sha256": list("aaabb"), "obs_sha256": list("ooxxx")}     # obs first, then actions
    assert classify(base, branch)["class"] == "R"
    leak = {"action_sha256": list("aabaa"), "obs_sha256": list("ooooo")}       # action differs, same obs
    assert classify(base, leak)["class"] == "X"


def test_anthropic_preset_and_version_header():
    from harnessstudio.tools.llm import OpenAICompatibleOperator
    op = OpenAICompatibleOperator("claude-sonnet-5-5", "anthropic", "HS_ANTHROPIC_API_KEY", temperature=None)
    assert op.base_url == "https://api.anthropic.com/v1" and op.token_param == "max_tokens"
    h = op._headers("k")
    assert h["anthropic-version"] == "2023-06-01" and h["Authorization"] == "Bearer k"
    assert "anthropic-version" not in OpenAICompatibleOperator("qwen3-vl", "nrp", "NRP_API_KEY")._headers("k")

