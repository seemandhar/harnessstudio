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


def test_astra_operator_refused():
    with pytest.raises(RegistrationError):
        OperatorPin(name="planner", model="gpt-6-astra", base_url="nrp", api_key_env="K")


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
