import unittest

from harnessstudio.core.capabilities import caps
from harnessstudio.core.errors import RegistrationError
from harnessstudio.core.method import ArtifactPin, Fidelity, MethodConfig, ModuleRef, validate


def make(**overrides):
    base = dict(name="demo", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                substitutions=(), required=caps("action_chunk"), slots={})
    base.update(overrides)
    return MethodConfig(**base)


class MethodTest(unittest.TestCase):
    def test_minimal_valid(self):
        self.assertEqual(validate(make()).name, "demo")

    def test_trainable_parameters_refused(self):
        with self.assertRaises(RegistrationError):
            validate(make(trainable_parameters=1))

    def test_tf_variant_needs_substitution(self):
        with self.assertRaises(RegistrationError):
            validate(make(fidelity=Fidelity.TF_VARIANT))
        validate(make(fidelity=Fidelity.TF_VARIANT, substitutions=("trained X -> rule Y",)))

    def test_controller_refused_in_m1(self):
        with self.assertRaises(RegistrationError):
            validate(make(slots={"controller": ModuleRef("os:getcwd")}))

    def test_selector_needs_proposer_with_k_ge_2(self):
        sel = ModuleRef("os:getcwd")
        with self.assertRaises(RegistrationError):
            validate(make(slots={"selector": sel}))
        with self.assertRaises(RegistrationError):
            validate(make(slots={"selector": sel, "proposer": ModuleRef("os:getcwd", {"num_samples": 1})}))
        validate(make(slots={"selector": sel, "proposer": ModuleRef("os:getcwd", {"num_samples": 2})}))

    def test_executor_declares_min_k_and_per_step(self):
        with self.assertRaises(RegistrationError):
            validate(make(slots={"executor": ModuleRef("os:getcwd", {"min_k": 4})}))
        with self.assertRaises(RegistrationError):
            validate(make(slots={"executor": ModuleRef("os:getcwd", {"min_k": 0, "per_step": False})}))
        validate(make(slots={"executor": ModuleRef("os:getcwd", {"min_k": 4, "per_step": False})}))

    def test_params_must_be_json(self):
        with self.assertRaises(RegistrationError):
            ModuleRef("os:getcwd", {"bad": object()})

    def test_bad_factory_string(self):
        with self.assertRaises(RegistrationError):
            ModuleRef("no_colon")

    def test_artifact_pin_needs_hash_and_license(self):
        with self.assertRaises(RegistrationError):
            ArtifactPin("w", "/x", "abc", "MIT")
        with self.assertRaises(RegistrationError):
            ArtifactPin("w", "/x", "a" * 64, "")

    def test_name_format(self):
        with self.assertRaises(RegistrationError):
            validate(make(name="Bad Name"))

    def test_to_json_roundtrip_is_stable(self):
        m = make(slots={"proposer": ModuleRef("os:getcwd", {"num_samples": 2, "b": [1, 2]}),
                        "selector": ModuleRef("os:getcwd")})
        self.assertEqual(m.to_json(), make(slots=dict(m.slots)).to_json())
        self.assertEqual(m.to_json()["slots"]["proposer"]["params"], {"b": [1, 2], "num_samples": 2})
