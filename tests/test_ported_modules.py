"""Ported Unified modules work as HarnessStudio slot modules (CPU, synthetic)."""
import unittest

import tests  # noqa: F401
from harnessstudio.core.arms import Arm
from harnessstudio.core.capabilities import caps, unified_names
from harnessstudio.core.method import Fidelity, MethodConfig, ModuleRef
from harnessstudio.core.pipeline import EpisodeSpec, run_episode
from harnessstudio.verify import fixtures as fx


class Chunk:
    def __init__(self, actions):
        self.actions = actions


class Obs:
    def __init__(self, values):
        self.values = values


class FakeNRP:
    def complete(self, model, messages, *, max_tokens, expected_returned_model):
        return {"catalog_sha256": "cat", "usage": {"total_tokens": 7},
                "message": {"content": "carefully pick up the synthetic cube"}}


def fake_client():
    return FakeNRP()


def run(slots, arm=Arm.CANDIDATE):
    from harnessstudio.runtime import PolicySpec
    m = MethodConfig(name="t", version="1", paper="none", fidelity=Fidelity.FAITHFUL, substitutions=(),
                     required=caps("action_chunk"), slots=slots)
    env = fx.SyntheticEnv(stall_steps=frozenset(range(20, 40)))
    spec = EpisodeSpec("c", 64, fx.CHUNK, fx.ACTION_DIM, 20.0, "synthetic", fx.PUBLIC_KEYS, "ns",
                       fx.instruction_of, fx.with_instruction, fx.input_hashes, fx.action_hashes,
                       action_layout=fx.ACTION_LAYOUT, proprio_eef=fx.PROPRIO_EEF)
    caps_all = caps("action_chunk", "seeded_sampling", "isolated_streams", "proprio", "variable_execute")
    return run_episode(m, arm, spec, env.observation(), env, fx.SyntheticClient(),
                       PolicySpec("s", "c", unified_names(caps_all)), trace=lambda r: None)


class PortedModulesTest(unittest.TestCase):
    def test_trajectory_medoid_selector(self):
        from harnessstudio.modules.selectors import TrajectoryMedoid
        sel = TrajectoryMedoid(scales=[1.0] * 12)
        chunks = [Chunk([[0.2] * 12] * 16), Chunk([[0.0] * 12] * 16), Chunk([[-0.1] * 12] * 16)]
        self.assertEqual(sel.select(chunks, {}, None), 1)
        record = run({"proposer": ModuleRef("harnessstudio.methods.medoid_bon:SeededPool", {"num_samples": 4}),
                      "selector": ModuleRef("harnessstudio.modules.selectors:TrajectoryMedoid", {"scales": [1.0] * 12})})
        self.assertEqual(record["steps"], 64)

    def test_public_stasis_advisory_never_cuts(self):
        ex = {"executor": ModuleRef("harnessstudio.modules.executors:PublicStasis", {"min_k": 4, "per_step": True})}
        record = run(ex)
        self.assertEqual(record["policy_calls"], 4)
        self.assertTrue(all(c is None for c in record["cuts"]))

    def test_public_stasis_cut_mode_cuts_during_stall(self):
        ex = {"executor": ModuleRef("harnessstudio.modules.executors:PublicStasis",
                                    {"min_k": 4, "per_step": True, "cut": True})}
        record = run(ex)
        self.assertGreater(record["policy_calls"], 4)
        self.assertTrue(any(c is not None for c in record["cuts"]))

    def test_instruction_rewrite_episode_op(self):
        from harnessstudio.modules.episode import InstructionRewrite
        op = InstructionRewrite("tests.test_ported_modules:fake_client", model="m", expected_returned_model="m",
                                catalog_sha256="cat", policy_id="s", checkpoint_identity="c",
                                memory_cards=[{"id": "k1", "policy_id": "s", "checkpoint_identity": "c",
                                               "instruction": "pick up the cube", "advice": "approach slowly",
                                               "evidence_sha256": "a" * 64}])
        self.assertEqual(op.start_episode(fx.INSTRUCTION, None), "carefully pick up the synthetic cube")
        self.assertEqual(op.tokens, 7)
        with self.assertRaises(RuntimeError):
            op.start_episode(fx.INSTRUCTION, None)
        record = run({"episode_op": ModuleRef("harnessstudio.modules.episode:InstructionRewrite", {
            "client_factory": "tests.test_ported_modules:fake_client", "model": "m", "expected_returned_model": "m",
            "catalog_sha256": "cat", "policy_id": "s", "checkpoint_identity": "c"})})
        self.assertEqual(record["instruction_used"], "carefully pick up the synthetic cube")

    def test_operator_refuses_astra(self):
        from harnessstudio.runtime.operators import NRPInstructionOperator
        with self.assertRaises(ValueError):
            NRPInstructionOperator(FakeNRP(), model="gpt-6-astra", expected_returned_model="x", catalog_sha256="c")
