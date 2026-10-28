import dataclasses
import unittest

from harnessstudio.core.arms import (Arm, executed_index, executed_k, honor_cut,
                                     instruction_used, num_samples, sampler_effective)
from harnessstudio.core.slots import DecisionCtx


class ArmTest(unittest.TestCase):
    def test_neutralization_table(self):
        self.assertEqual(executed_index(Arm.CANDIDATE, 3), 3)
        self.assertEqual(executed_index(Arm.NULL, 3), 0)
        self.assertEqual(executed_k(Arm.CANDIDATE, 4, 16), 4)
        self.assertEqual(executed_k(Arm.NULL, 4, 16), 16)
        self.assertEqual(executed_k(Arm.BASELINE, 4, 16), 16)
        self.assertFalse(honor_cut(Arm.CANDIDATE, False))
        self.assertTrue(honor_cut(Arm.NULL, False))
        self.assertEqual(instruction_used(Arm.CANDIDATE, "a", "b"), "b")
        self.assertEqual(instruction_used(Arm.NULL, "a", "b"), "a")
        self.assertTrue(sampler_effective(Arm.CANDIDATE))
        self.assertFalse(sampler_effective(Arm.NULL))
        self.assertEqual(num_samples(Arm.BASELINE, 8), 1)
        self.assertEqual(num_samples(Arm.NULL, 8), 8)

    def test_decision_ctx_has_no_identity_fields(self):
        names = {f.name for f in dataclasses.fields(DecisionCtx)}
        self.assertFalse(names & {"case_id", "task", "trial", "seed", "arm", "episode_seed"})
