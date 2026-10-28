import unittest
from unittest import mock

from harnessstudio.verify.tier1 import run_tier1


class Tier1Test(unittest.TestCase):
    def test_all_builtins_pass_tier1_on_gr00t(self):
        for name in ("identity", "medoid-bon", "chunk-interrupt", "chunktrust-ahs", "llm-subtask-planner"):
            report = run_tier1(name, "gr00t-n1.5")
            failed = [(c["name"], c["detail"]) for c in report["checks"] if not c["passed"]]
            self.assertEqual(failed, [], f"{name}: {failed}")
            self.assertTrue(report["passed"])

    def test_incompatible_method_fails_capability_check(self):
        from harnessstudio.core.capabilities import caps
        with mock.patch("harnessstudio.policies.gr00t_n15.CAPABILITIES", caps("action_chunk")):
            report = run_tier1("chunk-interrupt", "gr00t-n1.5")
        self.assertFalse(report["passed"])
        self.assertFalse(next(c for c in report["checks"] if c["name"] == "capabilities")["passed"])
