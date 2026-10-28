import json
import tempfile
import unittest
from pathlib import Path

import tests  # noqa: F401
from harnessstudio.eval.ledger import Ledger


class LedgerTest(unittest.TestCase):
    def test_require_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run1"
            run.mkdir()
            (run / "state.json").write_text(json.dumps({"gpu_hours_upper_bound": 5.5}))
            ledger = Ledger(Path(tmp) / "ledger.json")
            ledger.add(run)
            self.assertAlmostEqual(Ledger(Path(tmp) / "ledger.json").used(), 5.5)
            ledger.require(0.4)
            with self.assertRaises(RuntimeError):
                ledger.require(0.6)


class RunnerHelpersTest(unittest.TestCase):
    def setUp(self):
        import sys
        from tests import PACKAGE
        backend = PACKAGE / "backends/gr00t_robocasa365"
        for p in (str(backend), str(backend / "copied")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import runner
        self.runner = runner

    def test_plan_slots_interleaves(self):
        self.assertEqual(self.runner.plan_slots([0, 1], 2, 10), ["0.0", "1.0", "0.1", "1.1"])
        self.assertEqual(self.runner.plan_slots([0, 1, 2], 1, 2), ["0.0", "1.0"])
        with self.assertRaises(ValueError):
            self.runner.plan_slots([0], 0, 1)

    def test_placement_map(self):
        m = self.runner.placement_map({"0.0": [{"id": "a"}, {"id": "c"}], "1.0": [{"id": "b"}]})
        self.assertEqual(m, {"a": "0.0", "c": "0.0", "b": "1.0"})

    def test_parity_retry_decision(self):
        abort = {"parity_abort": True, "physical_steps_attempted": 0, "policy_calls": 0}
        self.assertEqual(self.runner.parity_retry_decision({}, 0, 1), "none")
        self.assertEqual(self.runner.parity_retry_decision(abort, 0, 1), "retry")
        self.assertEqual(self.runner.parity_retry_decision(abort, 1, 1), "invalid")
        with self.assertRaises(RuntimeError):
            self.runner.parity_retry_decision(dict(abort, policy_calls=1), 0, 1)


class LedgerCapTest(unittest.TestCase):
    def test_cap_is_read_from_the_file_and_changes_are_recorded(self):
        import tempfile
        from pathlib import Path
        from harnessstudio.eval.ledger import Ledger
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            led = Ledger(path)
            self.assertEqual(led.cap, 6.0)
            led.set_cap(12.0, "user raised the cap")
            again = Ledger(path)
            self.assertEqual(again.cap, 12.0)
            self.assertEqual(again.data["cap_history"][0]["from"], 6.0)
            again.require(11.0)
            with self.assertRaises(RuntimeError):
                again.require(12.5)
