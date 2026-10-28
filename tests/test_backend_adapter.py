"""Adapter dry-run tests (CPU; read the real task registry and checkpoint inventory, no GPU)."""
import json
import tempfile
import unittest
from pathlib import Path

from tests import PACKAGE

TABLE = PACKAGE / "data/canonical_tables/robocasa365_atomic_seen_dev_r2.json"


def dev_tasks():
    return sorted({k.split("--")[0] for k in json.loads(TABLE.read_text())["cases"]})


def adapter():
    import importlib.util
    path = PACKAGE / "backends/gr00t_robocasa365/adapter.py"
    spec = importlib.util.spec_from_file_location("hs_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policybench_available():
    from harnessstudio import config
    return (config.policybench() / "envs/gr00t-n15/bin/python").is_file()


@unittest.skipUnless(_policybench_available(), "needs RobotPolicyBench (set HARNESSSTUDIO_POLICYBENCH)")
class AdapterDryRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ad = adapter()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def dry(self, method, arm="baseline", baseline_run=None, stepping=None, tasks=None):
        cfg = Path(self.tmp.name) / f"{method}-{arm}-{stepping}.json"
        cfg.write_text(json.dumps({"schema": "harnessstudio-run/1", "method": method, "arm": arm,
                                   "baseline_run": baseline_run, "stepping_override": stepping}))
        argv = ["--policy", "gr00t-n1.5", "--benchmark", "robocasa365", "--harness", "harnessstudio",
                "--harness-config", str(cfg), "--dry-run", "--canonical-table", str(TABLE),
                "--groups", "atomic_seen", "--tasks", *(tasks or dev_tasks()), "--trials", "2", "3",
                "--gpus", "0", "--output", str(Path(self.tmp.name) / "out")]
        return self.ad.prepare(self.ad.parser().parse_args(argv), "harnessstudio")

    def test_baseline_cases_and_seed_rule(self):
        plan = self.dry("identity")
        self.assertEqual(len(plan["cases"]), 32)
        self.assertTrue(all(c["seed"] == 7 + 50 * c["task_index"] + c["trial"] for c in plan["cases"]))
        self.assertEqual(plan["harness_identity"], "gr00t-n15-n1-baseline")
        self.assertTrue(plan["videos"])

    def test_protocol_hash_is_arm_independent_but_stepping_dependent(self):
        base = self.dry("medoid-bon")
        cand = self.dry("medoid-bon", arm="candidate")
        self.assertEqual(base["protocol_sha256"], cand["protocol_sha256"])
        self.assertEqual(cand["harness_identity"], "medoid-bon@2/candidate")
        per_step = self.dry("chunk-interrupt")
        self.assertEqual(per_step["stepping"], "per_step")
        self.assertNotEqual(per_step["protocol_sha256"], base["protocol_sha256"])

    def test_config_rules(self):
        with self.assertRaises(ValueError):
            self.dry("identity", arm="null")  # null without baseline_run
        with self.assertRaises(ValueError):
            self.dry("medoid-bon", stepping="per_step")  # override only for identity
        override = self.dry("identity", stepping="per_step")
        self.assertEqual(override["stepping"], "per_step")
        self.assertEqual(override["harness_identity"], "gr00t-n15-n1-baseline-perstep")

    def test_call_bound_matches_formula(self):
        cand = self.dry("medoid-bon", arm="candidate")
        self.assertEqual(cand["native_policy_call_upper_bound"],
                         sum(8 * -(-c["horizon"] // 16) for c in cand["cases"]))
