"""Pi0.5 server determinism contract.

XLA's GPU autotuner picks kernels by timing them at compile time, so two Pi0.5 server processes
given the same request and seed returned actions differing by up to ~2e-3 (probe, 2026-10-28).
With autotuning off they were bitwise identical. Paired arms run in separate server processes,
so the backend must disable autotuning and the server must refuse to start without it.
"""
import json
import unittest
from unittest import mock

from harnessstudio.backends.pi05_robocasa365 import adapter, worker

FLAG = "--xla_gpu_autotune_level=0"


class Pi05DeterminismTest(unittest.TestCase):
    def test_server_env_disables_xla_autotuning(self):
        self.assertIn(FLAG, adapter.SERVER_ENV["XLA_FLAGS"].split())

    def test_backend_version_marks_the_determinism_change(self):
        self.assertEqual(worker.RUNNER_VERSION, "harnessstudio-pi05-2")
        self.assertEqual(adapter.RUNNER_VERSION, worker.RUNNER_VERSION)

    def test_protocol_records_the_rule(self):
        protocol = json.loads(adapter.PROTOCOL.read_text())
        self.assertIn(FLAG, protocol["determinism"]["xla_flags"])

    def test_server_refuses_to_start_without_the_flag(self):
        with mock.patch.dict("os.environ", {"XLA_FLAGS": ""}):
            with self.assertRaisesRegex(RuntimeError, "autotune"):
                worker.require_deterministic_xla()
        with mock.patch.dict("os.environ", {"XLA_FLAGS": FLAG}):
            worker.require_deterministic_xla()


if __name__ == "__main__":
    unittest.main()
