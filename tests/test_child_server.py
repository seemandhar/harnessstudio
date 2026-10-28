import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


@unittest.skipIf(torch is None, "GR00T env only")
class ServerTest(unittest.TestCase):
    def test_endpoints(self):
        from harnessstudio.host import server
        from harnessstudio.verify import fixtures as fx

        class FakePolicy:
            def get_action(self, obs):
                return fx._unflatten(torch.randn(16, 12).tolist())
        manifest_sha = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp) / "calls.jsonl").open("x") as log:
                ep = server.Endpoints(FakePolicy(), method_slots={}, manifest_sha256=manifest_sha, log=log,
                                      hashes=lambda v: {"__aggregate__": "x"})
                a = ep.get_action_seeded({"observation": {}, "noise_seed": 5})
                b = ep.get_action_seeded({"observation": {}, "noise_seed": 5})
                self.assertEqual(fx.flatten(a), fx.flatten(b))
                r = ep.propose({"observation": {}, "arm": "null", "decision_index": 0, "step_index": 0,
                                "seeds": [5, 6], "want": [], "method_manifest_sha256": manifest_sha})
                self.assertEqual(len(r["natives"]), 2)
                self.assertEqual(fx.flatten(r["natives"][0]), fx.flatten(a))
                self.assertNotEqual(fx.flatten(r["natives"][1]), fx.flatten(a))
                with self.assertRaises(ValueError):
                    ep.propose({"observation": {}, "arm": "null", "decision_index": 0, "step_index": 0,
                                "seeds": [5], "want": [], "method_manifest_sha256": "b" * 64})
            self.assertEqual(len((Path(tmp) / "calls.jsonl").read_text().splitlines()), 3)
