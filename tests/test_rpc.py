import unittest

from harnessstudio.host import rpc


class RpcTest(unittest.TestCase):
    def good(self, **kw):
        base = dict(observation={"a": 1}, arm="candidate", decision_index=0, step_index=0,
                    seeds=[1, 2], want=["sampler"], method_manifest_sha256="a" * 64)
        base.update(kw)
        return base

    def test_valid_request(self):
        req = rpc.parse_request(self.good(), manifest_sha256="a" * 64)
        self.assertEqual(req.seeds, [1, 2])

    def test_rejects(self):
        for bad in (dict(arm="baseline"), dict(seeds=[]), dict(seeds=[-1]), dict(decision_index=-1),
                    dict(want=["telepathy"]), dict(method_manifest_sha256="b" * 64), dict(extra=1)):
            with self.assertRaises(ValueError, msg=str(bad)):
                rpc.parse_request(self.good(**bad), manifest_sha256="a" * 64)

    def test_seeded_payload(self):
        self.assertEqual(rpc.parse_seeded({"observation": {}, "noise_seed": 3}), ({}, 3))
        with self.assertRaises(ValueError):
            rpc.parse_seeded({"observation": {}, "noise_seed": -3})

    def test_extras_size_bound(self):
        rpc.check_extras_size({"x": list(range(10))})
        with self.assertRaises(ValueError):
            rpc.check_extras_size({"x": "y" * (rpc.MAX_EXTRAS_BYTES + 1)})
        import numpy as np
        self.assertEqual(rpc.check_extras_size({"v": np.zeros((4, 16, 32), np.float32)}) >= 8192, True)
