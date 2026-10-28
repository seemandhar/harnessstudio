import unittest

from harnessstudio.core.budget import max_policy_calls
from harnessstudio.core.noise import NoiseSchedule, derive_seed

GOLDEN = 2066784501783831588  # chunktrust v2.2 crn.derive_seed(...,'CloseFridge--0002','primary',3)


class BudgetTest(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(max_policy_calls(num_samples=1, min_k=16, horizon=500), 32)
        self.assertEqual(max_policy_calls(num_samples=8, min_k=16, horizon=500), 256)
        self.assertEqual(max_policy_calls(num_samples=1, min_k=4, horizon=500), 125)

    def test_rejects_bad_values(self):
        for bad in (dict(num_samples=0, min_k=4, horizon=10), dict(num_samples=1, min_k=0, horizon=10),
                    dict(num_samples=1, min_k=4, horizon=0), dict(num_samples=1.0, min_k=4, horizon=10)):
            with self.assertRaises(ValueError):
                max_policy_calls(**bad)


class NoiseTest(unittest.TestCase):
    def test_matches_chunktrust_crn(self):
        self.assertEqual(derive_seed("chunktrust-beat-20261026/crn-v1", "CloseFridge--0002", "primary", 3), GOLDEN)

    def test_schedule_primary_shared_across_k(self):
        n = NoiseSchedule("ns", "case")
        one, eight = n.seeds(5, 1), n.seeds(5, 8)
        self.assertEqual(one[0], eight[0])
        self.assertEqual(len(set(eight)), 8)
        self.assertEqual(eight[0], derive_seed("ns", "case", "primary", 5))
        self.assertEqual(eight[3], derive_seed("ns", "case", "alt-3", 5))

    def test_stream_names(self):
        self.assertEqual(NoiseSchedule.stream_names(3), ["primary", "alt-1", "alt-2"])

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            NoiseSchedule("", "case")
        with self.assertRaises(ValueError):
            NoiseSchedule("ns", "case").seeds(-1, 1)
