import unittest

from harnessstudio.core.capabilities import Capability, caps, require, unified_names
from harnessstudio.core.errors import CapabilityError


class CapabilityTest(unittest.TestCase):
    def test_parse_and_unified_mapping(self):
        c = caps("action_chunk", "isolated_streams")
        self.assertEqual(c, frozenset({Capability.ACTION_CHUNK, Capability.ISOLATED_STREAMS}))
        names = unified_names(c)
        self.assertIn("independent_sampling", names)
        self.assertIn("isolated_sampling", names)
        self.assertIn("action_chunk", names)

    def test_unknown_capability_rejected(self):
        with self.assertRaises(ValueError):
            caps("telepathy")

    def test_require_reports_missing(self):
        with self.assertRaises(CapabilityError) as ctx:
            require(caps("denoise_step"), caps("action_chunk"), method="m", policy="p")
        self.assertIn("denoise_step", str(ctx.exception))
        require(caps("action_chunk"), caps("action_chunk", "proprio"), method="m", policy="p")


class AdapterDeclTest(unittest.TestCase):
    def test_gr00t_declarations(self):
        from harnessstudio.policies import adapter, names
        self.assertIn("gr00t-n1.5", names())
        a = adapter("GR00T")
        self.assertEqual(a.CHUNK_LEN, 16)
        self.assertEqual(a.ACTION_DIM, 12)
        self.assertEqual(len(a.PUBLIC_KEYS), 9)
        self.assertEqual(a.ACTION_LAYOUT["action.end_effector_position"], (5, 8))
        self.assertIn(Capability.DENOISE_STEP, a.CAPABILITIES)

    def test_matches_groot_worker(self):
        from harnessstudio.bench import groot_native as gw
        from harnessstudio.policies import gr00t_n15 as g
        self.assertEqual(g.PUBLIC_KEYS, tuple(sorted(gw.PUBLIC_KEYS)))
        self.assertEqual(g.ACTION_SHAPES, gw.ACTION_SHAPES)
