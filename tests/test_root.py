import re
import unittest

from tests import PACKAGE


class IndependenceTest(unittest.TestCase):
    def test_runtime_is_self_contained(self):
        import harnessstudio.runtime as rt
        self.assertTrue(hasattr(rt, "UnifiedHarness"))
        self.assertEqual(rt.VERSION, "1.1.0")

    def test_package_never_imports_robometaharness(self):
        """HarnessStudio is standalone; RoboMetaHarness depends on it, never the reverse."""
        allowed = {"methods/chunktrust_ahs/horizon.py"}  # byte-identical vendored file; unused lazy function
        pattern = re.compile(r"^\s*(from|import)\s+robometaharness", re.M)
        offenders = [str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*.py")
                     if pattern.search(p.read_text()) and str(p.relative_to(PACKAGE)) not in allowed]
        self.assertEqual(offenders, [])

    def test_config_env_overrides(self):
        import os
        from unittest import mock
        from harnessstudio import config
        with mock.patch.dict(os.environ, {"HARNESSSTUDIO_WORKSPACE": "/tmp/hs-ws", "HARNESSSTUDIO_LEASES": "/tmp/hs-l",
                                          "HARNESSSTUDIO_POLICYBENCH": "/tmp/hs-pb"}):
            self.assertEqual(str(config.runs_dir()), "/tmp/hs-ws/runs/harnessstudio")
            self.assertEqual(str(config.leases_dir()), "/tmp/hs-l")
            self.assertEqual(str(config.policybench()), "/tmp/hs-pb")
        self.assertTrue(config.DEFAULT_TABLE.is_file())
