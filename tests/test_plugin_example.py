import unittest
from importlib import metadata

import tests  # noqa: F401


def installed():
    try:
        metadata.distribution("hs-example-plugin")
        return True
    except metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(installed(), "pip install -e harnessstudio/examples/plugin_method first")
class PluginExampleTest(unittest.TestCase):
    def test_listed_and_closure_includes_editable_source(self):
        from harnessstudio.core import registry
        from harnessstudio.core.manifest import method_manifest
        config, origin = registry.get("first-sample")
        self.assertEqual(origin["kind"], "plugin")
        self.assertEqual(origin["distribution"], "hs-example-plugin")
        m = method_manifest(config, origin, policy="gr00t-n1.5")
        self.assertTrue(any(p.endswith("hs_example_plugin/__init__.py") for p in m["closure"]))
        self.assertEqual(m["plugin_top_packages"], ["hs_example_plugin"])

    def test_freezes_under_plugins_dir(self):
        import tempfile
        from pathlib import Path
        from harnessstudio.core import registry
        from harnessstudio.core.manifest import freeze, method_manifest
        config, origin = registry.get("first-sample")
        with tempfile.TemporaryDirectory() as tmp:
            frozen = freeze(method_manifest(config, origin, policy="gr00t-n1.5"), Path(tmp))
            self.assertIn("plugins/hs_example_plugin/__init__.py", frozen["files"])
            self.assertIn("harnessstudio/core/method.py", frozen["files"])
