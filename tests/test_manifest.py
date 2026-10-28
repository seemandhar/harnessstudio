import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harnessstudio.core import manifest
from harnessstudio.core.registry import get


class ManifestTest(unittest.TestCase):
    def test_builtin_manifest_is_deterministic(self):
        config, origin = get("medoid-bon")
        a = manifest.method_manifest(config, origin, policy="gr00t-n1.5")
        b = manifest.method_manifest(config, origin, policy="gr00t-n1.5")
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertTrue(any(p.endswith("medoid_bon.py") for p in a["closure"]))

    def test_editable_closure_walks_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "src" / "plug"
            (pkg / "sub").mkdir(parents=True)
            (pkg / "__init__.py").write_text("X = 1\n")
            (pkg / "sub" / "mod.py").write_text("Y = 2\n")
            dist = mock.Mock()
            dist.read_text.side_effect = lambda n: json.dumps(
                {"url": "file://" + str(Path(tmp) / "src"), "dir_info": {"editable": True}}) if n == "direct_url.json" else None
            files = manifest.distribution_closure(dist, top_packages=["plug"])
            self.assertEqual(sorted(p.name for p in files), ["__init__.py", "mod.py"])
            m = {"closure": {str(p): manifest.sha256_file(p) for p in files}, "sha256": "x",
                 "plugin_top_packages": ["plug"]}
            with tempfile.TemporaryDirectory() as out:
                frozen = manifest.freeze(m, Path(out))
                self.assertIn("plugins/plug/sub/mod.py", frozen["files"])

    def test_compiled_extension_refused(self):
        dist = mock.Mock()
        dist.read_text.return_value = None
        dist.files = ["plug/fast.so", "plug/__init__.py"]
        with self.assertRaises(manifest.RegistrationError):
            manifest.distribution_closure(dist, top_packages=["plug"])

    def test_freeze_layout_in_tree_plugin_and_package(self):
        pkg_file = manifest.PACKAGE_ROOT / "core" / "method.py"
        plugin = manifest.PACKAGE_ROOT.parent / "examples" / "plugin_method" / "hs_example_plugin" / "__init__.py"
        self.assertEqual(manifest._plugin_relpath(plugin, ["hs_example_plugin"]),
                         Path("plugins/hs_example_plugin/__init__.py"))
        m = {"closure": {str(pkg_file): manifest.sha256_file(pkg_file)}, "sha256": "x", "plugin_top_packages": []}
        with tempfile.TemporaryDirectory() as out:
            frozen = manifest.freeze(m, Path(out))
            self.assertIn("harnessstudio/core/method.py", frozen["files"])

    def test_freeze_copies_with_hashes(self):
        config, origin = get("identity")
        m = manifest.method_manifest(config, origin, policy="gr00t-n1.5")
        with tempfile.TemporaryDirectory() as tmp:
            frozen = manifest.freeze(m, Path(tmp))
            for rel, digest in frozen["files"].items():
                self.assertEqual(manifest.sha256_file(Path(tmp) / rel), digest)
