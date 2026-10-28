import unittest
from unittest import mock

from harnessstudio.core import registry
from harnessstudio.core.errors import RegistrationError


class RegistryTest(unittest.TestCase):
    def test_builtins_listed(self):
        methods = registry.all_methods(include_plugins=False)
        self.assertIn("identity", methods)
        config, origin = methods["identity"]
        self.assertEqual(origin["kind"], "builtin")
        self.assertEqual(config.slots, {})

    def test_get_unknown(self):
        with self.assertRaises(KeyError):
            registry.get("no-such-method")

    def test_duplicate_plugin_name_refused(self):
        identity_config = registry.get("identity")[0]  # resolve BEFORE patching (avoids recursion)
        fake = mock.Mock()
        fake.name = "identity"
        fake.load.return_value = lambda: identity_config
        fake.dist.metadata = {"Name": "dup"}
        fake.dist.version = "0"
        with mock.patch.object(registry, "plugin_entry_points", return_value=[fake]):
            with self.assertRaises(RegistrationError):
                registry.all_methods()
