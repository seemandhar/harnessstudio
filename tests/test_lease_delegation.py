"""Delegated GPU leases: an outer scheduler leases GPUs, then launches a HarnessStudio run on them.

Without delegation the runner's own acquire() collides with the outer lease ("already reserved").
With HARNESSSTUDIO_DELEGATED_LEASE set, the run checks that the lease covers it and is held by a live
ancestor, and never acquires or releases a lease itself.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests  # noqa: F401
from tests import PACKAGE
from harnessstudio import config
from harnessstudio.bench import resources

OWNER = "outer.run.idea.development"


def _runner():
    backend = PACKAGE / "backends/gr00t_robocasa365"
    for p in (str(backend), str(backend / "copied")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import runner
    return runner


def _inventory(ids):
    return [{"index": i, "uuid": f"u{i}", "total_mib": 80000, "used_mib": 1, "utilization_percent": 0,
             "processes": [], "processes_known": True} for i in ids]


CHILD = """
import sys
sys.path[:0] = {paths!r}
import runner
from harnessstudio.bench import resources
try:
    runner.require_delegated_lease({leases!r}, {owner!r}, {ids!r}, {workers!r})
    print("ACCEPTED")
except resources.ResourceError as exc:
    print("REFUSED", exc)
"""


class DelegatedLeaseTest(unittest.TestCase):
    def setUp(self):
        self.runner = _runner()
        self.tmp = tempfile.TemporaryDirectory()
        self.leases = Path(self.tmp.name)
        # This test process plays the outer scheduler: it holds the lease.
        resources.acquire(self.leases, OWNER, [0, 1], inventory=_inventory([0, 1, 2]),
                          instances_per_gpu=2, calibrated_instance_mib=20000)

    def tearDown(self):
        self.tmp.cleanup()

    def _child(self, ids, workers=1, owner=OWNER):
        backend = PACKAGE / "backends/gr00t_robocasa365"
        code = CHILD.format(paths=[str(PACKAGE.parent), str(backend), str(backend / "copied")],
                            leases=str(self.leases), owner=owner, ids=ids, workers=workers)
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout

    def test_a_second_owner_cannot_lease_the_same_gpus(self):
        with self.assertRaisesRegex(resources.ResourceError, "already reserved"):
            resources.acquire(self.leases, "benchmark.harnessstudio.x", [0, 1], inventory=_inventory([0, 1]))

    def test_a_descendant_of_the_holder_may_use_the_lease(self):
        self.assertIn("ACCEPTED", self._child([0, 1], 2))
        self.assertIn("ACCEPTED", self._child([1], 1))

    def test_gpus_outside_and_too_many_workers_are_refused(self):
        self.assertIn("outside delegated lease", self._child([0, 2]))
        self.assertIn("exceed delegated lease", self._child([0], 3))

    def test_missing_foreign_and_dead_holder_leases_are_refused(self):
        with self.assertRaisesRegex(resources.ResourceError, "not in the registry"):
            self.runner.require_delegated_lease(self.leases, "nobody", [0], 1)
        # this process holds the lease, so for itself the holder is not an ancestor: foreign
        with self.assertRaisesRegex(resources.ResourceError, "not a live ancestor"):
            self.runner.require_delegated_lease(self.leases, OWNER, [0], 1)
        registry = json.loads((self.leases / "resources.json").read_text())
        registry["leases"][OWNER]["pid"] = 2 ** 22 + 7               # a holder that no longer exists
        (self.leases / "resources.json").write_text(json.dumps(registry))
        self.assertIn("not a live ancestor", self._child([0]))

    def test_registry_is_unchanged_by_the_check(self):
        before = (self.leases / "resources.json").read_text()
        self._child([0, 1])
        self.assertEqual((self.leases / "resources.json").read_text(), before)

    def test_config_reads_the_delegated_lease_from_the_environment(self):
        with mock.patch.dict(os.environ, {"HARNESSSTUDIO_DELEGATED_LEASE": "outer.x"}):
            self.assertEqual(config.delegated_lease(), "outer.x")
        with mock.patch.dict(os.environ, {"HARNESSSTUDIO_DELEGATED_LEASE": ""}):
            self.assertIsNone(config.delegated_lease())

    def test_claim_and_release_never_touch_a_delegated_lease(self):
        manifest = {"lease_owner": OWNER, "lease_delegated": True, "workers_per_gpu": 1}
        with mock.patch.object(self.runner, "require_delegated_lease") as check, \
                mock.patch.object(resources, "acquire") as acquire, mock.patch.object(resources, "release") as release:
            self.assertTrue(self.runner.claim_lease(manifest, self.leases, [0], None))
            self.assertFalse(self.runner.release_lease(manifest, self.leases, True, []))
        check.assert_called_once()
        acquire.assert_not_called()
        release.assert_not_called()

    def test_own_lease_is_taken_and_released_only_after_clean_reaping(self):
        manifest = {"lease_owner": "benchmark.harnessstudio.own", "workers_per_gpu": 1}
        with mock.patch.object(resources, "acquire") as acquire, mock.patch.object(resources, "release") as release:
            self.assertFalse(self.runner.claim_lease(manifest, self.leases, [2], _inventory([2])))
            self.assertFalse(self.runner.release_lease(manifest, self.leases, False, ["worker alive"]))
            release.assert_not_called()
            self.assertTrue(self.runner.release_lease(manifest, self.leases, False, []))
        acquire.assert_called_once()
        release.assert_called_once()

    def test_both_backends_record_delegation_in_the_manifest(self):
        for rel in ("backends/gr00t_robocasa365/adapter.py", "backends/pi05_robocasa365/adapter.py"):
            text = (PACKAGE / rel).read_text()
            self.assertIn("lease_owner=delegated_lease() or", text, rel)
            self.assertIn("lease_delegated=bool(delegated_lease())", text, rel)


if __name__ == "__main__":
    unittest.main()
