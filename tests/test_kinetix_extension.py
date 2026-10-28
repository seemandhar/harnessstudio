"""Kinetix RTC extension: control-plane tests that run in the HarnessStudio interpreter (no JAX)."""
from __future__ import annotations

import contextlib
import hashlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

# Episode-execution path of GR00T N1.5 / Pi0.5 at the extension base (9c4a977 + live overlay, commit 7f321be).
EXECUTION_PATH_SHA256 = {
    "harnessstudio/core/pipeline.py": "25de033d703438c2e2501dc46d07e91d320b94aaafaba07f65ac4a4aebae48f7",
    "harnessstudio/core/runtime.py": "07c17b928aa4278937ea77d51b2a72ffcd658607474deddd2894927be03f798f",
    "harnessstudio/core/arms.py": "1877c093b5a7a7d6dd412a3421876347d5f785625c739c69c5fc037c21f72dc4",
    "harnessstudio/core/noise.py": "b8a12872cf1f7360772ad5dce1d394ee3dbb9306ff70670f621094d293d29f14",
    "harnessstudio/core/budget.py": "f5754ace2c8f224afafe72593117a55c4b31b51699bd65a2db0f15b77826c765",
    "harnessstudio/policies/gr00t_n15.py": "cb5f5030c0eb80a7687b994dc7e69ced0fdbe27ece0dfd0c8db9d1cd3f9b93dd",
    "harnessstudio/policies/pi05.py": "6cfc907b18cc57ed88f55e6b28758cfc884d8d8315bef056816d1a072235f14e",
    "harnessstudio/backends/gr00t_robocasa365/COPIED.json": "0b124ea232b01b31e12f3833b1a4b9828982a58ef16cb4fb21f8f871dcbdced7",
    "harnessstudio/backends/gr00t_robocasa365/__init__.py": "efe26437ea05ff16874dcf99c740e0d1e722053e021f7aad83233dc22a2eb683",
    "harnessstudio/backends/gr00t_robocasa365/adapter.py": "8b014f7becc65c67fc5f65c950693b82c11ec440f4f4a978462562982caf73d2",
    "harnessstudio/backends/gr00t_robocasa365/copied/crn.py": "69742a552864c889df4316d3b1019ba5051874388b1b294f387381ce060c59d6",
    "harnessstudio/backends/gr00t_robocasa365/copied/evaluator_log.py": "df5e8ed946aff71a07651fa48cfed8bb21adabb6b25f8b14030f2243fea243a8",
    "harnessstudio/backends/gr00t_robocasa365/copied/np_attribution_v2.py": "433cb737c87f8ad09e22d20b655c93501b473f1b1bac1b8dd17a5ce5b5e5ef3c",
    "harnessstudio/backends/gr00t_robocasa365/copied/parity.py": "db7b26fa850a201fdfb7211807f135dc605ef07628b8d57205753796c1c2052a",
    "harnessstudio/backends/gr00t_robocasa365/copied/protocol_v2_n1.json": "8e9346aff71377b5a2eac91a9fa89e17bab232410add3de61ae23b79af62a36a",
    "harnessstudio/backends/gr00t_robocasa365/copied/render_context_fix.py": "9d93bbeb690fe0b0bc0a255c41665963dcddec5d203385574a0a561cc78242eb",
    "harnessstudio/backends/gr00t_robocasa365/copied/ri_scan.py": "b550037ca451a9500425774506b1d75e1339b6ad4a0a00372ad8fd6b76ed9c10",
    "harnessstudio/backends/gr00t_robocasa365/copied/scene_determinism.py": "be83b72351ce4e540a16bcc02cecf722a594233d68c5ac8c487612dab132ebe1",
    "harnessstudio/backends/gr00t_robocasa365/copied/velocity_trace.py": "6148036a34000dbc86eaa161290ef6404e44d4985357de687bbe653485f6461c",
    "harnessstudio/backends/gr00t_robocasa365/reset_probe.py": "fb441fc2073502011aa610e4ec2ae2598a6067149fd7d8c48e5fb4398b7d6dde",
    "harnessstudio/backends/gr00t_robocasa365/runner.py": "2d0b6fcd486452c8263ea318355bc33131c104447e80f003001c89399050bc17",
    "harnessstudio/backends/gr00t_robocasa365/worker.py": "067ebe1fd78eef63b49a182bbbd1bf7187db635df072424ff2ad401b4a215a24",
    "harnessstudio/backends/pi05_robocasa365/__init__.py": "310602de6897dabedf4b6ff6ffcec7016e8bd843a512bd2e02690a1cc166724f",
    "harnessstudio/backends/pi05_robocasa365/adapter.py": "b98fdf8420e8cae86d962da9d6135740013bb6c694c4d24b400eadbd3919cbaf",
    "harnessstudio/backends/pi05_robocasa365/protocol_pi05_n1.json": "9b7dd8e6b2270dd465c5946722ab2402f34e2c1e49824a9be2284e08e1625c26",
    "harnessstudio/backends/pi05_robocasa365/worker.py": "fec04267f1c4ca1addb6c22c52170b13b3cff09e0aeb6aeca0fc383b42cac77b",
    "harnessstudio/runtime/PROVENANCE.json": "7589848c8c779b6572f105ee4ff224b5b04f6ebf1057bad1b85db442745c2342",
    "harnessstudio/runtime/__init__.py": "44997eb999bfd39cb5c06d1742773af5cc18d5bf897823b0e5b06971ef75f862",
    "harnessstudio/runtime/__main__.py": "0c9295680289873f4ab05f6eb4176e0a30e56024fc74a0fadca1dafd8b5c42aa",
    "harnessstudio/runtime/contracts.py": "03d3e7d6031708925469c1824b8700a94869c3ea5170a617d4cf164dbd68ef3c",
    "harnessstudio/runtime/core.py": "a896580e6cb5897302c8042e4ea16e59d7b5e5774caaec9b2596b4c2a871d876",
    "harnessstudio/runtime/memory.py": "45f1255e96b0cda4b0a24e90ff79e33a7e53f4067f59f65f0228c1e46f365443",
    "harnessstudio/runtime/modules.py": "7ee7f3c8a69a4b4683160e523553e97beffb4ab29719bdff32e46a9f6700a3c2",
    "harnessstudio/runtime/operators.py": "3fca180084859cc1bf97d37d2a6b7f21bbaeaafa8b28fcf89a0d529fa1a13b3c",
    "harnessstudio/runtime/parity.py": "679ceeb2ba5f4d2c0d6fb28755126c90749d7cf46af9e6e247e0bac21becbe5e",
}


class ExecutionPathUnchanged(unittest.TestCase):
    def test_execution_path_byte_identical(self):  # EC6(c)
        for rel, want in EXECUTION_PATH_SHA256.items():
            got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            self.assertEqual(got, want, rel)


class ImportIsolation(unittest.TestCase):
    def test_import_list_and_robocasa_tier1_never_import_jax(self):  # B5
        code = ("import sys, contextlib, io\n"
                "import harnessstudio\n"
                "from harnessstudio import cli\n"
                "from harnessstudio.policies import kinetix_flow\n"
                "with contextlib.redirect_stdout(io.StringIO()): cli.main(['list'])\n"
                "from harnessstudio.verify.tier1 import run_tier1\n"
                "assert run_tier1('identity', 'gr00t-n1.5')['passed']\n"
                "bad = [m for m in ('jax', 'flax', 'kinetix') if m in sys.modules]\n"
                "assert not bad, bad\n")
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                              env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])


class Protocol(unittest.TestCase):
    def test_require_protocol(self):
        from harnessstudio.core.errors import CapabilityError
        from harnessstudio.core.protocol import require_protocol
        from harnessstudio.policies import gr00t_n15, kinetix_flow, pi05
        self.assertEqual(require_protocol(gr00t_n15), (None, None))
        self.assertEqual(require_protocol(kinetix_flow, 1, 3), (1, 3))
        self.assertEqual(require_protocol(kinetix_flow, 4), (4, 4))
        for pa in (gr00t_n15, pi05):
            with self.assertRaises(CapabilityError):
                require_protocol(pa, 1, None)
            with self.assertRaises(CapabilityError):
                require_protocol(pa, None, 4)
        for d, s in ((2, 1), (0, 0), (0, 9), (-1, 1)):
            with self.assertRaises(CapabilityError):
                require_protocol(kinetix_flow, d, s)

    def test_cli_refuses_protocol_flags_on_robocasa_before_any_run(self):  # EC1
        from harnessstudio import cli
        with mock.patch("harnessstudio.eval.quick.quick", side_effect=AssertionError("must not run")),                 mock.patch("harnessstudio.eval.runs.evaluate", side_effect=AssertionError("must not run")):
            for argv in (["eval", "identity", "--policy", "gr00t-n1.5", "--inference-delay", "1"],
                         ["eval", "identity", "--policy", "pi0.5", "--execute-horizon", "4"],
                         ["run", "identity", "--policy", "gr00t-n1.5", "--task", "T", "--trial", "0",
                          "--inference-delay", "1"]):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    self.assertEqual(cli.main(argv), 2)
                self.assertIn("CapabilityError", err.getvalue())


class SlotsAndCapabilities(unittest.TestCase):
    def test_supported_slots_enforced(self):  # B3
        from harnessstudio.core import registry
        from harnessstudio.core.protocol import compatible
        from harnessstudio.policies import gr00t_n15, kinetix_flow, pi05
        methods = registry.all_methods()
        for name in ("identity", "rtc", "rtc-hard-masking"):
            self.assertTrue(compatible(methods[name][0], kinetix_flow), name)
        for name in ("llm-subtask-planner", "medoid-bon", "chunk-interrupt", "subtask-sequencer", "chunktrust-ahs"):
            self.assertFalse(compatible(methods[name][0], kinetix_flow), name)
        for pa in (gr00t_n15, pi05):
            self.assertFalse(compatible(methods["rtc"][0], pa))
            self.assertTrue(compatible(methods["identity"][0], pa))
            self.assertTrue(compatible(methods["llm-subtask-planner"][0], pa))

    def test_guide_sampler_requires_capability(self):
        import dataclasses
        from harnessstudio.core.capabilities import caps
        from harnessstudio.core.errors import RegistrationError
        from harnessstudio.core.method import validate
        from harnessstudio.methods import rtc
        bad = dataclasses.replace(rtc.config(), required=caps("action_chunk", "seeded_sampling", "async_execution"))
        with self.assertRaises(RegistrationError):
            validate(bad)

    def test_validate_does_not_import_samplers_of_methods_without_one(self):  # B5
        from harnessstudio.core import slots
        from harnessstudio.core.method import validate
        from harnessstudio.methods import identity
        with mock.patch.object(slots, "sampler_has_guide", side_effect=AssertionError("imported")):
            validate(identity.config())

    def test_tier1(self):
        from harnessstudio.verify.tier1 import run_tier1
        for name in ("rtc", "rtc-hard-masking", "identity"):
            rep = run_tier1(name, "kinetix-flow")
            self.assertTrue(rep["passed"], (name, [c for c in rep["checks"] if not c["passed"]]))
        rep = run_tier1("rtc", "gr00t-n1.5")
        self.assertEqual([c["name"] for c in rep["checks"] if not c["passed"]], ["capabilities"])
        rep = run_tier1("llm-subtask-planner", "kinetix-flow")
        self.assertEqual([c["name"] for c in rep["checks"] if not c["passed"]], ["capabilities"])
        names = [c["name"] for c in run_tier1("rtc", "kinetix-flow")["checks"]]
        self.assertIn("guide_contract", names)


class GuideNumerics(unittest.TestCase):
    def test_prefix_weights_docstring_example(self):
        import numpy as np
        from harnessstudio.methods.rtc import prefix_weights
        w = prefix_weights(np, 2, 6, 10, "linear")
        np.testing.assert_allclose(w, [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0], atol=1e-6)
        self.assertEqual(prefix_weights(np, 3, 5, 8, "zeros").tolist(), [1, 1, 1, 0, 0, 0, 0, 0])

    def test_guide_matches_paper_formula(self):
        import numpy as np
        from harnessstudio.methods.rtc import RTCGuidance, prefix_weights
        g = RTCGuidance("exp", 5.0)
        rng = np.random.default_rng(1)
        for d, s, t in ((0, 1, 0.0), (1, 1, 0.2), (2, 5, 0.6), (4, 4, 0.8)):
            x, v, prev = (rng.standard_normal((8, 6)).astype(np.float32) for _ in range(3))
            t = np.float32(t)
            x1 = (x + (1 - t) * v).astype(np.float32)
            with np.errstate(divide="ignore"):
                out = g.guide(np, x, t, v, x1, lambda c: 2 * c, prev, d, s, 8)
                w = 5.0 if t == 0 else min(5.0, float((1 - t) / t * (t * t + (1 - t) ** 2) / (1 - t) ** 2))
            want = v + w * 2 * (prev - x1) * prefix_weights(np, d, 8 - s, 8, "exp")[:, None]
            np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-6)

    def test_guide_contract_rejects_side_effects(self):
        import tempfile
        from harnessstudio.core.method import ModuleRef
        from harnessstudio.verify.guide_contract import check_guide_contract
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad_guide_mod.py").write_text(
                "class Bad:\n    identity = {}\n    def guide(self, xp, x_t, t, v_t, x1, vjp, prev, d, s, H):\n"
                "        print('side effect')\n        return v_t\n    def extras(self):\n        return {}\n")
            sys.path.insert(0, tmp)
            try:
                with self.assertRaises(AssertionError):
                    check_guide_contract(ModuleRef("bad_guide_mod:Bad"))
            finally:
                sys.path.remove(tmp)


if __name__ == "__main__":
    unittest.main()
