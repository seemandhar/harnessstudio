"""Kinetix RTC extension: tests that need JAX and the pinned upstream (authors' env).

Self-skip unless HARNESSSTUDIO_KINETIX_CHILD_TESTS=1 and jax/kinetix import. Run with the kinetix child
interpreter, JAX_PLATFORMS=cpu and PYTHONPATH=<worktree>:
    HARNESSSTUDIO_KINETIX_CHILD_TESTS=1 JAX_PLATFORMS=cpu PYTHONPATH=. $KINETIX_ENV/bin/python -m unittest tests.test_kinetix_child -v
"""
from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
import unittest
from pathlib import Path

ENABLED = os.environ.get("HARNESSSTUDIO_KINETIX_CHILD_TESTS") == "1"
try:
    import jax  # noqa: F401
    import kinetix  # noqa: F401
    HAVE_JAX = True
except Exception:
    HAVE_JAX = False


@unittest.skipUnless(ENABLED and HAVE_JAX, "kinetix child-env tests (set HARNESSSTUDIO_KINETIX_CHILD_TESTS=1 in the authors' env)")
class KinetixChild(unittest.TestCase):
    LEVEL = "worlds/l/grasp_easy.json"
    UNIVERSE3 = ("worlds/l/grasp_easy.json", "worlds/l/catcher_v3.json", "worlds/l/mjc_swimmer.json")

    def test_prefix_weights_equal_upstream(self):
        import jax.numpy as jnp
        import numpy as np
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        from harnessstudio.methods.rtc import prefix_weights
        M = RT.upstream()["model"]
        for sched in ("exp", "zeros", "linear", "ones"):
            for d in range(5):
                for s in range(max(1, d), 9):
                    a = np.asarray(prefix_weights(jnp, d, 8 - s, 8, sched))
                    b = np.asarray(M.get_prefix_weights(d, 8 - s, 8, sched))
                    self.assertEqual(a.tobytes(), b.tobytes(), (sched, d, s))

    def test_level_keys_match_eval_flow(self):
        import jax
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        case = RT.KinetixUniverseCase(self.UNIVERSE3, rollouts=2, inference_delay=1, execute_horizon=1)
        want = jax.random.split(jax.random.key(0), 3)                 # eval_flow.py:248
        self.assertEqual(jax.device_get(jax.random.key_data(case.level_keys)).tobytes(),
                         jax.device_get(jax.random.key_data(want)).tobytes())

    def test_null_materializes_guidance_level_batched(self):  # EC4 / B4 / R8
        import jax
        import jax.numpy as jnp
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        from harnessstudio.methods.rtc import RTCGuidance
        case = RT.KinetixUniverseCase(self.UNIVERSE3, rollouts=2, inference_delay=1, execute_horizon=1)
        g = RTCGuidance("exp", 5.0)
        obs = jnp.zeros((3, 2, 679), jnp.float32)
        prev = jnp.zeros((3, 2, 8, 6), jnp.float32)
        args = (case.level_keys, obs, prev, case.state_dicts, case.level_keys)
        counts = {}
        for eff in (True, False):
            fn = case.fns["guided"](g, eff)
            jaxpr = str(jax.make_jaxpr(fn)(*args))
            hlo = fn.lower(*args).compile().as_text()
            self.assertNotIn("callback", jaxpr)
            self.assertIn("transpose", jaxpr)              # the VJP survives in the null
            self.assertGreater(hlo.count("dot("), 0)
            counts[eff] = jaxpr.count("dot_general")
        # the null runs every candidate dot (guidance) plus the native batched forward
        self.assertGreaterEqual(counts[False], counts[True])
        print("R8_DOT_GENERAL", counts, file=sys.stderr)

    def test_null_timing_level_batched(self):  # EC4 timing (R8)
        import jax.numpy as jnp
        import numpy as np
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        from harnessstudio.methods.rtc import RTCGuidance
        case = RT.KinetixUniverseCase((self.LEVEL,), rollouts=64, inference_delay=1, execute_horizon=1)
        g = RTCGuidance("exp", 5.0)
        obs = jnp.asarray(np.random.default_rng(0).standard_normal((1, 64, 679)).astype(np.float32))
        prev = jnp.zeros((1, 64, 8, 6), jnp.float32)
        f = case.fns
        calls = {"baseline": lambda: f["naive"](case.level_keys, obs, case.state_dicts, case.level_keys)[1],
                 "null": lambda: f["guided"](g, False)(case.level_keys, obs, prev, case.state_dicts, case.level_keys)[1],
                 "candidate": lambda: f["guided"](g, True)(case.level_keys, obs, prev, case.state_dicts, case.level_keys)[1]}
        times = {}
        for k, fn in calls.items():
            fn().block_until_ready()                       # compile
            reps = []
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(20):
                    fn().block_until_ready()
                reps.append(time.perf_counter() - t0)
            times[k] = statistics.median(reps)
        ratio_nc, ratio_nb = times["null"] / times["candidate"], times["null"] / times["baseline"]
        print("EC4_TIMING", {k: round(v, 4) for k, v in times.items()}, round(ratio_nc, 3), round(ratio_nb, 3), file=sys.stderr)
        self.assertTrue(0.8 <= ratio_nc <= 1.25, (times, ratio_nc))
        self.assertGreaterEqual(ratio_nb, 1.3, (times, ratio_nb))

    def test_worker_smoke_one_level_universe_null_is_m_and_caps(self):
        import json
        from harnessstudio.backends.kinetix_rtc12 import worker as W
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            W.main(["--method", "rtc", "--points", "1:1", "--level-universe", self.LEVEL, "--rollouts", "2",
                    "--out", str(out)])
            att = json.loads((out / "attribution.json").read_text())
            (cid,) = att
            self.assertEqual(att[cid]["null"]["M"], 2)
            res = json.loads((out / "results/null" / cid / "result.json").read_text())
            self.assertEqual(res["guidance_vjp_steps"], 5 * res["decisions"])
            self.assertEqual(len(res["guidance_sha256"]), res["decisions"])
            self.assertEqual(res["decisions"], 256)
            self.assertIsNone(res["success"])
            with self.assertRaises(SystemExit):
                W.main(["--method", "rtc", "--points", "1:1", "--level-universe", self.LEVEL, "--rollouts", "65537",
                        "--out", str(Path(tmp) / "capped")])
            with self.assertRaises(Exception):
                W.main(["--method", "medoid-bon", "--points", "1:1", "--level-universe", self.LEVEL, "--rollouts", "2",
                        "--out", str(Path(tmp) / "refused")])

    def test_level_batched_first_steps_match_probe(self):  # unit test, never a parity verdict
        import numpy as np
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        from harnessstudio.backends.kinetix_rtc12 import worker as W
        from harnessstudio.core import registry
        from harnessstudio.policies import kinetix_flow as KF
        probe_dir = KF.kinetix_root() / "scripts"
        if not (probe_dir / "ref_probe.py").is_file():
            self.skipTest("ref_probe.py not available")
        sys.path.insert(0, str(probe_dir))
        import ref_probe as RP
        import eval_flow
        import jax
        B, K = 4, 20
        cwd = os.getcwd()
        try:  # ref_probe.build loads levels by path relative to the upstream checkout, as eval_flow.main
            os.chdir(KF.kinetix_root() / "upstream")
            _probe, levels, sds = RP.build(list(self.UNIVERSE3), str(KF.kinetix_root() / "assets" / "bc"))
        finally:
            os.chdir(cwd)
        for mname, mcfg, arm in (("identity", eval_flow.NaiveMethodConfig(), "baseline"),
                                 ("rtc", eval_flow.RealtimeMethodConfig(), "candidate")):
            cfg = eval_flow.EvalConfig(num_evals=B, inference_delay=1, execute_horizon=1, method=mcfg)
            ref = jax.device_get(_probe(cfg, jax.random.split(jax.random.key(0), 3), levels, sds))
            case = RT.KinetixUniverseCase(self.UNIVERSE3, rollouts=B, inference_delay=1, execute_horizon=1)
            _rec, port, _cl = W.run_case(registry.get(mname)[0], arm, case, record_steps=True)
            ex = np.concatenate(port.executed, axis=1)      # [L, T, B, A]
            for li in range(3):
                self.assertEqual(ex[li, :K].tobytes(), np.asarray(ref["executed"][li][:K]).tobytes(), (mname, li))


if __name__ == "__main__":
    unittest.main()
