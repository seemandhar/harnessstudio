"""Kinetix RTC extension: tests that need JAX and the pinned upstream (authors' env).

Self-skip unless HARNESSSTUDIO_KINETIX_CHILD_TESTS=1 and jax/kinetix import. Run with the kinetix child
interpreter, JAX_PLATFORMS=cpu and PYTHONPATH=<worktree>:
    HARNESSSTUDIO_KINETIX_CHILD_TESTS=1 JAX_PLATFORMS=cpu PYTHONPATH=. $KINETIX_ENV/bin/python -m unittest tests.test_kinetix_child -v
"""
from __future__ import annotations

import os
import tempfile
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

    def test_key_chain_order_matches_eval_flow(self):
        import jax
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        lk = jax.random.split(jax.random.key(0), 12)[0]
        chain = RT.KeyChain(lk)
        got = [jax.random.key_data(chain.split()) for _ in range(5)]
        rng, want = lk, []
        for _ in range(5):
            rng, key = jax.random.split(rng)
            want.append(jax.random.key_data(key))
        for g, w in zip(got, want):
            self.assertEqual(bytes(memoryview(jax.device_get(g))), bytes(memoryview(jax.device_get(w))))

    def test_null_materializes_guidance_and_has_no_callbacks(self):  # EC4 / EC3
        import jax
        import jax.numpy as jnp
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        from harnessstudio.methods.rtc import RTCGuidance
        case = RT.KinetixCase(self.LEVEL, rollouts=2, inference_delay=1, execute_horizon=1)
        g = RTCGuidance("exp", 5.0)
        obs = jnp.zeros((2, 679), jnp.float32)
        prev = jnp.zeros((2, 8, 6), jnp.float32)
        texts = {}
        for eff in (True, False):
            fn = case.fns["guided"](g, eff)
            texts[eff] = str(jax.make_jaxpr(fn)(case.params, jax.random.key(0), obs, prev))
            hlo = fn.lower(case.params, jax.random.key(0), obs, prev).compile().as_text()
            self.assertNotIn("callback", texts[eff])
            self.assertIn("transpose", texts[eff])          # the VJP survives in the null
            self.assertGreater(hlo.count("dot("), 0)
        self.assertGreaterEqual(texts[False].count("dot_general"), texts[True].count("dot_general"))

    def test_worker_smoke_null_is_m_and_caps(self):
        import json
        from harnessstudio.backends.kinetix_rtc12 import worker as W
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            W.main(["--method", "rtc", "--points", "1:1", "--levels", self.LEVEL, "--rollouts", "2", "--out", str(out)])
            att = json.loads((out / "attribution.json").read_text())
            self.assertEqual(att["d1s1--grasp_easy--b0"]["null"]["M"], 2)
            res = json.loads((out / "results/null/d1s1--grasp_easy--b0/result.json").read_text())
            self.assertEqual(res["guidance_vjp_steps"], 5 * res["decisions"])
            self.assertEqual(res["decisions"], 256)
            self.assertIsNone(res["success"])
            with self.assertRaises(SystemExit):
                W.main(["--method", "rtc", "--points", "1:1", "--levels", self.LEVEL, "--rollouts", "65537",
                        "--out", str(Path(tmp) / "capped")])
            with self.assertRaises(Exception):
                W.main(["--method", "medoid-bon", "--points", "1:1", "--levels", self.LEVEL, "--rollouts", "2",
                        "--out", str(Path(tmp) / "refused")])


if __name__ == "__main__":
    unittest.main()
