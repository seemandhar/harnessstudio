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

    def _null_programs(self, case, g):
        """The null's actual compiled calls per decision (repair 2): the naive program and the candidate program."""
        return case.fns["naive"], case.fns["guided"](g, True)

    def test_null_materializes_guidance_level_batched(self):  # EC4 / B4 / R8, repair-2 P2(c)
        import jax
        import jax.numpy as jnp
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        from harnessstudio.methods.rtc import RTCGuidance
        case = RT.KinetixUniverseCase(self.UNIVERSE3, rollouts=2, inference_delay=1, execute_horizon=1)
        g = RTCGuidance("exp", 5.0)
        obs = jnp.zeros((3, 2, 679), jnp.float32)
        prev = jnp.zeros((3, 2, 8, 6), jnp.float32)
        _naive, shadow = self._null_programs(case, g)
        args = (case.level_keys, obs, prev, case.state_dicts, case.level_keys)
        jaxpr = str(jax.make_jaxpr(shadow)(*args))
        hlo = shadow.lower(*args).compile().as_text()
        self.assertNotIn("callback", jaxpr)
        self.assertIn("transpose", jaxpr)                  # the null's shadow program keeps the VJP
        self.assertGreater(hlo.count("dot("), 0)
        out = shadow(*args)
        self.assertEqual(len(out), 3)                      # (rng, chunk, guided stack): the stack is a jit output
        with self.assertRaises(RuntimeError):              # the pre-repair-2 null program is retired (P1)
            case.fns["guided"](g, False)

    def test_null_calls_naive_and_candidate_shadow_only(self):  # repair-2 P2(a)
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        from harnessstudio.backends.kinetix_rtc12 import worker as W
        from harnessstudio.core import registry
        case = RT.KinetixUniverseCase((self.LEVEL,), rollouts=2, inference_delay=1, execute_horizon=1)
        fns = case.fns
        orig_naive, orig_guided = fns["naive"], fns["guided"]
        calls = []

        def naive(*a):
            calls.append("naive")
            return orig_naive(*a)

        def guided(sampler, effective):
            calls.append(f"guided({bool(effective)})")
            return orig_guided(sampler, effective)

        fns["naive"], fns["guided"] = naive, guided
        try:
            rec, port, client = W.run_case(registry.get("rtc")[0], "null", case)
        finally:
            fns["naive"], fns["guided"] = orig_naive, orig_guided
        decisions = rec["decisions"]
        self.assertEqual(calls[0], "naive")                # warm start (protocol, every arm)
        per_decision = calls[1:]
        self.assertEqual(len(per_decision), 2 * decisions)
        self.assertEqual(per_decision, ["naive", "guided(True)"] * decisions)
        self.assertNotIn("guided(False)", calls)
        self.assertEqual(len(client.guidance_sha256), decisions)

    def test_null_cost_is_candidate_plus_baseline(self):  # A5 on the null's ACTUAL calls (repair-2 P2(b)); timing report-only (T1)
        """null = baseline program + candidate program per decision, by construction (repair 2).

        Binding: dots(null's naive call) + dots(null's shadow call) == dots(baseline) + dots(candidate), in jaxpr and
        compiled HLO. The retired EC4 window FAILED at 1.476 / 1.493 (eb7721e5) and the A5 timing assertion at
        1.294 (ba8fda55, a5/child-tests.log); timing is printed only.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from harnessstudio.backends.kinetix_rtc12 import runtime as RT
        from harnessstudio.methods.rtc import RTCGuidance
        g = RTCGuidance("exp", 5.0)
        small = RT.KinetixUniverseCase((self.LEVEL,), rollouts=2, inference_delay=1, execute_horizon=1)
        obs2 = jnp.zeros((1, 2, 679), jnp.float32)
        prev2 = jnp.zeros((1, 2, 8, 6), jnp.float32)
        a_naive = (small.level_keys, obs2, small.state_dicts, small.level_keys)
        a_guided = (small.level_keys, obs2, prev2, small.state_dicts, small.level_keys)
        null_naive, null_shadow = self._null_programs(small, g)
        programs = {"baseline": (small.fns["naive"], a_naive), "candidate": (small.fns["guided"](g, True), a_guided),
                    "null_naive": (null_naive, a_naive), "null_shadow": (null_shadow, a_guided)}
        jd, hd = {}, {}
        for k, (fn, args) in programs.items():
            jd[k] = str(jax.make_jaxpr(fn)(*args)).count("dot_general")
            hd[k] = fn.lower(*args).compile().as_text().count(" dot(")
        print("A5_DOTS", jd, hd, file=sys.stderr)
        self.assertGreater(jd["baseline"], 0)
        self.assertEqual(jd["null_naive"] + jd["null_shadow"], jd["baseline"] + jd["candidate"])
        self.assertEqual(hd["null_naive"] + hd["null_shadow"], hd["baseline"] + hd["candidate"])

        case = RT.KinetixUniverseCase((self.LEVEL,), rollouts=64, inference_delay=1, execute_horizon=1)
        obs = jnp.asarray(np.random.default_rng(0).standard_normal((1, 64, 679)).astype(np.float32))
        prev = jnp.zeros((1, 64, 8, 6), jnp.float32)
        f = case.fns
        an = (case.level_keys, obs, case.state_dicts, case.level_keys)
        ag = (case.level_keys, obs, prev, case.state_dicts, case.level_keys)

        def null_call():
            f["naive"](*an)[1].block_until_ready()
            return f["guided"](g, True)(*ag)[1]
        calls = {"baseline": lambda: f["naive"](*an)[1], "null": null_call, "candidate": lambda: f["guided"](g, True)(*ag)[1]}
        for fn in calls.values():
            fn().block_until_ready()                       # compile
        load_before = os.getloadavg()
        reps = {k: [] for k in calls}
        for _ in range(3):                                 # arms interleaved round-robin per rep
            for k, fn in calls.items():
                t0 = time.perf_counter()
                for _ in range(20):
                    fn().block_until_ready()
                reps[k].append(time.perf_counter() - t0)
        load_after = os.getloadavg()
        times = {k: statistics.median(v) for k, v in reps.items()}
        nc, nb = times["null"] / times["candidate"], times["null"] / times["baseline"]
        additive = times["null"] / (times["candidate"] + times["baseline"])
        print("A5_TIMING", {k: round(v, 4) for k, v in times.items()},
              {"null/candidate": round(nc, 3), "null/baseline": round(nb, 3), "null/(candidate+baseline)": round(additive, 3)},
              {"window": "null/candidate >= 0.8, null/baseline >= 1.3, null/(candidate+baseline) in [0.8, 1.25]"},
              {"loadavg_before": load_before, "loadavg_after": load_after}, file=sys.stderr)
        if not (nc >= 0.8 and nb >= 1.3 and 0.8 <= additive <= 1.25):
            print("A5_TIMING WARN outside window (report-only; indicative on a shared host)", file=sys.stderr)

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
            cand = json.loads((out / "results/candidate" / cid / "result.json").read_text())
            self.assertEqual(res["guidance_sha256"][0], cand["guidance_sha256"][0])   # shadow == candidate at decision 0 (P2d)
            self.assertEqual(res["protocol"]["null_mode"], "baseline_program+candidate_shadow")
            self.assertIsNone(res["protocol"]["jax_default_matmul_precision"])
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
