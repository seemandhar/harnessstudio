"""scripts/witness.py on synthetic arm folders: RC4 pass/fail cases, M checks and the short-run metric. numpy only;
when harnessstudio is importable, the re-implemented action fingerprint is checked against the framework's."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "witness.py"
spec = importlib.util.spec_from_file_location("scale_gtok_witness", SCRIPT)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

TASK = "libero_10/put both moka pots on the stove#8"


def write_arm(root: Path, episodes: dict) -> str:
    """episodes: {episode_idx: (actions [n,7], obs_tags list[n], success)} -> an hs-style arm folder."""
    root.mkdir(parents=True)
    ep_dir = root / "vla_eval" / "episodes" / "LIBEROBenchmark_libero_10"
    ep_dir.mkdir(parents=True)
    with open(root / "records.jsonl", "w") as f:
        for i, (acts, obs, success) in episodes.items():
            acts = np.asarray(acts, dtype=np.float32)
            rec = {"benchmark": "libero", "task": TASK, "episode_idx": i, "valid": True, "success": success,
                   "accounting": {"steps": len(acts), "policy_seconds": 0.75 * len(acts)},
                   "action_sha256": [W.action_fingerprint(a) for a in acts], "obs_sha256": [str(o) for o in obs]}
            f.write(json.dumps(rec) + "\n")
            with open(ep_dir / f"task0000_ep{i:04d}_{'success' if success else 'fail'}.jsonl", "w") as g:
                for k, a in enumerate(acts):
                    g.write(json.dumps({"step": k, "action": [float(x) for x in a], "state": [0.0] * 8}) + "\n")
    return str(root)


def base_actions(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.3, size=(n, 7)).astype(np.float32)
    a[:, 6] = np.where((np.arange(n) // 40) % 2 == 0, -1.0, 1.0)
    return a


class Witness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        ref = base_actions()
        obs = list(range(len(ref)))
        self.ref = write_arm(self.tmp / "ref", {0: (ref, obs, False), 1: (ref, obs, True), 2: (ref, obs, False),
                                                3: (ref, obs, False), 4: (ref, obs, True)})
        good = ref.copy()
        good[50, 6] = -good[50, 6]            # case 0: first difference only in the gripper at step 50
        good[51:] = base_actions(seed=1)[51:]  # ... then the episode re-draws (allowed after the first difference)
        good[51:, 6] = 1.0
        bad = ref.copy()
        bad[30, 2] += np.float32(1e-3)        # case 1: first difference in dim 2
        obs_branch = list(obs)
        obs_branch[20] = "renderer"           # case 3: obs differs before the first action difference
        branch = ref.copy()
        branch[25, 6] = -branch[25, 6]
        short = ref[:150]                      # case 4: no action difference but a shorter episode
        self.cand = write_arm(self.tmp / "cand", {0: (good, obs, True), 1: (bad, obs, True), 2: (ref, obs, False),
                                                  3: (branch, obs_branch, False), 4: (short, obs[:150], True)})

    def test_fingerprint_matches_framework(self):
        try:
            from harnessstudio.context.trace import fingerprint
        except ImportError:
            self.skipTest("harnessstudio not importable here")
        a = np.array([0.1, -0.2, 0.3, 0.0, 1e-7, -0.5, 1.0], dtype=np.float32)
        self.assertEqual(W.action_fingerprint(a), fingerprint(a))

    def test_rc4_classification(self):
        out = self.tmp / "w.json"
        rc = W.main(["--candidate", self.cand, "--scale-ref", self.ref, "--greedy-ref", self.ref, "--out", str(out)])
        rep = json.loads(out.read_text())["candidate"]
        per = rep["witness_rc4"]["per_case"]
        self.assertTrue(per["t8i0"]["witness_ok"])
        self.assertEqual(per["t8i0"]["first_divergence"], 50)
        self.assertEqual(per["t8i0"]["dims_differing"], [6])
        self.assertFalse(per["t8i1"]["witness_ok"])
        self.assertEqual(per["t8i1"]["dims_differing"], [2])
        self.assertTrue(per["t8i2"]["witness_ok"])
        self.assertFalse(per["t8i2"]["diverged"])
        self.assertFalse(per["t8i3"]["witness_ok"])
        self.assertIn("renderer", per["t8i3"]["reason"])
        self.assertFalse(per["t8i4"]["witness_ok"])
        self.assertEqual(sorted(rep["witness_rc4"]["violations"]), ["t8i1", "t8i3", "t8i4"])
        self.assertEqual(rc, 1)

    def test_m_check(self):
        out = self.tmp / "m.json"
        self.assertEqual(W.main(["--null", self.ref, "--scale-ref", self.ref, "--greedy-ref", self.ref,
                                 "--out", str(out)]), 0)
        self.assertEqual(W.main(["--null", self.cand, "--scale-ref", self.ref, "--greedy-ref", self.ref,
                                 "--out", str(out)]), 1)

    def test_short_runs(self):
        a = np.zeros((12, 7), dtype=np.float32)
        a[:, 6] = [-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1]
        # runs: 3(-1) 1(+1) 2(-1) 2(+1) 3(-1) 1(+1); interior runs <= 2: 1, 2, 2 -> 3 (first/last excluded)
        self.assertEqual(W.gripper_short_runs(a), (3, 12))


if __name__ == "__main__":
    unittest.main()
