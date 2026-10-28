import json
import unittest

from harnessstudio.core.method import validate
from harnessstudio.methods import chunk_interrupt, chunktrust_ahs, medoid_bon


G = {"policy": {"action_layout": {"action.end_effector_position": (5, 8), "action.end_effector_rotation": (8, 11)},
                 "proprio_eef": ("state.end_effector_position_relative", None)}}
P = {"policy": {"action_layout": {"action.end_effector_position": (0, 3), "action.end_effector_rotation": (3, 6)},
                 "proprio_eef": ("observation/state", (0, 3))}}


class Chunk:
    def __init__(self, actions):
        self.actions = actions


class Obs:
    def __init__(self, values):
        self.values = values


def eef(x):
    return {"state.end_effector_position_relative": [[x, 0.0, 0.0]]}


class MBRTest(unittest.TestCase):
    def test_picks_the_consensus_sample(self):
        sel = medoid_bon.MBRSelect()
        # Mean L2 risk: +0.1 and -0.1 are equidistant from 0 but far from each other -> 0 wins. No ties.
        chunks = [Chunk([[0.1] * 12] * 16), Chunk([[0.0] * 12] * 16), Chunk([[-0.1] * 12] * 16)]
        self.assertEqual(sel.select(chunks, G, None), 1)

    def test_feature_is_cumulative_eef_and_ignores_gripper_and_base(self):
        sel = medoid_bon.MBRSelect()
        row = [9.0] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] + [9.0]  # dims 0-4 base/control, 11 gripper
        f = sel._feature([row, row], sel._resolve(G))
        self.assertEqual(f, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        a = [[0.0] * 12] * 16
        b = [[5.0] * 5 + [0.0] * 6 + [1.0]] * 16  # differs only in base/control/gripper
        c = [[0.0] * 5 + [0.3] * 6 + [0.0]] * 16
        self.assertEqual(sel.select([Chunk(c), Chunk(a), Chunk(b)], G, None), 1)

    def test_pi05_layout_resolves_first_six_dims(self):
        sel = medoid_bon.MBRSelect()
        self.assertEqual(sel._resolve(P), [0, 1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            sel._resolve({})

    def test_rejects_bad_dims(self):
        with self.assertRaises(ValueError):
            medoid_bon.MBRSelect(dims=[])

    def test_config_valid(self):
        config = validate(medoid_bon.config())
        self.assertEqual(config.paper, "arXiv:2601.02295v2")
        self.assertEqual(config.version, "2")
        self.assertEqual(config.slots["proposer"].params["num_samples"], 8)


class StallCutTest(unittest.TestCase):
    def test_cuts_after_window_of_stall(self):
        ex = chunk_interrupt.StallCut(min_k=4, per_step=True, window=4, rho=0.05)
        rows = [[0.0] * 5 + [1.0, 0.0, 0.0] + [0.0] * 4] * 16  # commands 0.05 m/step in x
        ex.plan(Chunk(rows), G, Obs(eef(0.0)))
        moving = [ex.continue_after(i, Obs(eef(0.05 * i))) for i in range(1, 5)]
        self.assertTrue(all(moving))
        ex.plan(Chunk(rows), G, Obs(eef(1.0)))
        stalled = [ex.continue_after(i, Obs(eef(1.0))) for i in range(1, 9)]
        self.assertEqual(stalled.index(False), 6)  # low ratio at 4,5,6,7 -> False at step 7

    def test_pi05_proprio_slice(self):
        ex = chunk_interrupt.StallCut(min_k=4, per_step=True, window=4, rho=0.05)
        rows = [[1.0, 0.0, 0.0] + [0.0] * 9] * 5
        state = lambda x: {"observation/state": [x, 0.0, 0.0] + [9.0] * 13}
        ex.plan(Chunk(rows), P, Obs(state(0.0)))
        self.assertEqual(ex.slice, (0, 3))
        self.assertTrue(all(ex.continue_after(i, Obs(state(0.05 * i))) for i in range(1, 5)))

    def test_config_valid(self):
        validate(chunk_interrupt.config())

    def test_controller_config_pinned(self):
        import hashlib
        path = chunk_interrupt.controller_config_path()
        if path is None:
            self.skipTest("robosuite not importable and HARNESSSTUDIO_ROBOSUITE_ROOT unset")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), chunk_interrupt.CONTROLLER_CONFIG_SHA256)
        arm = json.loads(path.read_text())["body_parts"]["arms"]["right"]
        self.assertEqual(arm["type"], "OSC_POSE")
        self.assertEqual(arm["input_max"], 1)
        self.assertEqual(arm["output_max"][:3], [chunk_interrupt.CONTROLLER_SCALE_M] * 3)
        self.assertEqual(arm["input_ref_frame"], "base")


class AHSTest(unittest.TestCase):
    def test_config_and_plan(self):
        import numpy as np
        validate(chunktrust_ahs.config())
        ex = chunktrust_ahs.modules.AHSExecutor(**chunktrust_ahs.config().slots["executor"].params)
        rng = np.random.default_rng(0)
        flat = rng.standard_normal((16, 12)).astype(np.float32)
        vel = rng.standard_normal((4, 16, 32)).astype(np.float32)
        k = ex.plan(Chunk(flat), {"sampler": {"velocity": vel}}, None)
        self.assertTrue(4 <= k <= 16)  # expected_round can return any integer in [4, 16]
        ex.record_executed(flat[:k])

    def test_params_match_reference_config(self):
        # Identical to the reference config of the original ChunkTrust v2.2 run (ahs_reference.json).
        self.assertEqual(chunktrust_ahs.AHS_PARAMS["horizon_candidates"], [4, 8, 12, 16])
        self.assertEqual(len(chunktrust_ahs.AHS_PARAMS), 19)

    def test_vendored_files_match_provenance(self):
        import hashlib
        from pathlib import Path
        d = Path(chunktrust_ahs.__file__).parent
        prov = json.loads((d / "PROVENANCE.json").read_text())
        for name, digest in prov["files"].items():
            self.assertEqual(hashlib.sha256((d / name).read_bytes()).hexdigest(), digest, name)
