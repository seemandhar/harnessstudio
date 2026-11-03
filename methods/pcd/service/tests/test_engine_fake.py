"""Fake-model unit tests (no weights, no torch model): parsing, mask rules, state machine, evidence, HTTP.

Run: PY -I -B -m unittest discover -s tests -t .   (from the service directory; PY = the perception venv python)
"""
import base64
import io
import json
import os
import sys
import threading
import unittest
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcd_perception import masks as M  # noqa: E402
from pcd_perception.engine import CounterfactualEngine  # noqa: E402
from pcd_perception.vocab import (CLASSNAMES, GDINO_ALIASES, InstructionParseError, alias_to_name,  # noqa: E402
                                  gdino_text, label_to_name, parse_objects)

H, W = 32, 40


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


class FakeTracker:
    def __init__(self, script, classes=None):
        self.script = script  # list of dict name->mask or None, one per track() call
        self.classes = list(classes or CLASSNAMES)
        self.loaded = None
        self.added = []
        self.calls = 0

    def load_first_frame(self, image):
        self.loaded = image.copy()

    def add_new_mask(self, frame_idx, obj_id, mask):
        self.added.append((frame_idx, obj_id, mask.copy()))

    def track(self, image):
        frame = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        ids = [oid for _, oid, _ in self.added]
        logits = []
        for oid in ids:
            m = frame.get(self.classes[oid - 1])
            logits.append(np.where(m, 1.0, -1.0)[None] if m is not None else -np.ones((1, H, W)))
        return ids, np.stack(logits) if logits else np.zeros((0, 1, H, W))


class FakeDetector:
    def __init__(self, script):
        self.script = script  # list of dict name->mask, one per call
        self.calls = 0

    def __call__(self, image, prompts, **kw):
        frame = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        self.last_kw = kw
        return [M.postprocess_detection_mask(frame.get(p)) for p in prompts]


def fake_inpaint(image, region):
    out = image.copy()
    out[region] = 7
    return out


def frame(v=100):
    return np.full((H, W, 3), v, np.uint8)


class TestVocab(unittest.TestCase):
    def test_simpler_instructions(self):
        self.assertEqual(parse_objects("pick coke can"), ["coke can"])
        self.assertEqual(parse_objects("move redbull can near apple"), ["redbull can", "apple"])
        self.assertEqual(parse_objects("close top drawer"), ["top drawer"])
        self.assertEqual(parse_objects("put carrot on plate"), ["carrot", "plate larger"])
        self.assertEqual(parse_objects("put the spoon on the towel"), ["spoon", "table cloth shorter"])
        self.assertEqual(parse_objects("stack the green block on the yellow block"), ["green cube", "yellow cube"])
        self.assertEqual(parse_objects("put eggplant into yellow basket"), ["eggplant", "yellow basket"])
        self.assertEqual(parse_objects("place apple into top drawer"), ["apple", "top drawer"])

    def test_no_lowercasing_and_errors(self):
        with self.assertRaises(InstructionParseError):
            parse_objects("Pick coke can")  # capital P: no template matches (instruction_templates.py:33-44)
        self.assertEqual(parse_objects("pick Coke Can"), ["Coke Can"])
        self.assertEqual(parse_objects("close top drawer", get_all_parts=True),
                         ["top drawer", "body", "middle drawer", "bottom drawer"])

    def test_gdino_text_and_labels(self):
        self.assertEqual(gdino_text(["coke can", "robot"], GDINO_ALIASES), "coke pop can. robot manipulator.")
        rev = alias_to_name(["coke can", "robot"], GDINO_ALIASES)
        self.assertEqual(label_to_name("coke pop", rev), "coke can")
        self.assertEqual(label_to_name("robot manipulator", rev), "robot")
        self.assertEqual(label_to_name("dresser", alias_to_name(["top drawer", "robot"], GDINO_ALIASES)), "top drawer")


class TestMasks(unittest.TestCase):
    def test_postprocess(self):
        self.assertIsNone(M.postprocess_detection_mask(np.zeros((H, W), bool)))
        m = box(2, 22, 2, 22)
        m[10, 10] = False           # hole -> filled
        m[28, 35] = True            # tiny component (< 0.1 x largest) -> dropped
        p = M.postprocess_detection_mask(m)
        self.assertTrue(p[10, 10])
        self.assertFalse(p[28, 35])

    def test_regions(self):
        target, robot = box(5, 10, 5, 10), box(8, 20, 8, 20)
        union = M.target_union((H, W), [target], [robot])
        self.assertFalse(union[9, 9])  # robot pixels cleared from the target union
        inp, paste = M.inpaint_regions(union, robot)
        self.assertTrue(inp[19, 19] and not paste[9, 9] and paste[4, 4])


class TestEngine(unittest.TestCase):
    def make(self, det_script, track_script):
        self.trackers = []

        def factory():
            t = FakeTracker(track_script)
            self.trackers.append(t)
            return t
        return CounterfactualEngine(FakeDetector(det_script), factory, fake_inpaint)

    def test_detect_until_first_hit_then_track_only(self):
        can, robot = box(4, 12, 4, 12), box(20, 30, 20, 30)
        eng = self.make([{}, {}, {"coke can": can, "robot": robot}, {"coke can": can}],
                        [{"coke can": can, "robot": robot}, {"robot": robot}, {"coke can": can, "robot": robot}])
        outs = [eng.step(frame(), "pick coke can") for _ in range(6)]
        self.assertEqual([o["phase"] for o in outs], ["detect", "detect", "first_detection", "track", "track",
                                                      "track"])
        self.assertEqual(eng.detector.calls, 3)            # never re-detects after tracking starts
        self.assertEqual(outs[0]["per_target"]["coke can"]["mask_present"], False)
        self.assertFalse(outs[0]["cf_differs"])            # empty mask: LaMa still runs, nothing pasted
        self.assertEqual(eng.calls["lama"], 6)            # inpainting every step
        self.assertTrue(outs[2]["cf_differs"])
        self.assertTrue(outs[4]["per_target"]["coke can"]["tracker_none"])
        ep = outs[-1]["episode"]
        self.assertEqual(ep["first_detection_step"], 2)
        self.assertEqual(ep["tracker_none_steps"]["coke can"], [4])
        self.assertFalse(ep["robot_only_first_detection"])
        ids = [oid for _, oid, _ in self.trackers[0].added]
        self.assertEqual(ids, [CLASSNAMES.index("coke can") + 1, CLASSNAMES.index("robot") + 1])

    def test_robot_only_first_detection_never_masks_target(self):
        can, robot = box(4, 12, 4, 12), box(20, 30, 20, 30)
        eng = self.make([{"robot": robot}, {"coke can": can, "robot": robot}], [{"robot": robot}])
        outs = [eng.step(frame(), "pick coke can") for _ in range(4)]
        ep = outs[-1]["episode"]
        self.assertTrue(ep["robot_only_first_detection"])
        self.assertEqual(ep["targets_never_masked"], ["coke can"])
        self.assertTrue(all(not o["cf_differs"] for o in outs))
        self.assertEqual(eng.detector.calls, 1)

    def test_instruction_change_rebuilds(self):
        apple, robot = box(4, 12, 4, 12), box(20, 30, 20, 30)
        eng = self.make([{"top drawer": apple, "robot": robot}], [{"top drawer": apple, "robot": robot}])
        eng.step(frame(), "open top drawer")
        eng.step(frame(), "open top drawer")
        out = eng.step(frame(), "place apple into top drawer")
        self.assertEqual(out["phase"], "first_detection")
        self.assertEqual(len(self.trackers), 2)
        self.assertEqual(out["episode"]["instruction_rebuilds"], 1)
        self.assertEqual(out["step"], 2)

    def test_unknown_class_refused(self):
        eng = self.make([{}], [{}])
        with self.assertRaises(ValueError):
            eng.step(frame(), "pick bowl")


class TestLiberoGoalParser(unittest.TestCase):
    """Setting B frozen table (parsers/libero_goal.json): every public libero_goal instruction parses; SIMPLER default."""

    EXPECTED = {
        "open the middle drawer of the cabinet": ["middle drawer", "cabinet"],
        "put the bowl on the stove": ["bowl", "stove"],
        "put the wine bottle on top of the cabinet": ["wine bottle", "cabinet"],
        "open the top drawer and put the bowl inside": ["top drawer", "bowl"],
        "put the bowl on top of the cabinet": ["bowl", "cabinet"],
        "push the plate to the front of the stove": ["plate", "stove"],
        "put the cream cheese in the bowl": ["cream cheese", "bowl"],
        "turn on the stove": ["stove"],
        "put the bowl on the plate": ["bowl", "plate"],
        "put the wine bottle on the rack": ["wine bottle", "rack"],
    }

    def setUp(self):
        from pcd_perception import parsers
        self.parsers = parsers
        self.spec = parsers.load("libero_goal")

    def test_all_ten_instructions(self):
        import json
        table = json.load(open(os.path.join(os.path.dirname(self.parsers.__file__), "libero_goal.json")))
        self.assertEqual(table["instructions"], list(self.EXPECTED))
        for ins, objs in self.EXPECTED.items():
            self.assertEqual(self.spec.parse(ins), objs, ins)
            for o in objs + ["robot"]:
                self.assertIn(o, self.spec.classnames)
        self.assertEqual(self.spec.classnames[0], "robot")
        self.assertEqual(self.spec.keep_objects, ("robot",))
        self.assertEqual(self.spec.aliases, {"robot": "robot manipulator", "cabinet": "black cabinet",
                                             "middle drawer": "drawer"})

    def test_no_simpler_rewrites_and_no_lowercasing(self):
        self.assertEqual(self.spec.parse("put the bowl on the plate"), ["bowl", "plate"])  # not 'plate larger'
        with self.assertRaises(InstructionParseError):
            self.spec.parse("Put the bowl on the plate")
        with self.assertRaises(InstructionParseError):
            self.spec.parse("pick coke can")

    def test_frozen_pin(self):
        import hashlib
        path = os.path.join(os.path.dirname(self.parsers.__file__), "libero_goal.json")
        self.assertEqual(hashlib.sha256(open(path, "rb").read()).hexdigest(), self.parsers.FROZEN["libero_goal"][1])

    def test_engine_modes(self):
        bowl, plate, robot = box(2, 8, 2, 8), box(10, 16, 10, 16), box(20, 30, 20, 30)
        det = FakeDetector([{"bowl": bowl, "plate": plate, "robot": robot}])
        trackers = []

        def factory():
            trackers.append(FakeTracker([{"bowl": bowl, "plate": plate, "robot": robot}], self.spec.classnames))
            return trackers[-1]
        eng = CounterfactualEngine(det, factory, fake_inpaint)
        self.assertEqual(eng.parser, "simpler")
        with self.assertRaises(ValueError):          # SIMPLER default: 'bowl' is not a SIMPLER class
            eng.reset("put the bowl on the plate")
        summary = eng.reset("put the bowl on the plate", parser="libero_goal")
        self.assertEqual(summary["targets"], ["bowl", "plate"])
        self.assertEqual(summary["parser"], "libero_goal")
        out = eng.step(frame(), "put the bowl on the plate")
        self.assertEqual(det.last_kw["aliases"]["robot"], "robot manipulator")
        ids = [oid for _, oid, _ in trackers[-1].added]
        self.assertEqual(ids, [self.spec.classnames.index(n) + 1 for n in ("bowl", "plate", "robot")])
        out = eng.step(frame(), "put the bowl on the plate")
        self.assertEqual(out["phase"], "track")
        self.assertTrue(out["per_target"]["plate"]["mask_present"])
        eng.reset("put the bowl on the plate")        # mode is sticky until a reset names another
        self.assertEqual(eng.parser, "libero_goal")
        eng.reset("pick coke can", parser="simpler")
        self.assertEqual(eng.state.targets, ["coke can"])
        self.assertEqual(det.calls, 1)
        eng.step(frame(), "pick coke can")
        self.assertEqual(det.last_kw, {})                # SIMPLER: detector called exactly as before

    def test_unknown_parser(self):
        with self.assertRaises(ValueError):
            self.parsers.load("libero_10")


class TestTorchFlags(unittest.TestCase):
    def test_authors_cudnn_flags(self):
        import torch
        from pcd_perception.runtime import set_authors_torch_flags, torch_flags
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
        flags = set_authors_torch_flags()
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertEqual((flags["cudnn_deterministic"], flags["cudnn_benchmark"]), (True, False))
        self.assertEqual(torch_flags(), flags)


class TestHTTP(unittest.TestCase):
    def test_roundtrip(self):
        from http.server import ThreadingHTTPServer
        from pcd_perception.client import PerceptionClient
        from pcd_perception.server import Service, make_handler
        can, robot = box(4, 12, 4, 12), box(20, 30, 20, 30)

        def factory():
            return CounterfactualEngine(FakeDetector([{"coke can": can, "robot": robot}]),
                                        lambda: FakeTracker([{"coke can": can, "robot": robot}]), fake_inpaint)
        svc = Service(factory, "cpu", {"fake": True}, False)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            cli = PerceptionClient(f"http://127.0.0.1:{server.server_address[1]}", session="e1")
            health = cli.health()
            self.assertTrue(health["ok"])
            self.assertIn("cudnn_deterministic", health["torch_flags"])
            cli.reset("pick coke can")
            img = frame()
            cf, out = cli.step(img, "pick coke can", return_masks=True)
            self.assertEqual(cf.shape, img.shape)
            self.assertEqual(out["phase"], "first_detection")
            self.assertIn("masks_npy", out)
            cf2, out2 = cli.step(img, "pick coke can")
            self.assertEqual(out2["phase"], "track")
            self.assertEqual(out2["calls"]["lama"], 2)
            # parser mode at /reset (field "parser"); a fresh session defaults to "simpler"
            r = PerceptionClient(cli.url, session="lg").reset("turn on the stove", parser="libero_goal")
            self.assertEqual((r["parser"], r["targets"]), ("libero_goal", ["stove"]))
            import urllib.error
            with self.assertRaises(urllib.error.HTTPError) as ctx:   # unknown mode refused, never a fallback
                PerceptionClient(cli.url, session="bad").reset("turn on the stove", parser="libero_10")
            self.assertEqual(ctx.exception.code, 500)
            self.assertIn("unknown parser", json.loads(ctx.exception.read())["error"])
            # a second session has its own episode state
            other = PerceptionClient(cli.url, session="e2")
            _, o3 = other.step(img, "pick coke can")
            self.assertEqual(o3["phase"], "first_detection")
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
