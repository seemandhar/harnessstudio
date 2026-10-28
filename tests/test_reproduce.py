"""CPU tests for reproduce acceptance and quick-eval record handling, on SYNTHETIC run directories."""
import json
import tempfile
import unittest
from pathlib import Path

import tests  # noqa: F401
from harnessstudio.eval import reproduce


class AcceptanceTest(unittest.TestCase):
    def test_rules(self):
        ok = reproduce.acceptance({"cases": 32, "M": 25, "R": 7, "X": 0, "ri_missing": []})
        self.assertTrue(ok["passed"])
        self.assertEqual(ok["R_bound"], 20)
        self.assertFalse(reproduce.acceptance({"cases": 32, "M": 31, "R": 0, "X": 1, "ri_missing": []})["passed"])
        all_r = reproduce.acceptance({"cases": 8, "M": 0, "R": 5, "X": 0, "ri_missing": []})
        self.assertFalse(all_r["passed"])
        self.assertFalse(all_r["checks"]["at_least_half_M"])
        self.assertTrue(all_r["checks"]["R_within_bound"])  # floor(20*8/32)=5, so R=5 is within the bound,
        # but an all-R result must still fail: 0 M pairs fails the at-least-half-M rule.
        self.assertFalse(reproduce.acceptance({"cases": 8, "M": 2, "R": 6, "X": 0, "ri_missing": []})["checks"]["R_within_bound"])

    def test_scaled_bound(self):
        self.assertEqual(reproduce.acceptance({"cases": 8, "M": 8, "R": 0, "X": 0, "ri_missing": []})["R_bound"], 5)


class ClassifyLogicTest(unittest.TestCase):
    """Drive the copied np_attribution_v2.classify on synthetic sealed run dirs."""
    def make_run(self, root, name, rows, success=False):
        case = "T--0001"
        d = Path(root) / name / "results" / case
        d.mkdir(parents=True)
        (d / "trace.jsonl").write_text("\n".join(json.dumps(dict(r, event="action_dispatch")) for r in rows) + "\n")
        (d / "result.json").write_text(json.dumps({"success": success}))
        (d / "completed.json").write_text("{}")
        return str(Path(root) / name), case

    def row(self, video="v", state="s", action="a", seed=1):
        return {"input": {"video.cam": video, "state.eef": state}, "action": action, "noise_seed": seed,
                "execute_steps": 16, "selected_k": 16}

    def test_m_r_x(self):
        from harnessstudio.eval.attribution import _npa
        npa = _npa()
        with tempfile.TemporaryDirectory() as tmp:
            parent, case = self.make_run(tmp, "p", [self.row(), self.row(action="b")])
            same, _ = self.make_run(tmp, "m", [self.row(), self.row(action="b")])
            render, _ = self.make_run(tmp, "r", [self.row(), self.row(video="w", action="c")])
            seed, _ = self.make_run(tmp, "x", [self.row(), self.row(action="b", seed=2)])
            clean = {case: False}
            self.assertEqual(npa.classify(case, parent, same, clean, clean, None)["class"], "M")
            self.assertEqual(npa.classify(case, parent, render, clean, clean, None)["class"], "R")
            self.assertEqual(npa.classify(case, parent, seed, clean, clean, None)["class"], "X")
            self.assertEqual(npa.classify(case, parent, render, clean, {case: True}, None)["class"], "X")
