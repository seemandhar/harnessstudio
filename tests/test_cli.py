import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from harnessstudio import cli


class CliTest(unittest.TestCase):
    def test_list_json(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["list", "--json"])
        self.assertEqual(code, 0)
        rows = {r["name"]: r for r in json.loads(out.getvalue())}
        self.assertTrue(set(rows) >= {"identity", "medoid-bon", "chunk-interrupt", "chunktrust-ahs"})
        self.assertTrue(rows["medoid-bon"]["compatible"]["gr00t-n1.5"])
        self.assertEqual(rows["chunk-interrupt"]["fidelity"], "tf_variant")

    def test_verify_tier1_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["verify", "identity", "--policy", "gr00t-n1.5", "--tier", "1",
                                 "--report-root", tmp])
            self.assertEqual(code, 0)
            reports = list(Path(tmp).rglob("report.json"))
            self.assertEqual(len(reports), 1)
            self.assertTrue(json.loads(reports[0].read_text())["passed"])
