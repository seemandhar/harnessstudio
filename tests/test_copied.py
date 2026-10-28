import hashlib
import json
import unittest
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1] / "harnessstudio" / "backends" / "gr00t_robocasa365"


class CopiedTest(unittest.TestCase):
    def test_copies_match_sources_and_record(self):
        table = json.loads((BACKEND / "COPIED.json").read_text())
        self.assertEqual(len(table), 9)
        for name, row in table.items():
            copy = hashlib.sha256((BACKEND / "copied" / name).read_bytes()).hexdigest()
            self.assertEqual(copy, row["sha256"], name)  # byte-identical to the recorded original
        self.assertTrue(table["ri_scan.py"]["sha256"].startswith("b550037c"))
