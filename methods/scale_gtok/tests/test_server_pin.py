"""Pins (review RC2): the token-position counter relies on the hook order of the UNMERGED OpenVLA server file
harnessstudio/policies/openvla_hs.py sha256 e7c6a4c3... (the server that produced the eval/devbase map records), and
the subclass relies on hs_scale.decoder internals (self_uncertainty, select_token). Any change to either file fails
here and requires re-verification of scale-gtok (a new candidate if it was frozen). No torch needed."""
from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

HS_ROOT = Path(__file__).resolve().parents[3]
SERVER = HS_ROOT / "harnessstudio" / "policies" / "openvla_hs.py"
SCALE_DECODER = HS_ROOT / "methods" / "scale" / "hs_scale" / "decoder.py"
SERVER_SHA256 = "e7c6a4c302fece60b338b9a6c14c7ac0cea4dea2057ff72b74c0abfa0aab174b"
SCALE_DECODER_SHA256 = "619d9b960bd84878a34825c1e35e7f182cade2515cc51721a01fa5fe14256d60"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def _calls(node: ast.AST, attr: str) -> list:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


class ServerPin(unittest.TestCase):
    def test_server_file_hash(self):
        self.assertEqual(_sha(SERVER), SERVER_SHA256,
                         "openvla_hs.py changed: re-verify the scale-gtok token-position counter against it")

    def test_scale_decoder_hash(self):
        self.assertEqual(_sha(SCALE_DECODER), SCALE_DECODER_SHA256,
                         "hs_scale/decoder.py changed: re-verify ScaleGripperArgmax (it subclasses ScaleDecoder)")

    def test_decode_hook_order(self):
        """_decode: one fresh generator per call, attention_temperature exactly once BEFORE the token loop, and one
        select_token per iteration of `for _ in range(action_dim)`."""
        tree = ast.parse(SERVER.read_text())
        dec = _method(tree, "HarnessOpenVLAServer", "_decode")
        temps = _calls(dec, "attention_temperature")
        selects = _calls(dec, "select_token")
        seeds = _calls(dec, "manual_seed")
        self.assertEqual(len(temps), 1)
        self.assertEqual(len(selects), 1)
        self.assertEqual(len(seeds), 1)
        loops = [n for n in ast.walk(dec) if isinstance(n, ast.For)]
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        self.assertIsInstance(loop.iter, ast.Call)
        self.assertEqual(getattr(loop.iter.func, "id", None), "range")
        self.assertEqual(getattr(loop.iter.args[0], "id", None), "action_dim")
        self.assertIn(selects[0], list(ast.walk(loop)))
        self.assertNotIn(temps[0], list(ast.walk(loop)))
        self.assertLess(temps[0].lineno, loop.lineno)
        self.assertLess(seeds[0].lineno, temps[0].lineno)
        self.assertEqual(temps[0].lineno, 186)      # line cited by the review (RC2)
        self.assertEqual(seeds[0].lineno, 185)      # line cited by the review (RC3)

    def test_sample_chunks_calls_decode_per_seed(self):
        """sample_chunks: one _decode per seed (each resets the counter); decoder state advances once per request
        with the first seed's infos (K = 1: the executed sample)."""
        tree = ast.parse(SERVER.read_text())
        sc = _method(tree, "HarnessOpenVLAServer", "sample_chunks")
        decodes = _calls(sc, "_decode")
        self.assertEqual(len(decodes), 1)
        loops = [n for n in ast.walk(sc) if isinstance(n, ast.For) and decodes[0] in list(ast.walk(n))]
        self.assertEqual(len(loops), 1)
        self.assertEqual(getattr(loops[0].iter, "id", None), "seeds")
        self.assertEqual(len(_calls(sc, "next_state")), 1)


if __name__ == "__main__":
    unittest.main()
