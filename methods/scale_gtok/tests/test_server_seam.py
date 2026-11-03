"""The real handler seam: HarnessOpenVLAServer.sample_chunks of the PINNED openvla_hs.py (sha256 e7c6a4c3...), loaded
from its file, driving `scale` and `scale-gtok` samplers on a fake OpenVLA model (CPU, no weights). Checks that, on
the server's own token loop, vision-temperature patch, de-tokenization, gripper mapping and per-episode state:
tokens 1-6, u, tau, gamma and the decoder state are SCALE's; token 7 is the argmax of the logits the server fed to
select_token; the executed action differs from SCALE's only in dim 6.

Needs torch and vla_eval (the OpenVLA server env); run it without installing anything (see test_decoder.py):

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH=harnessstudio:harnessstudio/methods/scale:harnessstudio/methods/scale_gtok \
    /home/sejain/.cache/uv/environments-v2/openvla-hs-73218eaab6c13068/bin/python -m unittest discover \
      -s harnessstudio/methods/scale_gtok/tests -v
"""
from __future__ import annotations

import hashlib
import importlib.util
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch
    import vla_eval  # noqa: F401
except ImportError:
    torch = None

HS_ROOT = Path(__file__).resolve().parents[3]
SERVER = HS_ROOT / "harnessstudio" / "policies" / "openvla_hs.py"
SERVER_SHA256 = "e7c6a4c302fece60b338b9a6c14c7ac0cea4dea2057ff72b74c0abfa0aab174b"
VOCAB, NBINS, DIM = 32000, 256, 7
BEGIN = VOCAB - NBINS


class FakeAttn:
    def __init__(self):
        self.head_dim = 64
        self.num_heads = 1
        self.scale = self.head_dim ** -0.5


class FakeModel:
    """Logits depend on the prompt, the vision-attention scale (gamma) and the tokens decoded so far (the cache)."""

    def __init__(self):
        self.vocab_size = VOCAB
        self.config = SimpleNamespace(n_action_bins=NBINS)
        bins = np.linspace(-1, 1, NBINS)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self.attn = FakeAttn()
        self.vision_backbone = SimpleNamespace(featurizer=SimpleNamespace(blocks=[SimpleNamespace(attn=self.attn)]),
                                               use_fused_vision_backbone=False)
        self.calls = []        # (history, logits) per forward

    def get_action_dim(self, key):
        return DIM

    def get_action_stats(self, key):
        return {"q01": np.full(DIM, -0.5), "q99": np.full(DIM, 0.5), "mask": np.array([True] * (DIM - 1) + [False])}

    def __call__(self, input_ids, attention_mask=None, pixel_values=None, past_key_values=None, use_cache=True,
                 return_dict=True):
        if past_key_values is None:
            hist = (tuple(int(x) for x in input_ids[0].tolist()), float(self.attn.scale))
        else:
            hist = past_key_values + (int(input_ids[0, -1]),)
        pos = len(hist) - 2
        key = repr(hist).encode()
        g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:6], "big"))
        logits = torch.randn(VOCAB, generator=g)
        spread = 0.5 if pos % 2 else 3.0
        logits[BEGIN:] = torch.randn(NBINS, generator=g) * spread + 4.0
        if pos == DIM - 1:   # two-mode gripper token
            logits[BEGIN:] = -4.0 + torch.randn(NBINS, generator=g) * 0.1
            logits[BEGIN + 3] = 6.0
            logits[VOCAB - 2] = 5.6
        self.calls.append((hist, logits.clone()))
        return SimpleNamespace(logits=logits[None, None, :], past_key_values=hist)


def load_server():
    assert hashlib.sha256(SERVER.read_bytes()).hexdigest() == SERVER_SHA256
    spec = importlib.util.spec_from_file_location("openvla_hs_pinned_e7c6a4c3", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    srv = mod.HarnessOpenVLAServer.__new__(mod.HarnessOpenVLAServer)   # no weights: skip __init__
    srv._infer_lock = threading.RLock()
    srv._sampler_prefixes = ("hs_",)
    srv._decoders, srv._states = {}, {}
    srv._model = FakeModel()
    srv.unnorm_key = "libero_10"
    srv._inputs = lambda obs: {"input_ids": torch.tensor([[1, 529, 29901, int(obs["tag"])]]),
                               "attention_mask": None, "pixel_values": torch.zeros(1)}
    return srv


def run_episode(factory, seeds, episode="ep0"):
    from hs_scale.decoder import DEFAULTS
    srv = load_server()
    sampler = {"factory": factory, "params": dict(DEFAULTS)}
    ctx = SimpleNamespace(session_id="s", episode_id=episode)
    out = []
    for d, seed in enumerate(seeds):
        n0 = len(srv._model.calls)
        [(action, extras)] = srv.sample_chunks({"tag": 100 + d}, [seed], sampler, (), ctx)
        calls = srv._model.calls[n0:]
        out.append({"action": action, "extras": extras, "logits7": calls[DIM - 1][1], "ncalls": len(calls),
                    "state": [list(x) for x in next(iter(srv._states.values()))]})
    return out


@unittest.skipIf(torch is None, "needs torch + vla_eval (the OpenVLA server env; see module docstring)")
class ServerSeam(unittest.TestCase):
    SEEDS = [1071524933, 2477133818, 591234, 77, 31337, 900001, 2 ** 31 - 1, 12, 4242, 65535, 3, 1717171717]

    @classmethod
    def setUpClass(cls):
        cls.scale = run_episode("hs_scale.decoder:ScaleDecoder", cls.SEEDS)
        cls.gtok = run_episode("hs_scale_gtok.decoder:ScaleGripperArgmax", cls.SEEDS)

    def test_one_forward_per_token(self):
        self.assertTrue(all(x["ncalls"] == DIM for x in self.gtok))

    def test_scale_identical_except_gripper_token(self):
        for d, (a, b) in enumerate(zip(self.scale, self.gtok)):
            ea, eb = a["extras"], b["extras"]
            self.assertEqual(ea["tokens"][:DIM - 1], eb["tokens"][:DIM - 1], d)
            self.assertEqual(ea["u"], eb["u"], d)
            self.assertEqual(ea["tau"], eb["tau"], d)
            self.assertEqual(ea["gamma"], eb["gamma"], d)
            self.assertEqual(a["state"], b["state"], d)
            self.assertEqual(a["action"][0, :DIM - 1].tobytes(), b["action"][0, :DIM - 1].tobytes(), d)
            self.assertEqual(a["action"].dtype, np.float32)
        self.assertTrue(any(x["extras"]["gamma"] != 1.0 for x in self.gtok), "gamma never left 1.0")

    def test_gripper_token_is_argmax_of_server_logits(self):
        for d, b in enumerate(self.gtok):
            argmax = BEGIN + int(torch.argmax(b["logits7"][BEGIN:]).item())
            self.assertEqual(b["extras"]["tokens"][DIM - 1], argmax, d)
            self.assertIn(float(b["action"][0, DIM - 1]), (-1.0, 1.0))

    def test_non_vacuous_gripper_difference(self):
        diff = [d for d, (a, b) in enumerate(zip(self.scale, self.gtok))
                if a["extras"]["tokens"][DIM - 1] != b["extras"]["tokens"][DIM - 1]]
        self.assertTrue(diff, "SCALE's gripper sample never differed from argmax in this fixture")
        flips = [d for d in diff if self.scale[d]["action"][0, DIM - 1] != self.gtok[d]["action"][0, DIM - 1]]
        self.assertTrue(flips, "no executed gripper sign flip in this fixture")


if __name__ == "__main__":
    unittest.main()
