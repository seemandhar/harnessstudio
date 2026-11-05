"""EXT openvla-vision-token-hook (docs/extensions/openvla-vision-token-hook.md): capability vision_token_hooks,
policy_ext/vision_token_hooks.py (protocol, vision_forward_hook, generate_with_hook) and the server subclass
policies/openvla_ttf_hs.py. CPU only; server tests need torch (skipped otherwise).

V1 static: capability registered; profiles; parents byte-identical (sha256 pins of openvla_hs.py, mixin, protocol).
V2 protocol: check_hook refusals; vision_forward_hook restores the stock forward (also on exceptions), counts calls,
   refuses wrong shapes/dtypes and non-fused backbones.
V3 generate_with_hook on a fake OpenVLA-like model: 29871 appended to input_ids only (the processor mask is passed
   unchanged), output_attentions passed, prefill attention of the declared layers returned, exactly one vision call.
V4 server dispatch: no sampler and decoding-hook samplers go to the parent unchanged; vision-token hooks run the
   hooked path; refusals (readouts, >1 seed, no activation_log, prefix); per-episode state kept and dropped at
   EPISODE_START; one activation record per hooked request.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import threading
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "toys"))

from harnessstudio.core import capabilities  # noqa: E402
from harnessstudio.profiles import load_profile  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SERVER_DEPS = HAS_TORCH and importlib.util.find_spec("transformers") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "needs torch")
needs_server = unittest.skipUnless(HAS_SERVER_DEPS, "needs torch + transformers (an OpenVLA server env)")

PARENT_PINS = {  # main 32c86a6; also pinned by tests/test_ext_openvla_residual_hook.py (R6)
    "harnessstudio/policies/openvla_hs.py": "e7c6a4c302fece60b338b9a6c14c7ac0cea4dea2057ff72b74c0abfa0aab174b",
    "harnessstudio/policy_ext/mixin.py": "b72ba1d14c4ee713389da2d115d5b927db8b42f4d4a40239ac7ab4e4a3924bc0",
    "harnessstudio/policy_ext/protocol.py": "4955649344effa9b4365c62578940f4d5569c9e7f2cabb11a45e3108111ac411",
}


def _sha(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestStatic(unittest.TestCase):
    def test_v1_capability_and_profiles(self):
        self.assertIn("vision_token_hooks", capabilities.CAPABILITIES)
        self.assertIn("vision_token_hooks", capabilities.check(load_profile("openvla-libero-ttf").capabilities))
        self.assertIn("vision_token_hooks", load_profile("fake-t3").capabilities)
        hs, ttf = load_profile("openvla-libero-hs"), load_profile("openvla-libero-ttf")
        self.assertEqual(set(ttf.capabilities) - set(hs.capabilities), {"vision_token_hooks"})
        self.assertEqual((ttf.chunking, ttf.execute_horizon), (hs.chunking, hs.execute_horizon))
        for name in ("fake-t0", "fake-t1", "fake-t2", "openvla-libero-hs", "openvla-simpler-star"):
            self.assertNotIn("vision_token_hooks", load_profile(name).capabilities, name)

    def test_v1_parents_byte_identical(self):
        self.assertEqual({p: _sha(p) for p in PARENT_PINS if _sha(p) != PARENT_PINS[p]}, {})

    def test_v1_sampler_registration_accepts_vision_token_hooks_only_when_declared(self):
        from harnessstudio.core.errors import RegistrationError
        from harnessstudio.core.method import MethodConfig, ModuleRef, validate
        base = dict(name="toy-vision", version="1", paper="none", fidelity="faithful",
                    slots={"proposer": ModuleRef("hs_ext_vision_toy:ToyVisionHook", {}),
                           "sampler": ModuleRef("hs_ext_vision_toy:ToyVisionHook", {})})
        req = {"act", "chunk", "seeded_sampling", "isolated_streams"}
        validate(MethodConfig(**base, required=frozenset(req | {"vision_token_hooks"})))
        with self.assertRaisesRegex(RegistrationError, "vision_token_hooks"):
            validate(MethodConfig(**base, required=frozenset(req)))

    def test_v1_tier1_profile_for_vision_methods_is_fake_t3(self):
        from harnessstudio.verify import tier1
        cfg = types.SimpleNamespace(required=frozenset({"act", "chunk", "seeded_sampling", "isolated_streams",
                                                        "vision_token_hooks"}))
        self.assertEqual(tier1.default_profile(cfg), "fake-t3")


@needs_torch
class TestProtocol(unittest.TestCase):
    def setUp(self):
        import torch
        from harnessstudio.policy_ext import vision_token_hooks as V
        self.torch, self.V = torch, V

    def _backbone(self, fused=True):
        torch = self.torch

        class Feat(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.embed_dim = dim
                self.patch_embed = types.SimpleNamespace(num_patches=4)

            def forward(self, x):
                return x.mean(dim=(2, 3), keepdim=False)[:, None, :1].expand(1, 4, self.embed_dim).clone()

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.use_fused_vision_backbone = fused
                self.featurizer, self.fused_featurizer = Feat(3), Feat(5)

            def forward(self, pixel_values):
                img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
                return torch.cat([self.featurizer(img), self.fused_featurizer(img_fused)], dim=2)

        return Backbone()

    def test_v2_check_hook(self):
        import hs_ext_vision_toy as toy
        self.V.check_hook(toy.ToyVisionHook())
        with self.assertRaises(self.V.VisionTokenHookError):
            self.V.check_hook(toy.NotAHook())
        bad = toy.ToyVisionHook()
        bad.OUTPUT_ATTENTIONS = False
        with self.assertRaisesRegex(self.V.VisionTokenHookError, "OUTPUT_ATTENTIONS"):
            self.V.check_hook(bad)

    def test_v2_forward_swapped_and_restored(self):
        torch, V = self.torch, self.V
        bb = self._backbone()
        pv = torch.randn(1, 6, 8, 8)
        stock = bb(pv)
        with V.vision_forward_hook(bb, lambda p, d, s: torch.cat([d, torch.zeros_like(s)], dim=2)) as rec:
            hooked = bb(pv)
        self.assertEqual(rec["calls"], 1)
        self.assertTrue(torch.equal(hooked[..., :3], stock[..., :3]) and float(hooked[..., 3:].abs().sum()) == 0)
        self.assertNotIn("forward", bb.__dict__)
        self.assertTrue(torch.equal(bb(pv), stock))
        with self.assertRaises(RuntimeError):
            with V.vision_forward_hook(bb, lambda p, d, s: torch.cat([d, s], dim=2)):
                raise RuntimeError("boom")
        self.assertNotIn("forward", bb.__dict__)
        with V.vision_forward_hook(bb, lambda p, d, s: d):
            with self.assertRaisesRegex(V.VisionTokenHookError, "expected"):
                bb(pv)
        with V.vision_forward_hook(bb, lambda p, d, s: torch.cat([d, s], dim=2).double()):
            with self.assertRaisesRegex(V.VisionTokenHookError, "expected"):
                bb(pv)
        with self.assertRaisesRegex(V.VisionTokenHookError, "fused"):
            with V.vision_forward_hook(self._backbone(fused=False), lambda p, d, s: d):
                pass

    def _fake_model(self):
        torch, test = self.torch, self
        bb = self._backbone()

        class Out:
            pass

        class Model:
            def __init__(self):
                self.vision_backbone = bb
                self.calls = []

            def generate(self, input_ids, max_new_tokens, output_attentions, return_dict_in_generate, do_sample,
                         attention_mask=None, pixel_values=None):
                self.calls.append({"ids_len": input_ids.shape[1], "mask_len": attention_mask.shape[1],
                                   "output_attentions": output_attentions, "do_sample": do_sample})
                patches = self.vision_backbone(pixel_values)
                seq = input_ids.shape[1] + patches.shape[1]
                base = int((patches.float().sum() * 1000).abs()) % 100
                new = torch.tensor([[base + i for i in range(max_new_tokens)]])
                o = Out()
                o.sequences = torch.cat([input_ids, new], dim=1)
                o.attentions = None
                if output_attentions:
                    o.attentions = tuple(
                        tuple(torch.full((1, 2, 1 if i else seq, seq + i), float(layer)) for layer in range(3))
                        for i in range(max_new_tokens))
                return o

        return Model()

    def test_v3_generate_with_hook(self):
        import hs_ext_vision_toy as toy
        torch, V = self.torch, self.V
        model = self._fake_model()
        hook = toy.ToyVisionHook(layer=2)
        state = hook.init_state()
        inputs = {"input_ids": torch.tensor([[1, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                  "pixel_values": torch.randn(1, 6, 8, 8)}
        tokens, attn, info, shape = V.generate_with_hook(model, hook, state, inputs, 7)
        self.assertEqual(len(tokens), 7)
        self.assertEqual(model.calls[0], {"ids_len": 4, "mask_len": 3, "output_attentions": True, "do_sample": False})
        self.assertEqual(sorted(attn), [2])
        self.assertEqual(float(attn[2].mean()), 2.0)
        self.assertEqual((shape["vision_calls"], shape["new_tokens"], shape["prefill_len"]), (1, 7, 8))
        self.assertEqual(info, {"requests": 1})
        # 29871 already present: not appended twice
        inputs["input_ids"] = torch.tensor([[1, 5, V.EMPTY_TOKEN]])
        V.generate_with_hook(model, hook, state, inputs, 7)
        self.assertEqual(model.calls[1]["ids_len"], 3)
        # a missing layer is an infrastructure error
        with self.assertRaisesRegex(V.VisionTokenHookError, "layer 9 missing"):
            V.generate_with_hook(model, toy.ToyVisionHook(layer=9), hook.init_state(), inputs, 7)


@needs_server
class TestServer(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from harnessstudio.policies import openvla_hs as H
        from harnessstudio.policies import openvla_ttf_hs as T
        self.np, self.torch, self.H, self.T = np, torch, H, T
        proto = TestProtocol()
        proto.setUp()
        model = proto._fake_model()
        model.vocab_size = 32000
        model.bin_centers = np.linspace(-1, 1, 255)
        model.config = types.SimpleNamespace(n_action_bins=256)
        model.language_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[0, 1, 2]))
        model.get_action_dim = lambda key=None: 7
        model.get_action_stats = lambda key=None: {"q01": [-1.0] * 7, "q99": [1.0] * 7, "mask": [True] * 6 + [False]}
        srv = object.__new__(T.HarnessOpenVLATTFServer)
        srv._model, srv.unnorm_key = model, "libero_10"
        srv._infer_lock, srv._states, srv._decoders = threading.RLock(), {}, {}
        srv._sampler_prefixes, srv._vhooks = ("hs_",), {}
        srv._log_lock, srv._log_seq = threading.Lock(), 0
        self.log = os.path.join(self._tmp(), "activation.jsonl")
        srv._activation_log = self.log
        srv._inputs = lambda obs: {"input_ids": torch.tensor([[1, 5, 6]]),
                                   "attention_mask": torch.ones(1, 3, dtype=torch.long),
                                   "pixel_values": torch.as_tensor(obs["px"])}
        self.srv, self.model = srv, model

    def _tmp(self):
        import tempfile
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d

    def _ctx(self, ep="e1"):
        return types.SimpleNamespace(session_id="s1", episode_id=ep)

    def _obs(self):
        return {"px": self.torch.randn(1, 6, 8, 8).numpy()}

    def test_v4_capability(self):
        self.assertEqual(self.T.HarnessOpenVLATTFServer.harness_capabilities
                         - self.H.HarnessOpenVLAServer.harness_capabilities, {"vision_token_hooks"})

    def test_v4_parent_paths_unchanged(self):
        seen = []
        orig = self.H.HarnessOpenVLAServer.sample_chunks
        self.H.HarnessOpenVLAServer.sample_chunks = lambda s, o, seeds, sampler, readouts, ctx: seen.append(sampler) or []
        try:
            self.srv.sample_chunks(self._obs(), [3], None, [], self._ctx())
            self.srv.sample_chunks(self._obs(), [3], {"factory": "hs_ext_vision_toy:NotAHook", "params": {}}, [],
                                   self._ctx())
        finally:
            self.H.HarnessOpenVLAServer.sample_chunks = orig
        self.assertEqual(len(seen), 2)
        self.assertFalse(os.path.exists(self.log))
        self.assertEqual(self.model.calls, [])

    def test_v4_hooked_path_state_and_log(self):
        sampler = {"factory": "hs_ext_vision_toy:ToyVisionHook", "params": {"layer": 1}}
        out1 = self.srv.sample_chunks(self._obs(), [3], sampler, [], self._ctx())
        out2 = self.srv.sample_chunks(self._obs(), [4], sampler, [], self._ctx())
        self.assertEqual(out1[0][0].shape, (1, 7))
        recs = [json.loads(line) for line in open(self.log)]
        self.assertEqual([r["seq"] for r in recs], [1, 2])
        self.assertEqual([r["info"]["requests"] for r in recs], [1, 2])          # state kept within the episode
        self.assertEqual(recs[0]["shape"]["attention_layers"], [1])
        self.assertEqual(recs[1]["seeds"], [4])
        hook = next(iter(self.srv._vhooks.values()))
        self.assertEqual(hook.bound, (4, (3, 5), 3))
        self.srv._drop_states(self._ctx())     # what the parent's on_episode_start/on_episode_end call first
        self.srv.sample_chunks(self._obs(), [5], sampler, [], self._ctx())
        self.assertEqual(json.loads(open(self.log).read().splitlines()[-1])["info"]["requests"], 1)
        self.srv.sample_chunks(self._obs(), [5], sampler, [], self._ctx(ep="e2"))
        self.assertEqual(json.loads(open(self.log).read().splitlines()[-1])["info"]["requests"], 1)
        self.assertNotIn("forward", self.model.vision_backbone.__dict__)

    def test_v4_refusals(self):
        sampler = {"factory": "hs_ext_vision_toy:ToyVisionHook", "params": {}}
        with self.assertRaisesRegex(ValueError, "readouts"):
            self.srv.sample_chunks(self._obs(), [1], sampler, ["logits"], self._ctx())
        with self.assertRaisesRegex(ValueError, "one seed"):
            self.srv.sample_chunks(self._obs(), [1, 2], sampler, [], self._ctx())
        with self.assertRaisesRegex(ValueError, "not allowed"):
            self.srv.sample_chunks(self._obs(), [1], {"factory": "os:system", "params": {}}, [], self._ctx())
        self.srv._activation_log = None
        with self.assertRaisesRegex(ValueError, "activation_log"):
            self.srv.sample_chunks(self._obs(), [1], sampler, [], self._ctx())


if __name__ == "__main__":
    unittest.main()
