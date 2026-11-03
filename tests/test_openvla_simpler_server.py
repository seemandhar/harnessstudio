"""EXT simpler-seeded, server half: HarnessOpenVLASimplerServer (mode simpler_pcd_authors), CPU only.

Needs torch + transformers + opencv (the run-local server envs); skipped elsewhere. Run in BOTH envs:

    env -u TRANSFORMERS_CACHE -u HF_HUB_CACHE CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=<scratch> HF_MODULES_CACHE=<scratch>/modules PYTHONPATH=<worktree> \
      <env python> -B -m unittest tests.test_openvla_simpler_server -v

The server is built with __new__ (never __init__, which loads 15 GB of weights). The stub VLA wraps a tiny REAL
LlamaForCausalLM with OpenVLA's vocabulary sizes (32064 logits, vocab_size 32000, 256 bins) and runs the REAL
transformers ``generate``. It mimics Prismatic's multimodal mask construction (modeling_prismatic.py:386-390): when an
attention_mask is passed, image tokens are spliced into it, so a mask one token shorter than the ids (predict_action
appends 29871 to the ids only, :512-518) reaches the Llama.

Covers S1 (parents byte-identical), S2 (AST: declared overrides only, no predict_action, no attention_mask to
generate, eager load, cuDNN flags), S-1 (the no-mask decode runs under eager attention and equals an independent
greedy reference; the parent's predict_action path crashes under eager as the negative control), S3 (native, seeded
and logits paths agree in float64; the bin table matches _actions on the boundary tokens; refusals), S-4 (pre-cast
float64 log), S-5 (action spec, observation params, eager-class assertion), A-7 (short decode), A-9 (task config
ignored).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import tempfile
import textwrap
import threading
import types
import unittest

import numpy as np

try:
    import cv2
    import torch
    from transformers import BatchFeature, LlamaConfig, LlamaForCausalLM
except ImportError:
    raise unittest.SkipTest("needs torch + transformers + opencv (run with a SIMPLER OpenVLA server env)")

from vla_eval.model_servers.predict import PredictModelServer  # noqa: E402
from harnessstudio.policies import openvla_hs as H  # noqa: E402
from harnessstudio.policies import openvla_logits_hs as L  # noqa: E402
from harnessstudio.policies import openvla_simpler_hs as S  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENVLA_HS_SHA256 = "e7c6a4c302fece60b338b9a6c14c7ac0cea4dea2057ff72b74c0abfa0aab174b"
OPENVLA_LOGITS_HS_SHA256 = "ddf04264cb476f2b9a33f3a2fb4d986d83f555b9275064794ea70a844e456f8d"
V_LOGITS, VOCAB, NB, AD, EMPTY = 32064, 32000, 256, 7, 29871
# openvla/openvla-7b@31f090d config.json norm_stats["fractal20220817_data"]["action"]
FRACTAL = {"mask": [True, True, True, True, True, True, False],
           "q01": [-0.22453527510166169, -0.14820013284683228, -0.231589707583189, -0.3517994859814644,
                   -0.4193011274933815, -0.43643461108207704, 0.0],
           "q99": [0.17824687153100965, 0.14938379630446405, 0.21842354819178575, 0.5892666035890578,
                   0.35272657424211445, 0.44796681255102094, 1.0]}
N_IMG = 4   # stand-in for Prismatic's 256 patch embeddings


class StubProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, pil_image):
        self.calls.append((prompt, np.asarray(pil_image).copy()))
        digest = hashlib.sha256(prompt.encode()).digest()
        ids = [1] + [100 + b for b in digest[:9]]            # prompt ids never end with 29871
        return BatchFeature({"input_ids": torch.tensor([ids], dtype=torch.long),
                             "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                             "pixel_values": torch.tensor(np.asarray(pil_image, dtype=np.float32))[None]})


class StubVLA:
    """Prismatic's generate/predict_action shape around a tiny real Llama (eager unless told otherwise)."""

    def __init__(self, attn="eager", seed=0):
        torch.manual_seed(seed)
        cfg = LlamaConfig(vocab_size=V_LOGITS, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                          bos_token_id=1, eos_token_id=2, pad_token_id=32000)
        cfg._attn_implementation = attn
        self.lm = LlamaForCausalLM(cfg).eval()
        with torch.no_grad():                                  # favour the action-token range, like OpenVLA
            self.lm.lm_head.weight[VOCAB - NB: VOCAB] *= 6.0
        self.lm.generation_config.eos_token_id = None
        self.lm.generation_config.pad_token_id = 32000
        self.language_model = self.lm
        self.vocab_size = VOCAB
        self.config = types.SimpleNamespace(n_action_bins=NB, _attn_implementation=attn)
        bins = np.linspace(-1.0, 1.0, NB)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self.generate_kwargs = []

    def get_action_dim(self, key):
        return len(self.get_action_stats(key)["q01"])

    def get_action_stats(self, key):
        assert key == "fractal20220817_data"
        return FRACTAL

    def _image_ids(self, pixel_values):
        d = hashlib.sha256(pixel_values.to(torch.float32).numpy().tobytes()).digest()
        return torch.tensor([[200 + b for b in d[:N_IMG]]], dtype=torch.long)

    def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **kw):
        self.generate_kwargs.append(sorted(list(kw) + (["attention_mask"] if attention_mask is not None else [])))
        full = torch.cat([input_ids[:, :1], self._image_ids(pixel_values), input_ids[:, 1:]], dim=1)
        if attention_mask is not None:
            # modeling_prismatic.py:386-390: the multimodal mask is built from the processor's (unextended) mask, so it
            # is one shorter than the sequence once 29871 was appended to the ids. Prismatic's prefill forward passes
            # it to the Llama with positions from the full sequence: reproduce exactly that forward.
            mm_mask = torch.cat([attention_mask[:, :1], torch.ones(1, N_IMG, dtype=attention_mask.dtype),
                                 attention_mask[:, 1:]], dim=1)
            with torch.no_grad():
                self.lm(input_ids=full, attention_mask=mm_mask, position_ids=torch.arange(full.shape[1])[None])
        out = self.lm.generate(input_ids=full, **kw)
        return torch.cat([input_ids, out[:, full.shape[1]:]], dim=1)

    def predict_action(self, input_ids=None, unnorm_key=None, **kwargs):
        """modeling_prismatic.py:506-536: appends 29871 to input_ids only, keeps the processor's attention_mask."""
        if not torch.all(input_ids[:, -1] == EMPTY):
            input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([EMPTY]).long(), dim=0)), dim=1)
        ad = self.get_action_dim(unnorm_key)
        generated = self.generate(input_ids, max_new_tokens=ad, **kwargs)
        return generated[0, -ad:].numpy()

    def reference_tokens(self, prompt_ids, pixel_values, n=AD):
        """Independent greedy reference: argmax over the forward's last-position logits, no mask, no generate."""
        ids = torch.cat([prompt_ids, torch.tensor([[EMPTY]])], dim=1)
        full = torch.cat([ids[:, :1], self._image_ids(pixel_values), ids[:, 1:]], dim=1)
        out = []
        with torch.no_grad():
            for _ in range(n):
                logits = self.lm(input_ids=full).logits[0, -1].float()
                t = int(torch.argmax(logits))
                out.append(t)
                full = torch.cat([full, torch.tensor([[t]])], dim=1)
        return np.array(out)


def make_server(attn="eager", log=None):
    srv = S.HarnessOpenVLASimplerServer.__new__(S.HarnessOpenVLASimplerServer)
    PredictModelServer.__init__(srv, chunk_size=1, action_ensemble="newest")   # no model: buffers/state only
    srv.model_path, srv.unnorm_key = "/stub", "fractal20220817_data"
    srv.jpeg_roundtrip = srv.center_crop = False
    srv._device = torch.device("cpu")
    srv._processor, srv._model = StubProcessor(), StubVLA(attn)
    srv._infer_lock = threading.RLock()
    srv._sampler_prefixes, srv._decoders, srv._states = ("hs_",), {}, {}
    srv._init_log(log)
    return srv


def obs(seed=0, text="close top drawer"):
    rng = np.random.default_rng(seed)
    return {"images": {"primary": rng.integers(0, 255, (512, 640, 3), dtype=np.uint8)}, "task_description": text}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class TestParentsAndAst(unittest.TestCase):
    def test_s1_parents_byte_identical(self):
        self.assertEqual(sha(inspect.getsourcefile(H)), OPENVLA_HS_SHA256)
        self.assertEqual(sha(inspect.getsourcefile(L)), OPENVLA_LOGITS_HS_SHA256)

    def test_s2_declared_overrides_only(self):
        cls = S.HarnessOpenVLASimplerServer
        own = {k for k, v in vars(cls).items() if callable(v) or isinstance(v, (staticmethod, classmethod))}
        self.assertEqual(own, {"__init__", "_check_model", "_init_log", "_log_action", "get_observation_params",
                               "get_action_spec", "get_action_stats", "_preprocess_image", "_prompt", "_inputs",
                               "_generate", "_actions", "predict", "sample_chunks", "_predict_recorded"})
        self.assertEqual(cls.harness_capabilities, L.HarnessOpenVLALogitsServer.harness_capabilities - {"decoding_hooks"})

    def test_s2_ast_no_predict_action_no_mask_eager_cudnn(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(S)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse([c for c in calls if c.func.attr == "predict_action"], "predict_action carries the short mask")
        gens = [c for c in calls if c.func.attr == "generate"]
        self.assertEqual(len(gens), 1)
        kws = {k.arg for k in gens[0].keywords}
        self.assertEqual(kws, {"input_ids", "pixel_values", "max_new_tokens", "do_sample", None})  # None = **extra
        self.assertNotIn("attention_mask", ast.unparse(gens[0]))
        fp = [c for c in calls if c.func.attr == "from_pretrained" and any(k.arg == "attn_implementation"
                                                                           for k in c.keywords)]
        self.assertEqual(len(fp), 1)
        self.assertEqual([ast.literal_eval(k.value) for k in fp[0].keywords if k.arg == "attn_implementation"], ["eager"])
        src = inspect.getsource(S)
        self.assertIn("torch.backends.cudnn.deterministic = True", src)
        self.assertIn("torch.backends.cudnn.benchmark = False", src)
        extra_keys = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                      and n.value.id == "extra"]
        self.assertEqual(sorted(ast.literal_eval(n.slice) for n in extra_keys), ["logits_processor", "stopping_criteria"])


class TestEagerNoMaskDecode(unittest.TestCase):
    """[S-1] on a tiny eager Llama through the real transformers generate."""

    def test_native_decode_runs_under_eager_and_matches_reference(self):
        srv = make_server()
        o = obs(1)
        tokens = srv._generate(o)
        prompt, img = srv._processor.calls[-1]
        self.assertEqual(prompt, "close top drawer")                                         # [a] raw instruction
        self.assertTrue(np.array_equal(img, cv2.resize(o["images"]["primary"], (224, 224),
                                                      interpolation=cv2.INTER_AREA)))       # [b]
        enc = StubProcessor()(prompt, img)
        ref = srv._model.reference_tokens(enc["input_ids"], enc["pixel_values"].to(torch.bfloat16))
        self.assertTrue(np.array_equal(tokens, ref), (tokens, ref))
        self.assertNotIn("attention_mask", srv._model.generate_kwargs[-1])
        self.assertEqual(type(srv._model.lm.model.layers[0].self_attn).__name__, "LlamaAttention")

    def test_negative_control_predict_action_with_processor_mask_crashes_under_eager(self):
        srv = make_server()
        o = obs(2)
        inputs = srv._processor("close top drawer", srv._preprocess_image(o))
        # the S-1 defect: eager attention adds a mask of length L+N_IMG to scores of length L+1+N_IMG
        with self.assertRaisesRegex(RuntimeError, "must match the size of tensor b .* at non-singleton dimension 3"):
            srv._model.predict_action(**inputs, unnorm_key="fractal20220817_data", do_sample=False)

    def test_sdpa_ignores_the_short_mask(self):   # why the LIBERO SDPA servers never hit S-1 (documentation)
        try:
            srv = make_server(attn="sdpa")
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"sdpa unavailable: {exc}")
        o = obs(3)
        inputs = srv._processor("close top drawer", srv._preprocess_image(o))
        srv._model.predict_action(**inputs, unnorm_key="fractal20220817_data", do_sample=False)


class TestPaths(unittest.TestCase):
    def test_s3_native_seeded_logits_agree_float64(self):
        srv = make_server()
        o = obs(4)
        a = srv.predict(o, types.SimpleNamespace(session_id="s", episode_id="e"))["actions"]
        self.assertEqual(a.dtype, np.float64)
        (b, eb), = srv.sample_chunks(o, [7], None, (), None)
        (c, ec), = srv.sample_chunks(o, [7], None, ("logits",), None)
        self.assertEqual(b.dtype, np.float64)
        self.assertEqual(a.tobytes(), b[0].tobytes())
        self.assertEqual(a.tobytes(), np.asarray(c[0]).tobytes())
        tokens = ec["logits_tokens"].astype(np.int64)
        self.assertTrue(np.array_equal(ec["logits"].argmax(-1), tokens))
        self.assertEqual(ec["logits"].dtype, np.float32)
        self.assertEqual(ec["logits"].shape, (AD, V_LOGITS))
        self.assertEqual(srv._actions(tokens).tobytes(), a.tobytes())
        # gripper is the raw bin centre (mask False), not -sign(2g-1)
        self.assertNotIn(a[-1], (-1.0, 1.0))

    def test_s3_bin_table_matches_actions_on_boundary_tokens(self):
        srv = make_server()
        table = srv._bin_table()
        self.assertEqual((table.shape, table.dtype), ((NB, AD), np.float64))
        self.assertEqual(table[255].tobytes(), table[254].tobytes())
        for t in (0, 31743, 31744, 31745, 31746, 31999, 32000, 32063):
            tok = np.full(AD, t)
            idx = np.clip(VOCAB - tok - 1, 0, NB - 1)
            self.assertEqual(table[idx, np.arange(AD)].tobytes(), srv._actions(tok).tobytes(), t)

    def test_refusals(self):
        srv = make_server()
        with self.assertRaises(ValueError):
            srv.sample_chunks(obs(), [1], {"factory": "hs_x:Y"}, (), None)
        with self.assertRaises(ValueError):
            srv.sample_chunks(obs(), [1], None, ("tokens",), None)
        with self.assertRaises(ValueError):
            srv.sample_chunks(obs(), [1], None, ("logits", "tokens"), None)
        with self.assertRaises(ValueError):
            srv._inputs(obs())
        with self.assertRaises(ValueError):
            srv._preprocess_image({"images": {"primary": np.zeros((4, 4, 3), np.float32)}})

    def test_a7_short_decode_is_infrastructure(self):
        srv = make_server()
        srv._model.lm.generation_config.eos_token_id = int(srv._generate(obs(5))[2])   # stop at the 3rd token
        with self.assertRaises(S.SimplerDecodeShort):
            srv._generate(obs(5))

    def test_a9_task_config_is_ignored(self):
        srv = make_server()
        a = srv.predict(obs(6), None)["actions"]
        import anyio
        anyio.run(srv.on_episode_start, {"task": {"name": "google_robot_close_drawer", "task_id": "seed16"}},
                  types.SimpleNamespace(session_id="s", episode_id="e"))
        self.assertEqual(srv.predict(obs(6), None)["actions"].tobytes(), a.tobytes())


class TestLogSpecsAttention(unittest.TestCase):
    def test_s4_precast_float64_log(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "actions.jsonl")
            srv = make_server(log=path)
            a = srv.predict(obs(8), types.SimpleNamespace(session_id="sess", episode_id="ep1"))["actions"]
            srv.sample_chunks(obs(8), [1], None, ("logits",), types.SimpleNamespace(session_id="sess", episode_id="ep1"))
            rows = [json.loads(x) for x in open(path)]
            self.assertEqual([r["path"] for r in rows], ["native", "logits"])
            self.assertEqual([r["seq"] for r in rows], [1, 2])
            self.assertEqual(bytes.fromhex(rows[0]["action_f64_hex"]), a.tobytes())
            self.assertEqual((rows[1]["session"], rows[1]["episode"]), ("sess", "ep1"))
        srv = make_server(log=None)
        srv.predict(obs(8), None)       # no log, no error

    def test_s5_specs_and_eager_assertion(self):
        srv = make_server()
        spec = srv.get_action_spec()
        self.assertEqual({k: v.format for k, v in spec.items()},
                         {"position": "delta_xyz", "rotation": "euler_xyz", "gripper": "raw"})
        self.assertEqual(srv.get_observation_params(), {})
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        srv._check_model()
        srv._model.lm.model.layers[0].self_attn.__class__ = type("LlamaSdpaAttention", (torch.nn.Module,), {})
        with self.assertRaises(RuntimeError):
            srv._check_model()
        srv2 = make_server()
        torch.backends.cudnn.benchmark = True
        try:
            with self.assertRaises(RuntimeError):
                srv2._check_model()
        finally:
            torch.backends.cudnn.benchmark = False

    def test_action_spec_matches_benchmark_spec(self):
        from vla_eval.specs import check_specs
        from harnessstudio.benchmarks.simpler_seeded import SimplerSeededBenchmark
        bench = SimplerSeededBenchmark("google_robot_pick_coke_can", [0])
        warnings = check_specs(make_server().get_action_spec(), bench.get_action_spec(), {}, {})
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
