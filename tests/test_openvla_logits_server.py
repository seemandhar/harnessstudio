"""EXT openvla-contrast, part 2: the REAL recorded decode of HarnessOpenVLALogitsServer (CPU, torch, transformers).

Needs torch + transformers 4.40.1 (the pinned OpenVLA server env); skipped elsewhere. Run (review R6):

    env -u TRANSFORMERS_CACHE -u HF_HUB_CACHE CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=<scratch> HF_MODULES_CACHE=<scratch>/modules PYTHONPATH=<worktree> \
      <openvla env python> -B -m unittest tests.test_openvla_logits_server -v

The server is built with __new__ (never __init__, which loads weights). Its model is a stub with Prismatic's
predict_action logic (remote code modeling_prismatic.py:506-536) around a tiny random LlamaForCausalLM whose
greedy decode runs through the REAL transformers `generate` / `_greedy_search`, so the recorders see exactly what
the argmax sees. The image reaches the model through the stub processor (pixels hashed into the prompt ids).
Covers U3 (predict copy AST), U4 + A11 (recorder rows float32 and bitwise the forward's last-position logits;
returned object identity), R3 (token-level self-check) and its sabotage, R2 refusals on the real path, the
EOS-early path, lock ownership, and the override positive control on the stub (R4).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap
import threading
import types
import unittest
from unittest import mock

import numpy as np

try:
    import torch
    import transformers
    from transformers import BatchFeature, LlamaConfig, LlamaForCausalLM
except ImportError:
    raise unittest.SkipTest("needs torch + transformers (run with the OpenVLA server env)")

from vla_eval.model_servers.openvla import OpenVLAModelServer  # noqa: E402
from harnessstudio.policies import openvla_hs as H  # noqa: E402
from harnessstudio.policies import openvla_logits_hs as L  # noqa: E402

OPENVLA_HS_SHA256 = "e7c6a4c302fece60b338b9a6c14c7ac0cea4dea2057ff72b74c0abfa0aab174b"
V_LOGITS, VOCAB, NB, AD, EMPTY = 384, 320, 256, 7, 5   # mirrors 32064 / 32000 / 256 / 7 / 29871


class StubProcessor:
    def __call__(self, prompt, pil_image):
        digest = hashlib.sha256(prompt.encode() + np.asarray(pil_image).tobytes()).digest()
        ids = [1] + [6 + b % 58 for b in digest[:11]]
        return BatchFeature({"input_ids": torch.tensor([ids], dtype=torch.long),
                             "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                             "pixel_values": torch.tensor(np.asarray(pil_image, dtype=np.float32))[None]})


class StubVLA:
    """Prismatic's predict_action around a tiny real Llama (greedy decoding through transformers.generate)."""

    def __init__(self, server, seed=0):
        torch.manual_seed(seed)
        cfg = LlamaConfig(vocab_size=V_LOGITS, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                          bos_token_id=1, eos_token_id=2, pad_token_id=0)
        self.lm = LlamaForCausalLM(cfg).eval()
        with torch.no_grad():                    # favour the action-token range, like OpenVLA after "Out:"
            self.lm.lm_head.weight[VOCAB - NB: VOCAB] *= 4.0
        self.lm.generation_config.eos_token_id = None          # set per test for the EOS path
        self.lm.generation_config.pad_token_id = 0
        self.server = server
        self.vocab_size = VOCAB
        self.config = types.SimpleNamespace(n_action_bins=NB, text_config=types.SimpleNamespace(vocab_size=V_LOGITS))
        bins = np.linspace(-1.0, 1.0, NB)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        rng = np.random.default_rng(seed)
        q01, q99 = rng.uniform(-1, -0.1, AD), rng.uniform(0.1, 1, AD)
        q01[-1], q99[-1] = 0.0, 1.0
        self.stats = {"q01": list(q01), "q99": list(q99), "mask": [True] * (AD - 1) + [False]}
        self.forward_last = []                   # outputs.logits[:, -1, :] of every forward (A11)
        hook_self = self

        def hook(module, args, kwargs, output):
            hook_self.forward_last.append(output.logits[:, -1, :].detach().clone())
        self.lm.register_forward_hook(hook, with_kwargs=True)

    def get_action_dim(self, unnorm_key=None):
        return AD

    def get_action_stats(self, unnorm_key=None):
        return self.stats

    def predict_action(self, input_ids=None, unnorm_key=None, **kwargs):
        assert self.server._infer_lock._is_owned(), "inference outside the server lock"
        kwargs.pop("pixel_values", None)
        kwargs.pop("attention_mask", None)
        if not torch.all(input_ids[:, -1] == EMPTY):
            input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([EMPTY]).long(), dim=0)), dim=1)
        generated_ids = self.lm.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)
        self.last_generated = generated_ids[0, input_ids.shape[1]:].tolist()
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        stats = self.get_action_stats(unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        high, low = np.array(stats["q99"]), np.array(stats["q01"])
        return np.where(mask, 0.5 * (normalized_actions + 1) * (high - low) + low, normalized_actions)


def make_server(seed=0, cls=L.HarnessOpenVLALogitsServer):
    s = cls.__new__(cls)
    s._infer_lock = threading.RLock()
    s._sampler_prefixes, s._decoders, s._states = ("hs_",), {}, {}
    s._device = torch.device("cpu")
    s.unnorm_key, s.jpeg_roundtrip, s.center_crop = "stub", True, True
    s._processor = StubProcessor()
    s._model = StubVLA(s, seed)
    return s


def obs(level=40, text="put the bowl on the plate"):
    img = (np.arange(32 * 32 * 3).reshape(32, 32, 3) * 7 + level).astype(np.uint8)
    return {"images": {"agentview": img}, "task_description": text, "states": np.zeros(8)}


class TestPins(unittest.TestCase):
    def test_parent_file_pinned(self):
        with open(H.__file__, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), OPENVLA_HS_SHA256)
        self.assertEqual(transformers.__version__, "4.40.1")

    def test_predict_copy_differs_only_by_the_recorders(self):
        """U3: AST of _predict_recorded == vla-eval predict, minus the recorder lines, two kwargs and the return."""
        def body(fn):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            return tree.body[0].body
        native = body(OpenVLAModelServer.predict)
        ours = body(L.HarnessOpenVLALogitsServer._predict_recorded)

        def strip(stmts, ours_side):
            out = []
            for st in stmts:
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):      # docstring
                    continue
                if ours_side and isinstance(st, ast.ImportFrom) and st.module == "transformers":
                    continue
                if ours_side and isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) \
                        and getattr(st.value.func, "id", None) == "_recorders":
                    continue
                if isinstance(st, ast.Return):
                    continue
                if ours_side and isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) \
                        and getattr(st.value.func, "attr", None) == "predict_action":
                    st = ast.Assign(targets=st.targets, value=ast.Call(
                        func=st.value.func, args=st.value.args,
                        keywords=[k for k in st.value.keywords if k.arg not in ("logits_processor", "stopping_criteria")]),
                        lineno=0)
                out.append(ast.dump(st, include_attributes=False))
            return out
        self.assertEqual(strip(native, False), strip(ours, True))
        returns = [st for st in native if isinstance(st, ast.Return)]
        self.assertEqual(ast.dump(returns[0].value), "Dict(keys=[Constant(value='actions')], values=[Name(id='action_arr', ctx=Load())])")


class TestRecordedDecode(unittest.TestCase):
    def test_native_action_unchanged_and_rows_exact(self):
        s = make_server()
        o = obs()
        with s._infer_lock:
            native = OpenVLAModelServer.predict(s, o, None)["actions"]
        s._model.forward_last.clear()
        (chunk, extras), = s.sample_chunks(o, [11], None, ("logits",), None)
        self.assertTrue(np.array_equal(chunk[0], native))                     # recorders change nothing
        rows = extras["logits"]
        self.assertEqual(rows.dtype, np.float32)
        self.assertEqual(rows.shape, (AD, V_LOGITS))
        fwd = torch.cat(s._model.forward_last).numpy()                         # A11: the forward's last logits
        self.assertTrue(np.array_equal(rows, fwd[:AD]))
        self.assertEqual(list(extras["logits_tokens"]), s._model.last_generated)
        self.assertTrue(np.array_equal(extras["logits_tokens"], rows.argmax(-1)))
        table = extras["logits_bin_actions"]
        idx = np.clip(extras["logits_token_end"] - extras["logits_tokens"].astype(np.int64) - 1, 0, extras["logits_n_bins"] - 1)
        self.assertTrue(np.array_equal(table[idx, np.arange(AD)], native))      # table action == native action
        self.assertEqual((extras["logits_token_end"], extras["logits_n_bins"]), (VOCAB, NB))

    def test_recorder_returns_the_same_tensor(self):
        rec, tok = L._recorders()
        x = torch.randn(1, 9)
        self.assertIs(rec(torch.zeros(1, 3, dtype=torch.long), x), x)
        self.assertTrue(torch.equal(rec.rows[0], x[0]))
        done = tok(torch.tensor([[4, 5, 6]]), x)
        self.assertEqual(tok.tokens, [6])
        self.assertFalse(bool(done.any()))

    def test_seed_independent_and_deterministic(self):
        s = make_server()
        out = s.sample_chunks(obs(), [1, 2], None, ("logits",), None)
        self.assertTrue(np.array_equal(out[0][0], out[1][0]))
        self.assertTrue(np.array_equal(out[0][1]["logits"], out[1][1]["logits"]))

    def test_override_positive_control(self):
        """R4 on the stub: a different first image changes logits; the action equals a plain native request on it."""
        s = make_server()
        a, b = obs(40), obs(41)
        (_, ea), = s.sample_chunks(a, [0], None, ("logits",), None)
        (cb, eb), = s.sample_chunks(b, [0], None, ("logits",), None)
        self.assertFalse(np.array_equal(ea["logits"], eb["logits"]))
        with s._infer_lock:
            plain_b = OpenVLAModelServer.predict(s, b, None)["actions"]
        self.assertTrue(np.array_equal(cb[0], plain_b))

    def test_eos_early_returns_ragged_rows(self):
        s = make_server()
        (_, full), = s.sample_chunks(obs(), [0], None, ("logits",), None)
        eos = int(full["logits_tokens"][2])
        t_stop = list(full["logits_tokens"]).index(eos) + 1          # generation ends at the first eos
        s._model.lm.generation_config.eos_token_id = eos
        (_, short), = s.sample_chunks(obs(), [0], None, ("logits",), None)
        self.assertLess(t_stop, AD)
        self.assertEqual(short["logits"].shape[0], t_stop)
        self.assertTrue(np.array_equal(short["logits"], full["logits"][:t_stop]))

    def test_no_readout_and_other_readouts_take_the_parent(self):
        s = make_server()
        o = obs()
        with mock.patch.object(L.HarnessOpenVLALogitsServer, "_predict_recorded", side_effect=AssertionError):
            (chunk, extras), = s.sample_chunks(o, [3], None, (), None)
            with self.assertRaisesRegex(ValueError, r"^this OpenVLA server offers no readouts"):
                s.sample_chunks(o, [3], None, ("velocity",), None)
        with s._infer_lock:
            native = OpenVLAModelServer.predict(s, o, None)["actions"]
        self.assertTrue(np.array_equal(chunk[0], native))
        self.assertEqual(extras, {"seed": 3})

    def test_logits_with_sampler_refused(self):
        s = make_server()
        with self.assertRaises(ValueError):
            s.sample_chunks(obs(), [0], {"factory": "hs_x.y:Z", "params": {}}, ("logits",), None)


class TestSelfCheckSabotage(unittest.TestCase):
    """R3/U9: the token-level self-check must catch recorders that lie."""

    def _with_recorders(self, mutate):
        real = L._recorders

        def bad():
            rec, tok = real()
            orig = rec.__call__.__func__

            class Bad(type(rec)):
                def __call__(self, input_ids, scores):
                    out = orig(self, input_ids, scores)
                    mutate(self)
                    return out
            rec.__class__ = Bad
            return rec, tok
        return mock.patch.object(L, "_recorders", bad)

    def test_perturbed_gripper_row_same_sign_is_caught(self):
        s = make_server()
        (_, ok), = s.sample_chunks(obs(), [0], None, ("logits",), None)
        tok6 = int(ok["logits_tokens"][6])
        tbl = ok["logits_bin_actions"]
        bin_of = lambda t: min(max(VOCAB - t - 1, 0), NB - 1)  # noqa: E731
        # move the gripper argmax to another token that de-tokenizes to the same -sign(2x-1) (action check blind)
        alt = next(t for t in range(VOCAB - NB, VOCAB) if t != tok6 and tbl[bin_of(t), 6] == tbl[bin_of(tok6), 6])

        def mutate(rec):
            if len(rec.rows) == AD:
                rec.rows[-1] = rec.rows[-1].clone()
                rec.rows[-1][alt] = rec.rows[-1].max() + 1.0
        with self._with_recorders(mutate):
            with self.assertRaises(L.RecordedDecodeMismatch):
                s.sample_chunks(obs(), [0], None, ("logits",), None)

    def test_recording_the_next_position_is_caught(self):
        s = make_server()

        def mutate(rec):
            if len(rec.rows) == 2 and not getattr(rec, "_dropped", False):
                rec._dropped = True
                del rec.rows[0]                         # rows now hold positions t+1
        with self._with_recorders(mutate):
            with self.assertRaises(L.RecordedDecodeMismatch):
                s.sample_chunks(obs(), [0], None, ("logits",), None)

    def test_bfloat16_rows_are_refused(self):
        s = make_server()

        def mutate(rec):
            rec.rows[-1] = rec.rows[-1].to(torch.bfloat16)
        with self._with_recorders(mutate):
            with self.assertRaises(L.RecordedDecodeMismatch):
                s.sample_chunks(obs(), [0], None, ("logits",), None)


class TestEnvironment(unittest.TestCase):
    def test_no_cuda_visible(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "", "run with CUDA_VISIBLE_DEVICES= (empty)")
        self.assertFalse(torch.cuda.is_available())

    def test_scratch_hf_cache(self):
        """R6: no shared HF/transformers cache is reachable from this test process."""
        self.assertNotIn("TRANSFORMERS_CACHE", os.environ)
        self.assertNotIn("HF_HUB_CACHE", os.environ)
        for key in ("HF_HOME", "HF_MODULES_CACHE"):        # a scratch directory, never the pinned server caches
            value = os.environ.get(key, "")
            self.assertTrue(value, f"set {key} to a scratch directory")
            self.assertFalse(value.startswith(("/bigdata/sejain/hs2-cache", "/bigdata/sejain/aspire-cache")), value)
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")


if __name__ == "__main__":
    unittest.main()
