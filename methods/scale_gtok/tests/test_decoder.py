"""ScaleGripperArgmax vs ScaleDecoder on stub logits (review RC1-RC3). Needs torch: run with an interpreter that has
torch WITHOUT installing anything into it, e.g. (CPU only, no bytecode written):

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH=harnessstudio:harnessstudio/methods/scale:harnessstudio/methods/scale_gtok \
    /home/sejain/.cache/uv/environments-v2/openvla-hs-73218eaab6c13068/bin/python -m unittest discover \
      -s harnessstudio/methods/scale_gtok/tests -v

The stub follows the server's protocol (openvla_hs.py e7c6a4c3 _decode): per decision a fresh
torch.Generator().manual_seed(seed), attention_temperature(state) once, then select_token once per action token,
with the next token's logits depending on the tokens chosen so far (like the KV-cached prefix), then
next_state(state, infos).
"""
from __future__ import annotations

import hashlib
import unittest

try:
    import torch
except ImportError:  # the HarnessStudio venv has no torch
    torch = None

BEGIN, END, VOCAB, DIM = 31744, 32000, 32000, 7


def stub_logits(history, decision, gamma):
    """Deterministic [vocab] logits from the decision, gamma and the tokens so far. Action-bin logits alternate
    between flat (high u -> SCALE samples) and peaked (low u), and the 7th token gets a two-mode gripper
    distribution, so SCALE's sampled gripper token differs from the argmax in some decisions."""
    key = f"{decision}|{gamma!r}|{list(history)}".encode()
    g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:6], "big"))
    logits = torch.randn(VOCAB, generator=g) * 2.0
    pos = len(history)
    spread = 0.6 if (decision + pos) % 2 else 4.0
    logits[BEGIN:END] = torch.randn(END - BEGIN, generator=g) * spread + 5.0
    if pos == DIM - 1:
        logits[BEGIN:END] = -4.0 + torch.randn(END - BEGIN, generator=g) * 0.1
        logits[BEGIN + 3] = 6.0                                    # 'open'-ish bin
        logits[END - 2] = 6.0 - (0.3 if decision % 3 else 1.5)    # 'close'-ish bin, close second mode
    return logits


def run_decisions(decoder, seeds):
    decoder.bind(BEGIN, END, DIM)
    state = decoder.init_state()
    out = []
    for d, seed in enumerate(seeds):
        gen = torch.Generator().manual_seed(seed)
        gamma = decoder.attention_temperature(state)
        tokens, infos, logits7 = [], [], None
        for _ in range(DIM):
            logits = stub_logits(tokens, d, gamma)
            if len(tokens) == DIM - 1:
                logits7 = logits.clone()
            token, info = decoder.select_token(logits, gen)
            tokens.append(int(token))
            infos.append(dict(info))
        out.append({"tokens": tokens, "infos": infos, "gamma": gamma, "state": state, "logits7": logits7})
        state = decoder.next_state(state, infos)
    out.append({"final_state": state})
    return out


@unittest.skipIf(torch is None, "needs torch (see module docstring)")
class GripperArgmax(unittest.TestCase):
    SEEDS = [1071524933, 2477133818, 7, 123456789, 3141592653, 2718281828, 42, 99, 4000000000, 1]

    def setUp(self):
        from hs_scale.decoder import DEFAULTS, ScaleDecoder
        from hs_scale_gtok.decoder import ScaleGripperArgmax
        self.scale = run_decisions(ScaleDecoder(**DEFAULTS), self.SEEDS)
        self.gtok = run_decisions(ScaleGripperArgmax(**DEFAULTS), self.SEEDS)
        self.ScaleGripperArgmax, self.DEFAULTS = ScaleGripperArgmax, DEFAULTS

    def test_motion_tokens_u_tau_gamma_state_equal_scale(self):
        self.assertGreaterEqual(len(self.SEEDS), 3)
        for d, (a, b) in enumerate(zip(self.scale[:-1], self.gtok[:-1])):
            self.assertEqual(a["tokens"][:DIM - 1], b["tokens"][:DIM - 1], f"decision {d}")
            self.assertEqual([i["u"] for i in a["infos"]], [i["u"] for i in b["infos"]], f"decision {d}")
            self.assertEqual([i["tau"] for i in a["infos"]], [i["tau"] for i in b["infos"]], f"decision {d}")
            self.assertEqual(set(b["infos"][-1]), {"u", "tau"})
            self.assertEqual(a["gamma"], b["gamma"])
            self.assertEqual(a["state"], b["state"])
        self.assertEqual(self.scale[-1]["final_state"], self.gtok[-1]["final_state"])
        self.assertTrue(any(x["gamma"] != 1.0 for x in self.gtok[:-1]), "gamma never left 1.0: vacuous test")

    def test_gripper_token_is_masked_argmax(self):
        for d, b in enumerate(self.gtok[:-1]):
            argmax = BEGIN + int(torch.argmax(b["logits7"][BEGIN:END]).item())
            self.assertEqual(b["tokens"][DIM - 1], argmax, f"decision {d}")

    def test_scale_would_have_sampled_differently(self):
        """Non-vacuous: SCALE's own gripper token differs from the argmax in some decision, with tau_7 >= 1e-3."""
        diffs = [d for d, (a, b) in enumerate(zip(self.scale[:-1], self.gtok[:-1]))
                 if a["tokens"][DIM - 1] != b["tokens"][DIM - 1]]
        self.assertTrue(diffs, "stub never makes SCALE's gripper sample differ from argmax")
        self.assertTrue(all(self.gtok[d]["infos"][-1]["tau"] >= 1e-3 for d in diffs))
        # Sabotage check: the unmodified ScaleDecoder fails the argmax property on these decisions.
        for d in diffs:
            argmax = BEGIN + int(torch.argmax(self.scale[d]["logits7"][BEGIN:END]).item())
            self.assertNotEqual(self.scale[d]["tokens"][DIM - 1], argmax)

    def test_counter_refusals(self):
        dec = self.ScaleGripperArgmax(**self.DEFAULTS)
        dec.bind(BEGIN, END, DIM)
        logits = stub_logits([], 0, 1.0)
        gen = torch.Generator().manual_seed(1)
        with self.assertRaises(RuntimeError):            # no decision open
            dec.select_token(logits, gen)
        dec.attention_temperature(dec.init_state())
        for _ in range(DIM):
            dec.select_token(logits, gen)
        with self.assertRaises(RuntimeError):            # 8th call since the reset (openvla-ksample K > 1 pattern)
            dec.select_token(logits, gen)
        dec.attention_temperature(dec.init_state())       # a new decision resets the counter
        dec.select_token(logits, gen)
        with self.assertRaises(ValueError):              # batched [B, vocab] logits are refused
            dec.select_token(torch.stack([logits, logits]), gen)

    def test_aborted_decision_recovers(self):
        """A decision that stopped mid-loop (server error) does not shift the next decision's positions."""
        dec = self.ScaleGripperArgmax(**self.DEFAULTS)
        dec.bind(BEGIN, END, DIM)
        gen = torch.Generator().manual_seed(5)
        dec.attention_temperature(dec.init_state())
        for k in range(3):
            dec.select_token(stub_logits([0] * k, 0, 1.0), gen)
        ref = run_decisions(self.ScaleGripperArgmax(**self.DEFAULTS), [11])
        dec2 = run_decisions(dec, [11])     # run_decisions re-binds and starts a fresh decision on the same object
        self.assertEqual(ref[0]["tokens"], dec2[0]["tokens"])


if __name__ == "__main__":
    unittest.main()
