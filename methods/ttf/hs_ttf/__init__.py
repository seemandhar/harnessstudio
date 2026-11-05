"""TTF-VLA (Temporal Token Fusion, arXiv:2508.19257v3, AAAI 2026) as a HarnessStudio v2 slot method.

TTF changes OpenVLA's visual tokens inside the policy server: at non-keyframe steps a patch reuses the previous
step's token unless its grayscale pixel difference exceeds a threshold or it is among the top-k patches of the
previous step's text-to-vision attention (layer 15). The harness side is SCALE-like: one sample per decision (K = 1)
and the ``sampler`` slot naming the server-side vision-token hook (``hs_ttf.server_hook:TTFVisionHook``), served by
the EXT openvla-vision-token-hook server (policies/openvla_ttf_hs.py, capability ``vision_token_hooks``).

Entry points (one package, three arms of the predeclared SETTING):
- ``ttf``         the paper's method (release with aptube_enabled=True, fusion_mode=hybrid, K=3, tau=0.03, k=70);
- ``ttf-paper-baseline``  the release's own baseline path (aptube_enabled=False: stock tokens, but the decode still runs with
                  output_attentions=True, as every arm of the release); the paper-faithful BASELINE arm;
- ``ttf-shadow``  overhead/null control: the whole TTF computation runs (masks, fusion, cache, stored attention) and
                  nothing is written back; its actions must equal ``ttf-paper-baseline`` bitwise.
The framework's derived null of any entry point sends no sampler, i.e. the native vla-eval decode (SDPA, causal
prefill); it is a separately reported determinism/protocol control, never a pairing arm.

Nothing heavy is imported here: the harness process builds these components for tier 0 without torch.
"""
from __future__ import annotations

from pathlib import Path

from .constants import PAPER
from .server_hook import DEFAULTS, TTFVisionHook  # noqa: F401  (factory "hs_ttf.server_hook:TTFVisionHook")

FACTORY = "hs_ttf.server_hook:TTFVisionHook"


class SingleSample:
    """Proposer: one seeded sample per decision (TTF is single-pass; the hook's state follows the executed action)."""

    def __init__(self, num_samples: int = 1) -> None:
        if num_samples != 1:
            raise ValueError("TTF draws exactly one sample per decision")
        self.num_samples = 1


def _config(card: str, mode: str):
    from harnessstudio.core.card import config_from_card
    from harnessstudio.core.method import ModuleRef

    return config_from_card(Path(__file__).with_name(card), slots={
        "proposer": ModuleRef("hs_ttf:SingleSample", {"num_samples": 1}),
        "sampler": ModuleRef(FACTORY, dict(DEFAULTS, mode=mode)),
    })


def config():
    return _config("paper.yaml", "ttf")


def config_off():
    return _config("paper_off.yaml", "off")


def config_shadow():
    return _config("paper_shadow.yaml", "shadow")
