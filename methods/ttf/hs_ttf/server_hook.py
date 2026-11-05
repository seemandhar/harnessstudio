"""TTF-VLA as a HarnessStudio vision-token hook (EXT openvla-vision-token-hook, capability ``vision_token_hooks``).

The server (``harnessstudio/policies/openvla_ttf_hs.py`` in the extension) imports this class by its factory name
(``hs_ttf.server_hook:TTFVisionHook``), binds it once and, for every request of an episode, calls

    vision(state, pixel_values, new_dino, new_siglip) -> (patches, info)     inside the prefill's vision forward
    after_decode(state, prefill_attentions, info) -> info                     after generate()

with the request's generate() running ``output_attentions=OUTPUT_ATTENTIONS`` (the release decodes every arm with
``output_attentions=True``, modeling_prismatic.py:742-747 and :792-799). Three modes, one class:

- ``mode="ttf"``     the enabled release (aptube_enabled=True, fusion_mode=hybrid by default): fused patches go to
                     the projector; the prefill attention of layer 15 is stored for the next step.
- ``mode="off"``     the release with aptube_enabled=False (its own baseline): stock patches, no state, but the decode
                     still runs with output_attentions=True ("Let output_attentions=True to make sure fair comparison",
                     modeling_prismatic.py:790-791).
- ``mode="shadow"``  overhead/null control: computes everything ``ttf`` computes (masks, fusion, cache, stored
                     attention) and writes nothing back: the projector receives the stock patches, so its actions
                     must equal ``off`` bitwise.
"""
from __future__ import annotations

from typing import Any, Mapping

from .constants import DINO_DIM, MULTI_LAYER_IDS, N_PATCHES, PAPER, SIGLIP_DIM

DEFAULTS: dict = {"mode": "ttf", **PAPER}
MODES = ("ttf", "off", "shadow")


class TTFVisionHook:
    VISION_TOKEN_HOOK = True          # the server refuses samplers without this marker on this path
    OUTPUT_ATTENTIONS = True          # every release arm decodes with output_attentions=True

    def __init__(self, **params: Any) -> None:
        cfg = dict(DEFAULTS)
        unknown = set(params) - set(cfg)
        if unknown:
            raise ValueError(f"unknown TTF params {sorted(unknown)}")
        cfg.update(params)
        if cfg["mode"] not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {cfg['mode']!r}")
        if cfg["fusion_mode"] not in ("pixel", "attention", "hybrid"):
            raise ValueError(f"fusion_mode {cfg['fusion_mode']!r} is not ported (the README's 'attention_guided' is "
                             "rejected by the release itself, modeling_prismatic.py:191)")
        if cfg["attention_mode"] not in ("text", "action"):
            raise ValueError(f"attention_mode {cfg['attention_mode']!r}")
        self.cfg = cfg
        self.mode = cfg["mode"]
        layers = list(MULTI_LAYER_IDS) if cfg["use_multi_layer"] else [int(cfg["attention_layer_id"])]
        # Attentions are needed only by modes that run the fusion state machine.
        self.ATTENTION_LAYERS = tuple(layers) if self.mode in ("ttf", "shadow") else ()
        self._bound = False

    # ---- server protocol ---------------------------------------------------------------------------------------
    def bind(self, n_patches: int, dims: tuple, num_layers: int) -> None:
        if int(n_patches) != N_PATCHES or tuple(int(d) for d in dims) != (DINO_DIM, SIGLIP_DIM):
            raise ValueError(f"TTF expects 256 patches of dims (1024, 1152), got {n_patches}, {dims}")
        if any(layer >= int(num_layers) for layer in self.ATTENTION_LAYERS):
            raise ValueError(f"attention layers {self.ATTENTION_LAYERS} >= num_layers {num_layers}")
        self._bound = True

    def init_state(self):
        from . import ttf_core  # torch: imported in the policy server only

        return ttf_core.TTFState()

    def vision(self, state, pixel_values, new_dino, new_siglip):
        """Return the patches the projector receives this request (plus logging info)."""
        import torch

        from . import ttf_core

        stock = torch.cat([new_dino, new_siglip], dim=2)
        if self.mode == "off":
            return stock, {"mode": "off"}
        fused, info = ttf_core.fuse(state, pixel_values, new_dino, new_siglip, self.cfg)
        info["mode"] = self.mode
        info["counters"] = dict(state.counters)
        if self.mode == "shadow":
            return stock, info
        return fused, info

    def after_decode(self, state, prefill_attentions: Mapping[int, Any], info: dict) -> dict:
        if self.mode == "off":
            return info
        from . import ttf_core

        ttf_core.store_attention(state, prefill_attentions)
        return info

    def params(self) -> dict:
        return dict(self.cfg)
