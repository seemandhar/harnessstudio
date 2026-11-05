"""Toy vision-token hook for tests/test_ext_openvla_vision_token_hook.py (EXT openvla-vision-token-hook).

``mode="zero_siglip"`` replaces the SigLIP half of the patches by zeros; ``mode="stock"`` returns them unchanged.
State counts requests per episode and keeps the last prefill attention it received. No torch import at module level.
"""
from __future__ import annotations


class ToyState:
    def __init__(self):
        self.requests = 0
        self.last_attention = None


class ToyVisionHook:
    VISION_TOKEN_HOOK = True
    OUTPUT_ATTENTIONS = True

    def __init__(self, mode: str = "zero_siglip", layer: int = 1):
        self.mode = mode
        self.ATTENTION_LAYERS = (int(layer),)
        self.bound = None

    def bind(self, n_patches, dims, num_layers):
        self.bound = (int(n_patches), tuple(dims), int(num_layers))

    def init_state(self):
        return ToyState()

    def vision(self, state, pixel_values, new_dino, new_siglip):
        import torch

        state.requests += 1
        sig = torch.zeros_like(new_siglip) if self.mode == "zero_siglip" else new_siglip
        return torch.cat([new_dino, sig], dim=2), {"requests": state.requests}

    def after_decode(self, state, prefill_attentions, info):
        state.last_attention = {k: tuple(v.shape) for k, v in prefill_attentions.items()}
        return dict(info, layers=sorted(prefill_attentions))


class NotAHook:
    """A decoding-hook-like object: must be refused on the vision path and passed to the parent."""

    def __init__(self, **kw):
        pass
