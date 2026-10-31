"""Policy capability names (spec section 4.1).

T0: act. T1: chunk, seeded_sampling, isolated_streams. T2: denoise_hooks, velocity_readout,
differentiable_denoiser, decoding_hooks (autoregressive policies). T3: attention_readout, logits_readout, predicted_futures.
"""
from __future__ import annotations

from .errors import RegistrationError

CAPABILITIES: dict[str, str] = {
    "act": "one action or chunk per observation (any vla-eval policy server)",
    "chunk": "returns the full action chunk for a seeded request",
    "seeded_sampling": "the caller sets the noise seed of each sample",
    "isolated_streams": "each sample depends only on its own seed",
    "denoise_hooks": "Sampler.on_step runs at every denoising step",
    "velocity_readout": "flow velocities are returned as extras",
    "differentiable_denoiser": "Sampler.guide gets a VJP of the denoised estimate",
    "decoding_hooks": "a server-side Sampler changes autoregressive token decoding (temperature, attention)",
    "attention_readout": "action-token attention is returned as extras",
    "logits_readout": "token logits are returned as extras",
    "predicted_futures": "predicted future frames are returned as extras",
}

T0 = frozenset({"act"})
T1 = T0 | {"chunk", "seeded_sampling", "isolated_streams"}


def check(names) -> frozenset[str]:
    names = frozenset(names)
    unknown = names - set(CAPABILITIES)
    if unknown:
        raise RegistrationError(f"Unknown capabilities {sorted(unknown)}; known: {sorted(CAPABILITIES)}")
    return names
