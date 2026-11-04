"""Policy capability names (spec section 4.1).

T0: act. T1: chunk, seeded_sampling, isolated_streams. T2: denoise_hooks, velocity_readout,
differentiable_denoiser, decoding_hooks (autoregressive policies), residual_hooks. T3: attention_readout, logits_readout,
predicted_futures.

residual_hooks (EXT openvla-residual-hook, policies/openvla_simpler_star_hs.py): a server-side *residual hook* named by
the request's `sampler` {factory, params} rewrites decoder-layer outputs inside the policy's forward pass. Protocol:
hook.READS (observation keys it may read; the server passes a new dict with only those keys, from HOOK_READABLE),
hook.bind(total_layers), hook.init_state(), hook.begin(state, obs_view) -> (mixer, info), hook.next_state(state, info);
mixer.reset() at layer 0 of every forward, mixer.step(h_in, h_out) -> h per layer, optional
mixer.correction_start_layer (policy_ext/residual_hooks.py). It is distinct from decoding_hooks (token selection and
vision-attention temperature, openvla_hs.py): a decoding_hooks server calls bind(action_token_begin=...) and fails on
a residual hook, and a residual_hooks server refuses readouts with a sampler and more than one seed.
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
    "residual_hooks": "a server-side residual hook rewrites decoder-layer residual updates inside the forward pass",
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
