"""Real-Time Chunking (RTC), inference-time version (Black, Galliker, Levine; arXiv 2506.07339v2).

Guided inpainting of the next action chunk toward the previous chunk's unexecuted tail, with soft
prefix weights and a clipped pseudoinverse-guidance weight. The execution side (inference delay d,
execute horizon s, prev[:d] ++ new[d:s]) is PROTOCOL, applied identically to every arm by the
kinetix-flow backend; this module only steers the flow velocity through guide().

Numerics follow real-time-chunking-kinetix @9296f31d src/model.py:21-55 (get_prefix_weights) and
:236-251 (pinv_corrected_velocity). Pure function over an array namespace xp (jax.numpy at run time,
numpy in tier 1); no side effects (tier-1 guide_contract + mandatory source review).
"""
from __future__ import annotations

from ..core.capabilities import caps
from ..core.method import Fidelity, MethodConfig, ModuleRef

SCHEDULES = ("ones", "zeros", "linear", "exp")


def prefix_weights(xp, start: int, end: int, total: int, schedule: str):
    """Same arithmetic as upstream model.get_prefix_weights (start = d, end = H - s, total = H)."""
    start = min(start, end)
    if schedule == "ones":
        w = xp.ones(total)
    elif schedule == "zeros":
        w = (xp.arange(total) < start).astype(xp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = xp.clip((start - 1 - xp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * xp.expm1(w) / (xp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return xp.where(xp.arange(total) >= end, 0, w).astype(xp.float32)  # no-op cast under jax (float32 already)


class RTCGuidance:
    """guide(): v + min(beta, ((1-t)/t) * (t^2 + (1-t)^2) / (1-t)^2) * VJP(W * (prev - x1_hat))."""

    def __init__(self, schedule: str = "exp", max_guidance_weight: float = 5.0):
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if not float(max_guidance_weight) > 0:
            raise ValueError("max_guidance_weight must be positive")
        self.schedule = schedule
        self.max_guidance_weight = float(max_guidance_weight)
        self.identity = {"module": "rtc_guidance", "schedule": schedule, "max_guidance_weight": self.max_guidance_weight,
                         "source": "real-time-chunking-kinetix@9296f31d src/model.py:236-251"}

    def guide(self, xp, x_t, t, v_t, x1_hat, vjp, prev, d, s, H):
        weights = prefix_weights(xp, d, H - s, H, self.schedule)
        error = (prev - x1_hat) * weights[:, None]
        pinv_correction = vjp(error)
        inv_r2 = (t ** 2 + (1 - t) ** 2) / ((1 - t) ** 2)
        c = xp.nan_to_num((1 - t) / t, posinf=self.max_guidance_weight)
        guidance_weight = xp.minimum(c * inv_r2, self.max_guidance_weight)
        return (v_t + guidance_weight * pinv_correction).astype(v_t.dtype)  # no-op under jax; keeps numpy float32

    def extras(self) -> dict:
        return {}


def _config(name, schedule, description):
    return MethodConfig(
        name=name, version="1", paper="arXiv:2506.07339v2 (Real-Time Chunking, NeurIPS 2025)",
        fidelity=Fidelity.FAITHFUL, substitutions=(),
        required=caps("action_chunk", "seeded_sampling", "differentiable_denoiser", "async_execution"),
        slots={"sampler": ModuleRef("harnessstudio.methods.rtc:RTCGuidance",
                                    {"schedule": schedule, "max_guidance_weight": 5.0})},
        description=description)


def config():
    return _config("rtc", "exp", "RTC soft-masked guided inpainting (exp schedule, beta 5) under protocol-level "
                   "inference delay; kinetix-flow only")


def config_hard_masking():
    return _config("rtc-hard-masking", "zeros", "RTC ablation: hard masking (only the d frozen actions guided)")
