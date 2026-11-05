"""Vision-token hooks for OpenVLA-style servers (EXT openvla-vision-token-hook, capability ``vision_token_hooks``).

A *vision-token hook* is a server-side object named by the request's ``sampler`` {factory, params}. It sees the
patch tokens the fused vision backbone (DINOv2 + SigLIP) produced for the request's image and returns the patch
tokens the projector receives instead, and after the decode it receives the prefill attention of the decoder layers
it asked for. Used by TTF-VLA (temporal token fusion, arXiv:2508.19257v3); any method that edits or caches visual
tokens across the steps of an episode fits the same protocol.

Protocol (checked by :func:`check_hook`):

    VISION_TOKEN_HOOK = True                  marker; a decoding_hooks or residual_hooks object is refused
    OUTPUT_ATTENTIONS: bool                   decode with generate(output_attentions=True) (the request's decode
                                              numerics then follow transformers' attention fallback; see the doc)
    ATTENTION_LAYERS: tuple[int, ...]         decoder layers whose PREFILL attention [1, heads, seq, seq] is passed
                                              to after_decode (needs OUTPUT_ATTENTIONS when non-empty)
    bind(n_patches, dims, num_layers)         once, after loading
    init_state() -> state                     first hooked request of an episode (dropped at EPISODE_START/END)
    vision(state, pixel_values, new_dino, new_siglip) -> (patches, info)
                                              exactly once per request, inside the prefill's vision forward;
                                              patches must have the stock shape and dtype
    after_decode(state, prefill_attentions, info) -> info
                                              once per request, after generate()

The server never changes the stock vision forward for requests without such a sampler (baseline, derived null,
native requests): :func:`vision_forward_hook` is a context manager that swaps ``vision_backbone.forward`` only for the
duration of one hooked request, under the server's inference lock, and restores it in ``finally``.
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable


class VisionTokenHookError(RuntimeError):
    """A hooked request violated the protocol (an infrastructure failure, never a policy outcome)."""


REQUIRED = ("bind", "init_state", "vision", "after_decode")


def is_vision_token_hook(obj: Any) -> bool:
    return getattr(obj, "VISION_TOKEN_HOOK", False) is True


def check_hook(obj: Any) -> None:
    if not is_vision_token_hook(obj):
        raise VisionTokenHookError(f"{type(obj).__name__} is not a vision-token hook (VISION_TOKEN_HOOK is not True)")
    missing = [name for name in REQUIRED if not callable(getattr(obj, name, None))]
    if missing:
        raise VisionTokenHookError(f"{type(obj).__name__} lacks {missing}")
    layers = tuple(getattr(obj, "ATTENTION_LAYERS", ()))
    if layers and not bool(getattr(obj, "OUTPUT_ATTENTIONS", False)):
        raise VisionTokenHookError("ATTENTION_LAYERS needs OUTPUT_ATTENTIONS = True")
    if not all(isinstance(layer, int) and layer >= 0 for layer in layers):
        raise VisionTokenHookError(f"ATTENTION_LAYERS must be non-negative ints, got {layers!r}")


@contextlib.contextmanager
def vision_forward_hook(vision_backbone: Any, fn: Callable[[Any, Any, Any], Any]):
    """Replace ``vision_backbone.forward`` by the stock fused split + ``fn(pixel_values, new_dino, new_siglip)``.

    The stock forward of OpenVLA's PrismaticVisionBackbone (fused case) is::

        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)

    The hooked forward computes the same two featurizer outputs and returns ``fn``'s patches. Yields a dict that
    counts calls (``calls``) so the caller can assert exactly one vision forward per request.
    """
    import torch

    if not getattr(vision_backbone, "use_fused_vision_backbone", False):
        raise VisionTokenHookError("vision-token hooks need the fused (DINOv2 + SigLIP) vision backbone")
    record = {"calls": 0}
    had_instance_forward = "forward" in vision_backbone.__dict__
    saved = vision_backbone.__dict__.get("forward")

    def hooked_forward(pixel_values):
        record["calls"] += 1
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        new_dino, new_siglip = vision_backbone.featurizer(img), vision_backbone.fused_featurizer(img_fused)
        patches = fn(pixel_values, new_dino, new_siglip)
        expected = (new_dino.shape[0], new_dino.shape[1], new_dino.shape[2] + new_siglip.shape[2])
        if tuple(patches.shape) != expected or patches.dtype != new_dino.dtype:
            raise VisionTokenHookError(f"hook returned patches {tuple(patches.shape)} {patches.dtype}, "
                                       f"expected {expected} {new_dino.dtype}")
        return patches

    vision_backbone.forward = hooked_forward
    try:
        yield record
    finally:
        if had_instance_forward:
            vision_backbone.forward = saved
        else:
            vision_backbone.__dict__.pop("forward", None)


EMPTY_TOKEN = 29871  # Llama "▁": OpenVLA's predict_action appends it after "Out:" (openvla_hs.EMPTY_TOKEN)


def generate_with_hook(model: Any, hook: Any, state: Any, inputs: Any, action_dim: int) -> tuple:
    """One hooked request on an OpenVLA model: the TTF-VLA release's predict_action call.

    ``inputs`` is the processor output (``input_ids``, ``attention_mask``, ``pixel_values``), exactly what the native
    ``predict_action(**inputs, do_sample=False)`` receives. Like OpenVLA's predict_action (and the TTF release,
    modeling_prismatic.py:714-800), token 29871 is appended to ``input_ids`` only (the processor mask stays one
    shorter, P-20261103-11), then::

        model.generate(input_ids, max_new_tokens=action_dim, output_attentions=hook.OUTPUT_ATTENTIONS,
                       return_dict_in_generate=True, attention_mask=..., pixel_values=..., do_sample=False)

    with the vision backbone's forward swapped so ``hook.vision`` picks the projector's patches. Returns
    ``(tokens, prefill_attentions, info, shape)``; raises VisionTokenHookError on any shape violation.
    """
    import torch

    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == EMPTY_TOKEN):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([EMPTY_TOKEN]).long(), dim=0).to(input_ids.device)), dim=1)
    out_attn = bool(getattr(hook, "OUTPUT_ATTENTIONS", False))
    layers = tuple(getattr(hook, "ATTENTION_LAYERS", ()))
    holder: dict = {}

    def fn(pixel_values, new_dino, new_siglip):
        patches, info = hook.vision(state, pixel_values, new_dino, new_siglip)
        holder["info"] = dict(info or {})
        return patches

    kwargs = {k: v for k, v in dict(inputs).items() if k != "input_ids"}   # attention_mask, pixel_values
    with vision_forward_hook(model.vision_backbone, fn) as rec:
        out = model.generate(input_ids, max_new_tokens=int(action_dim), output_attentions=out_attn,
                             return_dict_in_generate=True, do_sample=False, **kwargs)
    sequences = out.sequences
    problems = []
    if rec["calls"] != 1:
        problems.append(f"{rec['calls']} vision forwards, expected 1")
    n_new = int(sequences.shape[1] - input_ids.shape[1])
    if n_new != int(action_dim):
        problems.append(f"{n_new} new tokens, expected {action_dim}")
    attn: dict = {}
    if out_attn:
        steps = out.attentions
        if steps is None or len(steps) != int(action_dim):
            problems.append(f"{0 if steps is None else len(steps)} attention steps, expected {action_dim}")
        else:
            prefill = steps[0]
            for layer in layers:
                if layer >= len(prefill):
                    problems.append(f"layer {layer} missing ({len(prefill)} layers)")
                    continue
                t = prefill[layer]
                if t.dim() != 4 or t.shape[-1] != t.shape[-2]:
                    problems.append(f"layer {layer} prefill attention shape {tuple(t.shape)}")
                attn[layer] = t
    if "info" not in holder:
        problems.append("the hook's vision() was not called")
    if problems:
        raise VisionTokenHookError("; ".join(problems))
    tokens = sequences[0, -int(action_dim):].cpu().numpy().tolist()
    shape = {"vision_calls": rec["calls"], "new_tokens": n_new, "output_attentions": out_attn,
             "attention_layers": list(layers),
             "prefill_len": int(next(iter(attn.values())).shape[-1]) if attn else None}
    return tokens, attn, holder["info"], shape
