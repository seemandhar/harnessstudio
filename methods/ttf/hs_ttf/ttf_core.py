"""TTF-VLA core (arXiv:2508.19257v3), ported from the authors' release.

Source: github.com/PKU-XLab/TTF-VLA @ afbd6bf38fa08b6c5895f58677fe241ce9607100 (MIT, Copyright (c) the TTF-VLA
authors), files ``experiments/robot/aptube_manager.py`` (APTubeManager) and ``prismatic/extern/hf/modeling_prismatic.py``
(PrismaticVisionBackbone.forward, hard-fusion branch; OpenVLAForActionPrediction.predict_action attention capture).

Only the paper's configuration is ported: hard fusion (``smooth_fusion_enabled=False``) with ``fusion_mode`` in
{"pixel", "attention", "hybrid"} ("hybrid" = the paper's pixel-attention TTF, Eq. 4) and the single-layer
text-to-vision attention mode (Eq. 7) or action-to-vision mode (Eq. 8). The semantic and smooth-fusion modes of the
release are not used by the paper's Table 1 and are not ported.

Line references (afbd6bf):
- keyframe rule                         aptube_manager.py:94-95
- grayscale pixel mask (Eq. 5-6)        aptube_manager.py:97-142
- cache update                          aptube_manager.py:208-229 (fused tokens are cached, step counter +1)
- text/action-to-vision merge (Eq. 7-9) aptube_manager.py:231-297 (token positions v 1..256, t 257..291, a 292)
- top-k (VLA-Cache)                     aptube_manager.py:299-330 (cv2.resize to the same 16x16 is an identity;
                                        stable descending sort)
- attention mask                        aptube_manager.py:332-377 (no attention yet -> recompute all; exception ->
                                        recompute all)
- hybrid = pixel OR attention           aptube_manager.py:379-424 (exception -> pixel-only)
- hard fusion                           modeling_prismatic.py:194-244
- attention capture                     modeling_prismatic.py:740-790 (attentions[0] = the prefill's layers)

Behaviour-neutral differences (no numeric change; parity: runs/repro-next-20261105/parity/):
- per-episode state object instead of the release's process-wide singleton (state is reset at episode start in both;
  the release keeps ``last_attention_layers`` across episodes, but step 0 is always a keyframe whose own prefill
  overwrites it before any use);
- only the configured attention layer is kept (the release keeps all 32 layers and reads one);
- no prints, timers, GFLOP or visualisation bookkeeping; fallbacks are counted instead of printed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .constants import (A_TOKEN_START, DINO_DIM, GRAY, MULTI_LAYER_IDS, N_PATCHES, PAPER,  # noqa: F401
                        SIGLIP_DIM, T_TOKEN_END, T_TOKEN_START, V_TOKEN_END, V_TOKEN_START)


@dataclass
class TTFState:
    """Per-episode TTF state (APTubeManager fields reset by reset_state + the attention it carries)."""
    step_counter: int = 0
    last_pixel_values: Optional[torch.Tensor] = None
    last_vision_tokens: Optional[torch.Tensor] = None
    last_attention_layers: Optional[dict] = None   # {layer_id: [1, heads, seq, seq]} of the previous prefill
    counters: dict = field(default_factory=lambda: {"keyframes": 0, "fused": 0, "hybrid_fallback": 0,
                                                    "attention_fallback": 0, "no_attention": 0})


def is_keyframe(state: TTFState, keyframe_interval: int) -> bool:
    """aptube_manager.py:94-95."""
    return state.step_counter == 0 or state.last_pixel_values is None or state.step_counter % keyframe_interval == 0


def _mask_one_part(prev_img_part: torch.Tensor, curr_img_part: torch.Tensor, threshold: float):
    """aptube_manager.py:106-128 verbatim (Eq. 5-6 and m_pixel = 1[d > tau])."""
    img_prev = prev_img_part.to(torch.float32)
    img_curr = curr_img_part.to(torch.float32)
    weights = torch.tensor(list(GRAY), device=img_curr.device).view(1, 3, 1, 1)
    gray_prev = torch.sum(img_prev * weights, dim=1, keepdim=True)
    gray_curr = torch.sum(img_curr * weights, dim=1, keepdim=True)
    diff = torch.abs(gray_prev - gray_curr)
    patch_avg_diff = F.avg_pool2d(diff, kernel_size=14, stride=14)
    diff_values = patch_avg_diff.flatten()
    mask = diff_values > threshold
    return mask, diff_values


def pixel_recompute_mask(state: TTFState, pixel_values: torch.Tensor, keyframe_interval: int,
                         threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """aptube_manager.py:97-142 (DINO = channels 0-2, SigLIP = channels 3-5)."""
    if is_keyframe(state, keyframe_interval):
        ones = torch.ones(N_PATCHES, dtype=torch.bool, device=pixel_values.device)
        return ones, ones.clone()
    m_dino, _ = _mask_one_part(state.last_pixel_values[:, :3, :, :], pixel_values[:, :3, :, :], threshold)
    m_siglip, _ = _mask_one_part(state.last_pixel_values[:, 3:, :, :], pixel_values[:, 3:, :, :], threshold)
    return m_dino, m_siglip


def token_attention_merge(layers: dict, layer_ids, attention_mode: str) -> torch.Tensor:
    """aptube_manager.py:231-297: per-patch relevance [256] from the stored attentions (float32, on CPU)."""
    aggregated = None
    n_layers = len(layer_ids)
    for layer_id in layer_ids:
        if layer_id not in layers:
            continue
        attn_map = layers[layer_id].to(torch.float32).squeeze(0).mean(dim=0)
        layer_scores = torch.zeros(N_PATCHES, device=attn_map.device)
        if attention_mode == "text":
            if T_TOKEN_START < attn_map.shape[0]:
                seq_len = attn_map.shape[0]
                actual_t_end = min(T_TOKEN_END, seq_len)
                if actual_t_end > T_TOKEN_START:
                    layer_scores = attn_map[T_TOKEN_START:actual_t_end, V_TOKEN_START:V_TOKEN_END].mean(dim=0)
        elif attention_mode == "action":
            if A_TOKEN_START < attn_map.shape[0]:
                layer_scores = attn_map[A_TOKEN_START, V_TOKEN_START:V_TOKEN_END]
        aggregated = layer_scores if aggregated is None else aggregated + layer_scores
    if n_layers > 1:
        aggregated = aggregated / n_layers
    return aggregated.cpu()


def top_attention_patches(scores: torch.Tensor, top_k: int) -> list:
    """aptube_manager.py:299-330: 16x16 grid, (cv2.resize to 16x16 = identity), stable descending sort, first k."""
    attn = scores.cpu().numpy().reshape(16, 16)
    flat = [(i * 16 + j, attn[i, j]) for i in range(16) for j in range(16)]
    flat.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in flat[:top_k]]


def attention_recompute_mask(state: TTFState, device, layer_ids, attention_mode: str, top_k: int):
    """aptube_manager.py:332-377."""
    if state.last_attention_layers is None:
        state.counters["no_attention"] += 1
        ones = torch.ones(N_PATCHES, dtype=torch.bool, device=device)
        return ones, ones.clone()
    try:
        relevance = token_attention_merge(state.last_attention_layers, layer_ids, attention_mode)
        important = top_attention_patches(relevance, top_k)
        mask = torch.zeros(N_PATCHES, dtype=torch.bool, device=device)
        for idx in important:
            mask[idx] = True
        return mask, mask.clone()
    except Exception:  # the release prints a traceback and recomputes every patch
        state.counters["attention_fallback"] += 1
        ones = torch.ones(N_PATCHES, dtype=torch.bool, device=device)
        return ones, ones.clone()


def recompute_masks(state: TTFState, pixel_values: torch.Tensor, cfg: dict):
    """The non-keyframe mask of the configured fusion mode (modeling_prismatic.py:206-217)."""
    mode = cfg["fusion_mode"]
    layer_ids = list(MULTI_LAYER_IDS) if cfg["use_multi_layer"] else [cfg["attention_layer_id"]]
    if mode == "pixel":
        return pixel_recompute_mask(state, pixel_values, cfg["keyframe_interval"], cfg["patch_diff_threshold"])
    if mode == "attention":
        return attention_recompute_mask(state, pixel_values.device, layer_ids, cfg["attention_mode"],
                                        cfg["attention_top_k"])
    if mode == "hybrid":
        try:
            p_dino, p_siglip = pixel_recompute_mask(state, pixel_values, cfg["keyframe_interval"],
                                                    cfg["patch_diff_threshold"])
            a_dino, a_siglip = attention_recompute_mask(state, pixel_values.device, layer_ids,
                                                        cfg["attention_mode"], cfg["attention_top_k"])
            return p_dino | a_dino, p_siglip | a_siglip
        except Exception:  # aptube_manager.py:420-424: fall back to pixel-only
            state.counters["hybrid_fallback"] += 1
            return pixel_recompute_mask(state, pixel_values, cfg["keyframe_interval"], cfg["patch_diff_threshold"])
    raise ValueError(f"Invalid fusion mode {mode!r}; please use 'pixel', 'attention', or 'hybrid'")


def fuse(state: TTFState, pixel_values: torch.Tensor, new_dino: torch.Tensor, new_siglip: torch.Tensor,
         cfg: dict) -> Tuple[torch.Tensor, dict]:
    """One vision forward of the enabled release (modeling_prismatic.py:194-244 + update_cache).

    ``new_dino`` [1, 256, 1024] and ``new_siglip`` [1, 256, 1152] are the featurizers' outputs on the current frame.
    Returns the patches [1, 256, 2176] for the projector and an info dict. Updates ``state`` like update_cache.
    """
    info = {"step": state.step_counter}
    if is_keyframe(state, cfg["keyframe_interval"]):
        patches = torch.cat([new_dino, new_siglip], dim=2)
        state.counters["keyframes"] += 1
        info.update(keyframe=True, reused_dino=0, reused_siglip=0)
    else:
        m_dino, m_siglip = recompute_masks(state, pixel_values, cfg)
        cached_dino, cached_siglip = torch.split(state.last_vision_tokens, [DINO_DIM, SIGLIP_DIM], dim=2)
        final_dino = cached_dino.clone()
        final_siglip = cached_siglip.clone()
        final_dino[0, m_dino] = new_dino[0, m_dino]
        final_siglip[0, m_siglip] = new_siglip[0, m_siglip]
        patches = torch.cat([final_dino, final_siglip], dim=2)
        state.counters["fused"] += 1
        info.update(keyframe=False, reused_dino=int((~m_dino).sum().item()),
                    reused_siglip=int((~m_siglip).sum().item()))
    # update_cache (aptube_manager.py:208-229; identical for pixel / attention / hybrid)
    state.last_pixel_values = pixel_values.detach().clone()
    state.last_vision_tokens = patches.detach().clone()
    state.step_counter += 1
    return patches, info


def store_attention(state: TTFState, prefill_layers: dict) -> None:
    """modeling_prismatic.py:788: ``manager.last_attention_layers = attentions[0]`` (the prefill's layers)."""
    state.last_attention_layers = {int(k): v.detach() for k, v in prefill_layers.items()}
