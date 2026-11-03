"""Mask post-processing and counterfactual composition (re-derived, numpy/OpenCV only).

Upstream references (Koorye/PCD @ 31b027e):
- contrast_utils/mask_predictors.py:71-89          detection-frame post-processing: None for empty; 4-connected
                                                   components; drop components with area < 0.1 x largest; external
                                                   contours filled (holes closed) -> bool
- contrast_utils/utils.py:41-45                    dilation: cv2.dilate, ones((k, k), uint8), one iteration
- contrast_utils/contrast_image_generator.py:177-190 union of target masks, then keep-object (robot) pixels cleared
- contrast_utils/inpainters.py:20                  dilation size 5
- contrast_utils/inpainters.py:24-33               inpaint region = dilate(mask | robot); paste region =
                                                   dilate(mask) & ~robot (robot undilated); image copy elsewhere
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

DILATE_SIZE = 5


def postprocess_detection_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None or not mask.any():
        return None
    work = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=4)
    biggest = stats[1:, cv2.CC_STAT_AREA].max()
    if count > 1:
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 0.1 * biggest:
                work[labels == label] = 0
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(work)
    cv2.fillPoly(filled, contours, 1)
    return filled > 0


def dilate(mask: Optional[np.ndarray], size: int = DILATE_SIZE) -> Optional[np.ndarray]:
    if mask is None:
        return None
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8), iterations=1).astype(bool)


def target_union(shape: Tuple[int, int], target_masks: Sequence[Optional[np.ndarray]],
                 keep_masks: Sequence[Optional[np.ndarray]]) -> np.ndarray:
    """contrast_image_generator.py:177-190 with reverse_mask=False."""
    union = np.zeros(shape, dtype=bool)
    for m in target_masks:
        if m is not None:
            union |= m
    for m in keep_masks:
        if m is not None:
            union[m] = False
    return union


def inpaint_regions(mask: np.ndarray, robot: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(region given to the inpainter, region pasted back) -- inpainters.py:26-31."""
    inpaint_region = dilate(mask | robot if robot is not None else mask)
    paste_region = dilate(mask) & ~robot if robot is not None else dilate(mask)
    return inpaint_region, paste_region


def compose(image: np.ndarray, inpainted: np.ndarray, paste_region: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[paste_region] = inpainted[paste_region]
    return out


def tile_vertical(images: List[np.ndarray]) -> np.ndarray:
    """utils.py tile_images for two frames: ceil(sqrt(2)) = 2 rows x 1 column (parallel_inference.py:199,224)."""
    rows = int(np.ceil(np.sqrt(len(images))))
    cols = int(np.ceil(len(images) / rows))
    h, w, c = images[0].shape
    out = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, img in enumerate(images):
        r, k = i // cols, i % cols
        out[r * h:(r + 1) * h, k * w:(k + 1) * w] = img
    return out
