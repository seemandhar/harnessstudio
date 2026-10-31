"""Pixel <-> world geometry for the RGB-D benchmark subclasses (harnessstudio.benchmarks.libero_rgbd)."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _unflip(u: float, v: float, cam: Mapping) -> tuple[float, float]:
    if cam.get("flipped"):
        return cam["width"] - 1 - u, cam["height"] - 1 - v
    return u, v


def world_to_pixel(xyz, cam: Mapping) -> tuple[float, float]:
    """World point -> (u, v) in the (possibly flipped) published image."""
    p = np.asarray(cam["world_to_pixel"], dtype=np.float64) @ np.append(np.asarray(xyz, dtype=np.float64), 1.0)
    # robosuite convention: p = [col * z, row * z, z]; image index = [row, col]
    col, row = p[0] / p[2], p[1] / p[2]
    u, v = _unflip(col, row, cam)
    return float(u), float(v)


def pixel_to_world(u: float, v: float, depth_image, cam: Mapping) -> np.ndarray:
    """(u, v) in the published image + its depth map -> world point."""
    h, w = depth_image.shape[:2]
    ui, vi = int(round(np.clip(u, 0, w - 1))), int(round(np.clip(v, 0, h - 1)))
    z = float(depth_image[vi, ui])
    col, row = _unflip(ui, vi, cam)
    pix = np.array([col * z, row * z, z, 1.0])
    world = np.linalg.inv(np.asarray(cam["world_to_pixel"], dtype=np.float64)) @ pix
    return world[:3] / world[3]
