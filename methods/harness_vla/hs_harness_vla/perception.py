"""Perception for the Harness VLA port: calibration-frame views, back-projection and SAM3 segmentation.

Only public sensor data is used: the benchmark's RGB images, metric depth and camera calibration
(`depth` / `camera` keys from harnessstudio.benchmarks.libero_rgbd), and proprioception (`states`).

Frames. vla-eval publishes images flipped on both axes. RPent's planner works in the *calibration frame*
(robosuite pixel convention = the raw render flipped vertically, i.e. the true camera view; RPent
`agentview.png`). The published image is that frame mirrored horizontally (libero_rgbd docstring,
validated on real frames), so the calibration-frame view is `published[:, ::-1]` and back-projection
uses the robosuite matrices directly with no flip. Pixels are [row, col] as in RPent.

When depth/camera keys are missing (the fake tier-1 stack, or a benchmark without the RGB-D subclass),
every perception tool returns {"error": ...} instead of raising, so the planner can continue.
"""
from __future__ import annotations

import base64
import io
import json
import struct
import urllib.request
import zlib
from typing import Any, Mapping, Optional

import numpy as np

CAMERAS = {"agentview": ("agentview", "agentview"), "wrist": ("wrist", "robot0_eye_in_hand")}


class PerceptionUnavailable(Exception):
    pass


def _views(obs: Mapping[str, Any], camera: str) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """(rgb, depth, cam) in the calibration frame for 'agentview' or 'wrist'."""
    if camera not in CAMERAS:
        raise PerceptionUnavailable(f"unknown camera {camera!r} (use 'agentview' or 'wrist')")
    image_key, depth_key = CAMERAS[camera]
    images = obs.get("images") or {}
    depth = (obs.get("depth") or {}).get(depth_key)
    cam = (obs.get("camera") or {}).get(depth_key)
    if image_key not in images:
        raise PerceptionUnavailable(f"no {camera} image in this observation")
    if depth is None or cam is None:
        raise PerceptionUnavailable(f"no metric depth / calibration for {camera} on this benchmark")
    rgb = np.asarray(images[image_key])
    d = np.asarray(depth, dtype=np.float32)
    flip = cam.get("pixel_flip", "none")
    if flip == "horizontal":
        rgb, d = rgb[:, ::-1], d[:, ::-1]
    elif flip == "both":
        rgb, d = rgb[::-1, ::-1], d[::-1, ::-1]
    elif flip != "none":
        raise PerceptionUnavailable(f"unknown pixel_flip {flip!r}")
    if rgb.shape[:2] != d.shape[:2]:
        raise PerceptionUnavailable(f"{camera} image {rgb.shape[:2]} and depth {d.shape[:2]} differ")
    return np.ascontiguousarray(rgb), np.ascontiguousarray(d), cam


def calibration_image(obs: Mapping[str, Any], camera: str = "agentview") -> Optional[np.ndarray]:
    """The image the planner sees (calibration frame when calibration exists, else as published)."""
    try:
        return _views(obs, camera)[0]
    except PerceptionUnavailable:
        images = obs.get("images") or {}
        key = CAMERAS.get(camera, (camera,))[0]
        return np.asarray(images[key]) if key in images else None


def world_map(depth: np.ndarray, cam: Mapping[str, Any]) -> np.ndarray:
    """[H, W, 3] world points for every calibration-frame pixel (robosuite world_to_pixel inverse)."""
    h, w = depth.shape
    rows, cols = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    pix = np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1)
    inv = np.linalg.inv(np.asarray(cam["world_to_pixel"], dtype=np.float64))
    world = pix @ inv.T
    return (world[..., :3] / world[..., 3:4]).astype(np.float64)


def _valid(points: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (depth > 0.0) & (depth < 10.0)


def back_project(obs: Mapping[str, Any], camera: str = "agentview", row: Optional[int] = None,
                 col: Optional[int] = None, row_range=None, col_range=None, z_min=None, z_max=None,
                 scale: float = 1.0) -> dict:
    """RPent back_project over the published depth: one pixel, or a region midpoint with a z band.
    `scale` converts the planner's displayed-image pixels to the native grid (display / native)."""
    try:
        _, depth, cam = _views(obs, camera)
    except PerceptionUnavailable as exc:
        return {"error": str(exc), "fallback": "perception is unavailable here; plan without world coordinates"}
    wmap = world_map(depth, cam)
    h, w = depth.shape
    if row_range is not None or col_range is not None:
        if row_range is None or col_range is None:
            return {"error": "region mode needs both row_range=[r0,r1] and col_range=[c0,c1]"}
        try:
            r0, r1 = sorted(int(round(float(v) / scale)) for v in row_range)
            c0, c1 = sorted(int(round(float(v) / scale)) for v in col_range)
        except (TypeError, ValueError):
            return {"error": "row_range/col_range must each be [min, max] numbers"}
        r0, r1, c0, c1 = max(0, r0), min(h, r1 + 1), max(0, c0), min(w, c1 + 1)
        if r1 <= r0 or c1 <= c0:
            return {"error": f"empty region after clamping to the {h}x{w} native image"}
        pts = wmap[r0:r1, c0:c1].reshape(-1, 3)
        ok = _valid(wmap[r0:r1, c0:c1], depth[r0:r1, c0:c1]).reshape(-1)
        pts = pts[ok]
        if z_min is not None:
            pts = pts[pts[:, 2] >= float(z_min)]
        if z_max is not None:
            pts = pts[pts[:, 2] <= float(z_max)]
        if len(pts) < 4:
            return {"error": f"too few valid pixels in the region ({len(pts)}); widen it or the z band"}
        return {"camera": camera, "mode": "region",
                "center_xyz": [round(float((pts[:, 0].min() + pts[:, 0].max()) / 2), 4),
                               round(float((pts[:, 1].min() + pts[:, 1].max()) / 2), 4),
                               round(float(np.median(pts[:, 2])), 4)],
                "median_xyz": [round(float(np.median(pts[:, i])), 4) for i in range(3)], "n_valid": int(len(pts))}
    if row is None or col is None:
        return {"error": "give row and col (pixel), or row_range and col_range (region)"}
    r, c = int(round(float(row) / scale)), int(round(float(col) / scale))
    if not (0 <= r < h and 0 <= c < w):
        return {"error": f"pixel ({row},{col}) is outside the displayed image"}
    if not _valid(wmap[r, c][None], depth[r, c][None])[0]:
        return {"error": f"invalid depth at pixel ({row},{col}); pick another pixel"}
    return {"camera": camera, "pixel": [int(row), int(col)], "world_xyz": [round(float(v), 4) for v in wmap[r, c]],
            "depth_m": round(float(depth[r, c]), 4)}


def png_bytes(array: Any) -> bytes:
    a = np.clip(np.asarray(array), 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    color = 2 if a.ndim == 3 else 0
    height, width = a.shape[:2]
    raw = b"".join(b"\x00" + a[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _decode_png_mask(data: bytes, shape) -> np.ndarray:
    """Decode RPent's grayscale PNG mask (PIL if installed, else a small stdlib decoder)."""
    try:
        from PIL import Image
        arr = np.asarray(Image.open(io.BytesIO(data)))
        arr = arr[..., 0] if arr.ndim == 3 else arr
        return (arr > 0).reshape(tuple(shape))
    except ImportError:
        pass
    pos, idat, width, height, color, bitdepth = 8, b"", None, None, None, None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            width, height, bitdepth, color = struct.unpack(">IIBB", body[:10])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    if bitdepth == 1:  # PIL writes mode "1" images as 1-bit
        raw = zlib.decompress(idat)
        stride = (width + 7) // 8
        rows = []
        for r in range(height):
            line = np.frombuffer(raw[r * (stride + 1) + 1:(r + 1) * (stride + 1)], dtype=np.uint8)
            rows.append(np.unpackbits(line)[:width])
        return np.asarray(rows, dtype=bool).reshape(tuple(shape))
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw = zlib.decompress(idat)
    stride = width * channels
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for r in range(height):
        f = raw[r * (stride + 1)]
        line = np.frombuffer(raw[r * (stride + 1) + 1:(r + 1) * (stride + 1)], dtype=np.uint8).astype(np.int32)
        cur = np.zeros(stride, dtype=np.int32)
        for i in range(stride):
            a = cur[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            x = line[i]
            if f == 0:
                v = x
            elif f == 1:
                v = x + a
            elif f == 2:
                v = x + b
            elif f == 3:
                v = x + ((a + b) >> 1)
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                v = x + (a if pa <= pb and pa <= pc else (b if pb <= pc else c))
            cur[i] = v & 0xFF
        out[r] = cur
        prev = cur
    return (out.reshape(height, width, channels)[..., 0] > 0).reshape(tuple(shape))


class Sam3Client:
    """Minimal client for RPent's SAM3 RPC server (POST /call, method sam3.segment)."""

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def segment(self, image: np.ndarray, text_prompt: Optional[str] = None, point=None, min_score: float = 0.2) -> dict:
        kwargs: dict = {"image_base64": base64.b64encode(png_bytes(image)).decode("ascii"), "min_score": float(min_score)}
        if text_prompt:
            kwargs["text_prompt"] = text_prompt
        else:
            kwargs["point"] = [int(point[0]), int(point[1])]
        body = json.dumps({"method": "sam3.segment", "args": [], "kwargs": kwargs, "session_id": None}).encode()
        req = urllib.request.Request(self.url + "/call", data=body, headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            reply = json.loads(resp.read().decode())
        if not reply.get("ok", False):
            raise RuntimeError(f"SAM3 server error: {str(reply.get('error'))[:200]}")
        result = dict(reply.get("result") or {})
        if result.get("found") and result.get("mask_png_base64"):
            result["mask"] = _decode_png_mask(base64.b64decode(result.pop("mask_png_base64")), result["mask_shape"])
        return result


def segment(obs: Mapping[str, Any], client: Optional[Sam3Client], camera: str = "agentview",
            prompt: str = "", point=None, min_score: float = 0.2, scale: float = 1.0) -> tuple[dict, Optional[np.ndarray]]:
    """RPent `segment`: SAM3 on the current calibration-frame image, mask -> median world xyz.
    Returns (result, overlay image or None)."""
    prompt = (prompt or "").strip()
    if bool(prompt) == (point is not None):
        return {"error": "segment needs exactly one of prompt or point"}, None
    try:
        rgb, depth, cam = _views(obs, camera)
    except PerceptionUnavailable as exc:
        return {"error": str(exc), "fallback": "perception is unavailable here; plan without world coordinates"}, None
    if client is None:
        return {"error": "segmentation service not configured", "fallback": "use back_project on pixels"}, None
    native_point = None
    if point is not None:
        try:
            native_point = [int(round(float(point[0]) / scale)), int(round(float(point[1]) / scale))]
        except (TypeError, ValueError, IndexError):
            return {"error": "point must be [row, col]"}, None
    try:
        data = client.segment(rgb, text_prompt=prompt or None, point=native_point, min_score=min_score)
    except Exception as exc:  # noqa: BLE001 - service failure is reported to the planner
        return {"error": f"segmentation service call failed: {type(exc).__name__}: {str(exc)[:160]}",
                "fallback": "use back_project on pixels"}, None
    out: dict = {"camera": camera, "found": bool(data.get("found")),
                 "score": None if data.get("score") is None else round(float(data["score"]), 3)}
    if prompt:
        out["prompt"] = prompt
    else:
        out["point"] = list(point)
    box = data.get("box")
    if box:  # SAM3 box is [x0, y0, x1, y1] in native pixels; report [row0, col0, row1, col1] in display pixels
        out["box_rc"] = [round(box[1] * scale), round(box[0] * scale), round(box[3] * scale), round(box[2] * scale)]
    mask = data.get("mask")
    if not out["found"] or mask is None:
        out["error"] = str(data.get("reason") or "SAM3 found no mask")
        out["fallback"] = "rephrase by colour + shape + relation, lower min_score, or use back_project"
        return out, None
    wmap = world_map(depth, cam)
    ok = mask & _valid(wmap, depth)
    pts = wmap[ok]
    ys, xs = np.nonzero(mask)
    out["n_pixels"] = int(mask.sum())
    out["centroid_rc"] = [round(float(np.median(ys)) * scale), round(float(np.median(xs)) * scale)]
    if len(pts) < 10:
        out["world_xyz"] = None
        out["world_error"] = f"too few valid depth pixels ({len(pts)})"
    else:
        out["world_xyz"] = [round(float(np.median(pts[:, i])), 4) for i in range(3)]
        out["world_x_range"] = [round(float(np.percentile(pts[:, 0], 5)), 4), round(float(np.percentile(pts[:, 0], 95)), 4)]
        out["world_y_range"] = [round(float(np.percentile(pts[:, 1], 5)), 4), round(float(np.percentile(pts[:, 1], 95)), 4)]
        out["world_z_top"] = round(float(np.percentile(pts[:, 2], 95)), 4)
    overlay = rgb.astype(np.float32).copy()
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255.0, 0.0, 0.0])
    return out, overlay.astype(np.uint8)
