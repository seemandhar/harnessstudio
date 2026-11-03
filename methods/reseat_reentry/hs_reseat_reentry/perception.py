"""Public RGB-D perception for reseat-reentry: retained packets, back-projection, marked crops, and the hover
target (back-projected point, local object top in a small disk, table plane).

Vendored from the witnessed-rescue method package (methods/witnessed_rescue, perception module after fix2b, sha256
896ad9db...; its frame helpers `views`/`world_map` come from the Harness VLA port in methods/harness_vla, pinned there).
Inputs: published RGB, metric depth and camera calibration (libero_rgbd `depth`/`camera`, harness-only added public
sensors; the frozen policy never sees them) and proprio `states`. No object poses, no simulator state.
Frames: vla-eval publishes images flipped on both axes; robosuite's matrices use the calibration frame (= the true
camera view) = the published image mirrored horizontally (`pixel_flip: horizontal`). Pixels are [row, col] in the
calibration frame; the planner sees calibration-frame images (2x).
New here: hover_target (review r2 merged_spec.response); the r1 table-height 'background' rejection is NOT applied to
hover targets (the witness decides semantics; review: 'not a reject criterion here').
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

CAMERAS = {"agentview": ("agentview", "agentview"), "wrist": ("wrist", "robot0_eye_in_hand")}

DEFAULTS = {
    "image_scale": 2,              # planner images are upsampled 2x (256 -> 512); pixels are given in that frame
    "table_bin": 0.005,            # m, histogram bin of the table-height mode
    "table_margin": 0.01,          # reject points below z_table + 1 cm (background / table)
    "robot_reject_px": 10,         # native px around the projected end effector
    "robot_reject_m": 0.045,       # m: points this close to the eef (xy) and above eef_z - 0.02 are gripper
    "workspace_radius": 0.5,       # m from the current eef (xy)
    "workspace_height": 0.35,      # m above the table
    "min_height": 0.015,           # object extent checks (m)
    "max_height": 0.30,
    "max_xy": 0.25,
    "grow_dz": 0.015,              # region growing: neighbour depth continuity (m)
    "grow_radius": 0.15,           # region growing: max xy distance from the pointed point (m)
    "object_floor": 0.008,         # region growing: points must be this far above the table (m)
    "crop_half": 40,               # native px half-size of the marked crop
    "column_radius": 0.04,         # place free-column check
    "column_exclude": 0.06,        # gripper + carried object volume: xy radius around the eef ...
    "column_exclude_below": 0.02,  # ... from (eef z - h_obj - this) upward (points below it stay observed)
    "column_max_points": 6,
    "column_min_points": 20,       # fewer observed column points -> not free ('column_unobserved')
    "surface_radius": 0.02,
    # r2 hover target (frozen scalars, review merged_spec.frozen_scalars)
    "transit_above_table": 0.25,   # safe transit z = max(current eef z, table + 0.25 m)
    "hover_above_top": 0.10,       # hover z = local object top + 0.10 m
    "top_disk": 0.04,              # local top = max depth-z within this xy radius of the back-projected point
    "top_floor_above_table": 0.05, # the local top is floored at table + 0.05 m
}


class PerceptionUnavailable(Exception):
    pass


def sha(array: Any) -> str:
    a = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def views(obs: Mapping[str, Any], camera: str) -> tuple:
    """(rgb, depth, cam) in the calibration frame (vendored harness_vla `_views`)."""
    if camera not in CAMERAS:
        raise PerceptionUnavailable(f"unknown camera {camera!r}")
    image_key, depth_key = CAMERAS[camera]
    images = obs.get("images") or {}
    depth = (obs.get("depth") or {}).get(depth_key)
    cam = (obs.get("camera") or {}).get(depth_key)
    if image_key not in images:
        raise PerceptionUnavailable(f"no {camera} image in this observation")
    if depth is None or cam is None:
        raise PerceptionUnavailable(f"no metric depth / calibration for {camera}")
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


def rgb_view(obs: Mapping[str, Any], camera: str) -> Optional[np.ndarray]:
    """Calibration-frame RGB when calibration exists (else as published; None when absent)."""
    try:
        return views(obs, camera)[0]
    except PerceptionUnavailable:
        images = obs.get("images") or {}
        key = CAMERAS.get(camera, (camera,))[0]
        if key not in images:
            return None
        rgb = np.asarray(images[key])
        cam = (obs.get("camera") or {}).get(CAMERAS.get(camera, (camera, camera))[1]) or {}
        if cam.get("pixel_flip") == "horizontal":
            rgb = rgb[:, ::-1]
        return np.ascontiguousarray(rgb)


def world_map(depth: np.ndarray, cam: Mapping[str, Any]) -> np.ndarray:
    """[H, W, 3] world points for every calibration-frame pixel (vendored harness_vla `world_map`)."""
    h, w = depth.shape
    rows, cols = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    pix = np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1)
    inv = np.linalg.inv(np.asarray(cam["world_to_pixel"], dtype=np.float64))
    world = pix @ inv.T
    return (world[..., :3] / world[..., 3:4]).astype(np.float64)


def valid_mask(points: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (depth > 0.0) & (depth < 10.0)


def project(xyz, cam: Mapping[str, Any]) -> tuple:
    """World point -> (row, col) in the calibration frame."""
    p = np.asarray(cam["world_to_pixel"], dtype=np.float64) @ np.append(np.asarray(xyz, dtype=np.float64), 1.0)
    return float(p[1] / p[2]), float(p[0] / p[2])


def table_height(wmap: np.ndarray, ok: np.ndarray, ee: np.ndarray, p: Mapping[str, Any]) -> Optional[float]:
    """Dominant horizontal plane below the end effector: the mode of world z (5 mm bins) over valid pixels within
    the workspace radius. A geometric statistic of the current frame, not a fitted model."""
    pts = wmap[ok]
    if len(pts) < 50:
        return None
    near = np.linalg.norm(pts[:, :2] - ee[:2], axis=1) <= p["workspace_radius"]
    z = pts[near, 2]
    z = z[z < ee[2] + 0.02]
    if len(z) < 50:
        return None
    b = float(p["table_bin"])
    lo = np.floor(z.min() / b) * b
    counts = np.bincount(((z - lo) / b).astype(np.int64))
    k = int(np.argmax(counts))
    sel = z[((z - lo) / b).astype(np.int64) == k]
    return float(np.median(sel))


def min_area_rect(xy: np.ndarray) -> tuple:
    """(center xy, (long, short) extents, angle of the SHORT axis in rad) by a 1-degree sweep (rotating calipers
    equivalent for these small point sets; deterministic, no fitting)."""
    best = None
    for deg in range(0, 90):
        a = np.deg2rad(deg)
        u = np.array([np.cos(a), np.sin(a)])
        v = np.array([-np.sin(a), np.cos(a)])
        pu, pv = xy @ u, xy @ v
        eu, ev = float(pu.max() - pu.min()), float(pv.max() - pv.min())
        area = eu * ev
        if best is None or area < best[0] - 1e-12:
            cu, cv = (pu.max() + pu.min()) / 2, (pv.max() + pv.min()) / 2
            center = cu * u + cv * v
            short_angle = a if eu <= ev else a + np.pi / 2
            best = (area, center, (max(eu, ev), min(eu, ev)), float(short_angle))
    return best[1], best[2], best[3]


@dataclass
class Packet:
    """A retained public packet: the exact images/depth/calibration/proprio of one decision step."""
    step: int
    obs: dict
    hashes: dict = field(default_factory=dict)

    @classmethod
    def capture(cls, obs: Mapping[str, Any], step: int) -> "Packet":
        hashes = {}
        for k, v in (obs.get("images") or {}).items():
            hashes[f"images/{k}"] = {"sha256": sha(v), "shape": list(np.asarray(v).shape)}
        for k, v in (obs.get("depth") or {}).items():
            hashes[f"depth/{k}"] = {"sha256": sha(v), "shape": list(np.asarray(v).shape)}
        st = obs.get("states", obs.get("state"))
        if st is not None:
            hashes["states"] = {"sha256": sha(np.asarray(st, dtype=np.float64)), "value":
                                [round(float(x), 5) for x in np.asarray(st, dtype=np.float64).reshape(-1)]}
        return cls(int(step), dict(obs), hashes)

    def state(self) -> Optional[np.ndarray]:
        st = self.obs.get("states", self.obs.get("state"))
        return None if st is None else np.asarray(st, dtype=np.float64).reshape(-1)


def packet_table_z(packet: "Packet", params: Optional[Mapping[str, Any]] = None) -> Optional[float]:
    """Table-plane height of a packet's agentview depth (None without depth/calibration/proprio)."""
    p = dict(DEFAULTS)
    p.update(params or {})
    try:
        _, depth, cam = views(packet.obs, "agentview")
    except PerceptionUnavailable:
        return None
    st = packet.state()
    if st is None or len(st) < 3:
        return None
    wmap = world_map(depth, cam)
    return table_height(wmap, valid_mask(wmap, depth), st[:3], p)


def _grow(wmap, ok, depth, seed, z_floor, p, exclude) -> np.ndarray:
    h, w = depth.shape
    r0, c0 = seed
    origin = wmap[r0, c0]
    mask = np.zeros((h, w), dtype=bool)
    allowed = ok & (wmap[..., 2] > z_floor) & ~exclude
    allowed &= np.linalg.norm(wmap[..., :2] - origin[:2], axis=-1) <= p["grow_radius"]
    if not allowed[r0, c0]:
        return mask
    q = deque([(r0, c0)])
    mask[r0, c0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc] and allowed[rr, cc] \
                    and abs(float(depth[rr, cc]) - float(depth[r, c])) < p["grow_dz"]:
                mask[rr, cc] = True
                q.append((rr, cc))
    return mask


def _gripper_exclusion(wmap, ee, p) -> np.ndarray:
    return (np.linalg.norm(wmap[..., :2] - ee[:2], axis=-1) < p["robot_reject_m"]) & (wmap[..., 2] > ee[2] - 0.02)


def locate(packet: Packet, row: float, col: float, params: Mapping[str, Any], role: str = "target") -> dict:
    """Back-project the planner's pixel (display frame) on the packet's agentview depth and run the geometric
    witness checks. role="target": an object to grasp (all checks); role="destination": a support (no object
    extent check, robot exclusion still applies)."""
    p = dict(DEFAULTS)
    p.update(params or {})
    out: dict = {"role": role, "pixel_display": [float(row), float(col)], "packet_step": packet.step}
    try:
        rgb, depth, cam = views(packet.obs, "agentview")
    except PerceptionUnavailable as exc:
        return {**out, "ok": False, "reason": f"perception_unavailable: {exc}"}
    st = packet.state()
    if st is None or len(st) < 3:
        return {**out, "ok": False, "reason": "no proprio"}
    ee = st[:3]
    scale = float(p["image_scale"])
    h, w = depth.shape
    r, c = int(round(float(row) / scale)), int(round(float(col) / scale))
    out["pixel_native"] = [r, c]
    if not (0 <= r < h and 0 <= c < w):
        return {**out, "ok": False, "reason": "pixel_outside_image"}
    wmap = world_map(depth, cam)
    ok = valid_mask(wmap, depth)
    if not ok[r, c]:
        return {**out, "ok": False, "reason": "invalid_depth"}
    pt = wmap[r, c]
    out["xyz"] = [round(float(v), 4) for v in pt]
    z_table = table_height(wmap, ok, ee, p)
    if z_table is None:
        return {**out, "ok": False, "reason": "no_table_plane"}
    out["z_table"] = round(z_table, 4)
    er, ec = project(ee, cam)
    out["ee_pixel_native"] = [round(er, 1), round(ec, 1)]
    out["ee_pixel_distance"] = round(float(np.hypot(er - r, ec - c)), 1)
    # Robot/gripper pixels are rejected by WHAT THE DEPTH SEES at the pixel: the back-projected point inside the
    # gripper volume (xy within robot_reject_m of the eef and above eef z - 2 cm). A pure pixel-distance rule
    # around the projected eef would reject every target the gripper hovers above (after a drop or a too-high
    # empty close), so pixel proximity is recorded but only decides together with the 3D test.
    excl = _gripper_exclusion(wmap, ee, p)
    if excl[r, c]:
        near = out["ee_pixel_distance"] < p["robot_reject_px"]
        return {**out, "ok": False, "reason": "robot_pixel (gripper volume" + (", at the projected eef)" if near else ")")}
    if np.linalg.norm(pt[:2] - ee[:2]) > p["workspace_radius"] or pt[2] > z_table + p["workspace_height"]:
        return {**out, "ok": False, "reason": "outside_workspace"}
    if role == "destination":
        near = ok & (np.linalg.norm(wmap[..., :2] - pt[:2], axis=-1) <= p["surface_radius"]) & ~excl
        zs = wmap[near][:, 2]
        z_surface = float(np.median(zs)) if len(zs) >= 4 else float(pt[2])
        out.update(ok=True, reason="ok", z_surface=round(z_surface, 4), center_xyz=out["xyz"])
        return out
    if pt[2] < z_table + p["table_margin"]:
        return {**out, "ok": False, "reason": "background (at or below the table plane)"}
    mask = _grow(wmap, ok, depth, (r, c), z_table + p["object_floor"], p, excl)
    n = int(mask.sum())
    out["n_pixels"] = n
    if n < 8:
        return {**out, "ok": False, "reason": "too_few_object_pixels"}
    pts = wmap[mask]
    z_top = float(np.percentile(pts[:, 2], 95))
    height = z_top - z_table
    center, (long_, short), yaw_short = min_area_rect(pts[:, :2])
    out.update(z_top=round(z_top, 4), height=round(height, 4), extent_xy=[round(long_, 4), round(short, 4)],
               yaw_short=round(yaw_short, 4), center_xyz=[round(float(center[0]), 4), round(float(center[1]), 4),
                                                           round(z_top, 4)])
    if not (p["min_height"] <= height <= p["max_height"]):
        return {**out, "ok": False, "reason": f"object height {height:.3f} m outside [{p['min_height']}, {p['max_height']}]"}
    if long_ > p["max_xy"]:
        return {**out, "ok": False, "reason": f"object xy extent {long_:.3f} m > {p['max_xy']}"}
    out.update(ok=True, reason="ok")
    out["_mask"] = mask
    return out


def free_column(packet: Packet, dest: Mapping[str, Any], h_obj: float, ee, params: Mapping[str, Any]) -> dict:
    """Is the depth column above the destination support free? Points within column_radius (xy) of the
    destination, between z_surface + 2 cm and the carry height, excluding only the 3D gripper + carried-object
    volume (xy within column_exclude of the eef AND above eef z - h_obj - column_exclude_below). Never vacuous:
    with fewer than column_min_points observed column points the column counts as not free ('column_unobserved')."""
    p = dict(DEFAULTS)
    p.update(params or {})
    try:
        _, depth, cam = views(packet.obs, "agentview")
    except PerceptionUnavailable as exc:
        return {"free": False, "reason": f"perception_unavailable: {exc}"}
    wmap = world_map(depth, cam)
    ok = valid_mask(wmap, depth)
    d = np.asarray(dest["center_xyz"], dtype=np.float64)
    zs = float(dest["z_surface"])
    ee = np.asarray(ee, dtype=np.float64)
    z = wmap[..., 2]
    near = ok & (np.linalg.norm(wmap[..., :2] - d[:2], axis=-1) <= p["column_radius"])
    carried = (np.linalg.norm(wmap[..., :2] - ee[:2], axis=-1) < p["column_exclude"]) \
        & (z > ee[2] - float(h_obj) - p["column_exclude_below"])
    near &= ~carried
    occ = near & (z > zs + 0.02) & (z < zs + h_obj + 0.08)
    n, m = int(occ.sum()), int(near.sum())
    if m < p["column_min_points"]:
        return {"free": False, "occupied_points": n, "column_points": m, "reason": "column_unobserved"}
    return {"free": n <= p["column_max_points"], "occupied_points": n, "column_points": m,
            "reason": "ok" if n <= p["column_max_points"] else "occupied"}


# -- images for the planner -------------------------------------------------------------------------------------
def upsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def draw_marker(img: np.ndarray, row: float, col: float, radius: int = 7) -> np.ndarray:
    """A red ring + cross centred on (row, col) of an RGB uint8 image (copy)."""
    out = np.array(img, dtype=np.uint8, copy=True)
    h, w = out.shape[:2]
    rr, cc = np.mgrid[0:h, 0:w]
    dist = np.hypot(rr - row, cc - col)
    ring = (dist >= radius - 1.2) & (dist <= radius + 1.2)
    cross = ((np.abs(rr - row) <= 0.6) | (np.abs(cc - col) <= 0.6)) & (dist <= radius + 4) & (dist >= 2.5)
    out[ring | cross] = (255, 0, 0)
    return out


def zoom_view(packet: Packet, row_native: float, col_native: float, params: Mapping[str, Any],
              camera: str = "agentview") -> np.ndarray:
    """Unmarked 4x zoom of a fixed crop window (crop_half native px around the pixel) of one packet. Two packets
    with the same pixel give the same window, so before/after crops show what moved."""
    p = dict(DEFAULTS)
    p.update(params or {})
    rgb = rgb_view(packet.obs, camera)
    if rgb is None:
        raise PerceptionUnavailable(f"no {camera} image")
    k = int(p["crop_half"])
    h, w = rgb.shape[:2]
    r0 = int(np.clip(round(row_native) - k, 0, max(0, h - 2 * k)))
    c0 = int(np.clip(round(col_native) - k, 0, max(0, w - 2 * k)))
    return upsample(np.ascontiguousarray(rgb[r0:r0 + 2 * k, c0:c0 + 2 * k]), 4)


def marked_views(packet: Packet, row_native: float, col_native: float, params: Mapping[str, Any],
                 camera: str = "agentview") -> tuple:
    """(full marked image, zoomed marked crop) for the same-frame witness, both in display pixels."""
    p = dict(DEFAULTS)
    p.update(params or {})
    rgb = rgb_view(packet.obs, camera)
    if rgb is None:
        raise PerceptionUnavailable(f"no {camera} image")
    s = int(p["image_scale"])
    full = draw_marker(upsample(rgb, s), row_native * s, col_native * s, radius=7 * s // 2 + 3)
    k = int(p["crop_half"])
    h, w = rgb.shape[:2]
    r0 = int(np.clip(round(row_native) - k, 0, max(0, h - 2 * k)))
    c0 = int(np.clip(round(col_native) - k, 0, max(0, w - 2 * k)))
    crop = rgb[r0:r0 + 2 * k, c0:c0 + 2 * k]
    z = 4
    crop = draw_marker(upsample(crop, z), (row_native - r0) * z, (col_native - c0) * z, radius=14)
    return full, crop


def hover_target(packet: Packet, row: float, col: float, params: Mapping[str, Any]) -> dict:
    """Back-project the planner's pixel (display frame) and compute the open-gripper re-seat target:
    hover = (x, y, local top + hover_above_top), local top = max valid z within top_disk (xy) of the point, outside the
    gripper volume, floored at table + top_floor_above_table; clipped to table + workspace_height; transit z =
    max(current eef z, table + transit_above_table). Geometric rejections: outside image, invalid depth, gripper
    volume, outside the workspace (radius around the eef). Also returns a size cue (height above the table and, when
    region growing finds the object, its footprint) for the witness prompt."""
    p = dict(DEFAULTS)
    p.update(params or {})
    out: dict = {"pixel_display": [float(row), float(col)], "packet_step": packet.step}
    try:
        rgb, depth, cam = views(packet.obs, "agentview")
    except PerceptionUnavailable as exc:
        return {**out, "ok": False, "reason": f"perception_unavailable: {exc}"}
    st = packet.state()
    if st is None or len(st) < 3:
        return {**out, "ok": False, "reason": "no proprio"}
    ee = st[:3]
    scale = float(p["image_scale"])
    h, w = depth.shape
    r, c = int(round(float(row) / scale)), int(round(float(col) / scale))
    out["pixel_native"] = [r, c]
    if not (0 <= r < h and 0 <= c < w):
        return {**out, "ok": False, "reason": "pixel_outside_image"}
    wmap = world_map(depth, cam)
    ok = valid_mask(wmap, depth)
    if not ok[r, c]:
        return {**out, "ok": False, "reason": "invalid_depth"}
    pt = wmap[r, c]
    out["xyz"] = [round(float(v), 4) for v in pt]
    z_table = table_height(wmap, ok, ee, p)
    if z_table is None:
        return {**out, "ok": False, "reason": "no_table_plane"}
    out["z_table"] = round(z_table, 4)
    excl = _gripper_exclusion(wmap, ee, p)
    if excl[r, c]:
        return {**out, "ok": False, "reason": "robot_pixel (gripper volume)"}
    if np.linalg.norm(pt[:2] - ee[:2]) > p["workspace_radius"] or pt[2] > z_table + p["workspace_height"]:
        return {**out, "ok": False, "reason": "outside_workspace"}
    disk = ok & ~excl & (np.linalg.norm(wmap[..., :2] - pt[:2], axis=-1) <= p["top_disk"])
    zs = wmap[disk][:, 2]
    top = float(zs.max()) if len(zs) else float(pt[2])
    top = max(top, z_table + p["top_floor_above_table"])
    hover_z = min(top + p["hover_above_top"], z_table + p["workspace_height"])
    transit_z = max(float(ee[2]), z_table + p["transit_above_table"])
    out.update(ok=True, reason="ok", local_top=round(top, 4), disk_points=int(len(zs)),
               hover_xyz=[round(float(pt[0]), 4), round(float(pt[1]), 4), round(hover_z, 4)],
               transit_z=round(transit_z, 4), height=round(top - z_table, 4), extent_xy=None)
    mask = _grow(wmap, ok, depth, (r, c), z_table + p["object_floor"], p, excl)
    if int(mask.sum()) >= 8:
        _, (long_, short), _ = min_area_rect(wmap[mask][:, :2])
        if long_ <= p["max_xy"]:
            out["extent_xy"] = [round(long_, 4), round(short, 4)]
    return out
