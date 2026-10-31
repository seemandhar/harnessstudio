"""Sanity check of the calibration-frame geometry on a dumped real observation (HS_HARNESS_VLA_DEBUG_DIR).

Projects the end-effector position (public proprio) into the calibration-frame agentview image and back-projects
the depth there, and reports the image direction of +x / +y. Usage: python check_geometry.py OBS.npz [OUT.png]
"""
from __future__ import annotations

import sys

import numpy as np

from hs_harness_vla import perception as P


def main(path: str, out_png: str = "") -> None:
    z = np.load(path)
    obs = {"images": {}, "depth": {}, "camera": {}}
    for k in z.files:
        parts = k.split(".")
        if parts[0] in ("images", "depth"):
            obs[parts[0]][parts[1]] = z[k]
        elif parts[0] == "camera":
            obs["camera"].setdefault(parts[1], {})[parts[2]] = z[k].item() if z[k].ndim == 0 else z[k]
    states = z["states"]
    print("instruction:", str(z["instruction"]))
    print("eef:", states[:3])
    rgb, depth, cam = P._views(obs, "agentview")  # noqa: SLF001
    M = np.asarray(cam["world_to_pixel"], dtype=np.float64)

    def proj(p):
        v = M @ np.append(np.asarray(p, dtype=np.float64), 1.0)
        return v[1] / v[2], v[0] / v[2]   # (row, col) in the calibration frame
    r, c = proj(states[:3])
    print(f"eef projects to (row, col) = ({r:.1f}, {c:.1f}) in the {rgb.shape[:2]} calibration frame")
    for d, name in (([0.1, 0, 0], "+x"), ([0, 0.1, 0], "+y"), ([0, 0, 0.1], "+z")):
        r2, c2 = proj(states[:3] + np.asarray(d))
        print(f"{name} 10 cm moves the pixel by (drow, dcol) = ({r2 - r:+.1f}, {c2 - c:+.1f})")
    bp = P.back_project(obs, "agentview", row=int(round(r)), col=int(round(c)))
    print("back_project at the eef pixel:", bp)
    wmap = P.world_map(depth, cam)
    print("table-ish median z (bottom quarter of the image):", float(np.median(wmap[int(0.75 * depth.shape[0]):, :, 2])))
    if out_png:
        from PIL import Image
        img = rgb.copy()
        rr, cc = int(round(r)), int(round(c))
        img[max(0, rr - 2):rr + 3, max(0, cc - 2):cc + 3] = [255, 0, 0]
        Image.fromarray(img).resize((512, 512), Image.NEAREST).save(out_png)
        print("wrote", out_png)


if __name__ == "__main__":
    main(*sys.argv[1:])
