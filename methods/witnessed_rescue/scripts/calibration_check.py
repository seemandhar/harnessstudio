"""Review check A4 on retained RGB-D packets (written by the controller when HS_WR_PACKET_DIR is set, e.g. from
tier-2 / respondoff episodes): per packet, the projected end effector (grip site) pixel, the back-projected point at
that pixel and its distance to the eef, the table-plane height and eef z - table z. Writes an overlay PNG per packet
with the projected eef marked so a blind labeller can check 'EE projects onto the gripper within 6 px'.
No policy, no outcome.

  python scripts/calibration_check.py PACKET_DIR --out calib.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hs_witnessed_rescue import perception as P  # noqa: E402


def load(npz_path: str) -> P.Packet:
    z = np.load(npz_path)
    meta = json.loads(Path(npz_path[:-4] + ".json").read_text())
    obs = {"images": {}, "depth": {}, "camera": meta.get("camera") or {}}
    for k in z.files:
        if k.startswith("images__"):
            obs["images"][k[8:]] = z[k]
        elif k.startswith("depth__"):
            obs["depth"][k[7:]] = z[k]
    if "states" in z.files:
        obs["states"] = z["states"]
    return P.Packet(int(meta["step"]), obs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packet_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from harnessstudio.tools.llm import png_bytes
    rows = []
    for f in sorted(glob.glob(str(Path(a.packet_dir) / "**" / "*.npz"), recursive=True)):
        pk = load(f)
        try:
            rgb, depth, cam = P.views(pk.obs, "agentview")
        except P.PerceptionUnavailable as exc:
            rows.append({"packet": f, "error": str(exc)})
            continue
        ee = pk.state()[:3]
        r, c = P.project(ee, cam)
        wmap = P.world_map(depth, cam)
        ok = P.valid_mask(wmap, depth)
        ri, ci = int(round(r)), int(round(c))
        inside = 0 <= ri < depth.shape[0] and 0 <= ci < depth.shape[1]
        bp = wmap[ri, ci] if inside and ok[ri, ci] else None
        zt = P.table_height(wmap, ok, ee, P.DEFAULTS)
        rows.append({"packet": f, "step": pk.step, "ee": [round(float(v), 4) for v in ee],
                     "ee_pixel": [round(r, 1), round(c, 1)], "inside": inside,
                     "backprojected_at_ee_pixel": None if bp is None else [round(float(v), 4) for v in bp],
                     "dist_bp_to_ee_m": None if bp is None else round(float(np.linalg.norm(bp - ee)), 4),
                     "table_z": None if zt is None else round(zt, 4),
                     "ee_minus_table_m": None if zt is None else round(float(ee[2] - zt), 4)})
        png = P.draw_marker(P.upsample(rgb, 2), r * 2, c * 2, radius=10)
        Path(f[:-4] + "_eeproj.png").write_bytes(png_bytes(png))
    d = [x["dist_bp_to_ee_m"] for x in rows if x.get("dist_bp_to_ee_m") is not None]
    summary = {"packets": len(rows), "within_3cm": sum(1 for v in d if v <= 0.03), "with_backprojection": len(d)}
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
