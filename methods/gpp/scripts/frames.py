#!/usr/bin/env python3
"""Frame packets for the gpp physical/semantic witnesses (review N4, H1-RC3/RC4, H2-RC3/RC4): for every primitive in a
finished job's episodes, a contact sheet of the public agentview mp4 at the recorded steps fire, start, end, +10, +30
(accounting.extra['gpp'].fires[].frames), plus a JSON with the mp4 sha256 and the frame offset.

    python3 harnessstudio/methods/gpp/scripts/frames.py --plan EVAL_DIR/plan.json [--job ID ...] [--out-dir DIR]

Run with a python that has cv2 (system python3; not the HarnessStudio venv, not the policy server env).
Frame index = step + offset, offset = n_frames - n_steps (the evidence convention of idea-r2b frames.py; for
LIBEROSelect videos PITFALLS P-20261101-15 gives offset 2). The sheet is evidence for a human/critic judgement; this
script judges nothing.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

KEYS = ("fire", "start", "end", "plus10", "plus30")


def frames_of(mp4: str):
    cap = cv2.VideoCapture(mp4)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--job", action="append", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    jobs = [j for j in plan["jobs"] if args.job is None or j["id"] in args.job]
    out_dir = Path(args.out_dir or Path(plan["eval_dir"]) / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for j in jobs:
        for arm in j["arms"].split(","):
            arm_dir = Path(j["out"]) / arm
            recs = {}
            rf = arm_dir / "records.jsonl"
            if not rf.is_file():
                continue
            for line in rf.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    recs[int(r["episode_idx"])] = r
            for k, init in enumerate(j["inits"]):
                r = recs.get(init)
                fires = (((r or {}).get("accounting") or {}).get("extra") or {}).get("gpp", {}).get("fires", [])
                mp4s = glob.glob(str(arm_dir / ("vla_eval/episodes/*/task0000_ep%04d_*.mp4" % k)))
                if not fires or len(mp4s) != 1:
                    continue
                fr = frames_of(mp4s[0])
                steps = int(r["accounting"]["steps"])
                off = len(fr) - steps
                for n, f in enumerate(fires):
                    tiles = []
                    for key in KEYS:
                        s = (f.get("frames") or {}).get(key)
                        if s is None:
                            continue
                        img = fr[min(len(fr) - 1, max(0, s + off))].copy()
                        img = cv2.resize(img, (256, 256))
                        cv2.putText(img, f"{j['scene']} {arm} {f['module']}/{f['variant']} {key} s{s}", (4, 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
                        tiles.append(img)
                    png = out_dir / f"{j['id']}-{arm}-fire{n}-{f['module']}.png"
                    cv2.imwrite(str(png), np.concatenate(tiles, 1))
                    index.append({"job": j["id"], "arm": arm, "scene": j["scene"], "fire": n, "module": f["module"],
                                  "variant": f["variant"], "outcome": f.get("outcome"), "frames": f.get("frames"),
                                  "png": str(png), "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                                  "mp4": mp4s[0], "mp4_sha256": hashlib.sha256(open(mp4s[0], "rb").read()).hexdigest(),
                                  "n_frames": len(fr), "steps": steps, "offset": off})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} packets -> {out_dir}")


if __name__ == "__main__":
    main()
