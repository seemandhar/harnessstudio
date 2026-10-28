"""M/R/X first-divergence attribution of paired runs (spec §4.5) with the copied ChunkTrust tools.

RI-2 render scan: copied ri_scan.py (needs cv2 -> run with the GR00T env interpreter).
Classification: copied np_attribution_v2.classify (stdlib), parent = reference/baseline run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

COPIED = Path(__file__).resolve().parents[1] / "backends" / "gr00t_robocasa365" / "copied"
def _default_bench() -> Path:
    from .. import config
    return config.policybench()


def child_python(run_dir) -> Path:
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    bench = manifest.get("policybench")
    return Path(bench if bench else _default_bench()) / "envs/gr00t-n15/bin/python"


def ri_scan(run_dir, out_json) -> Path:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(child_python(run_dir)), str(COPIED / "ri_scan.py"), "--runs", str(run_dir),
                           "--out", str(out_json)], capture_output=True, text=True, timeout=1800,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    if proc.returncode not in (0, 3) or not out_json.is_file():  # 3 = some video flagged (still a valid report)
        raise RuntimeError("RI-2 scan failed: " + proc.stderr[-800:])
    return out_json


def _npa():
    if str(COPIED) not in sys.path:
        sys.path.insert(0, str(COPIED))
    import np_attribution_v2
    return np_attribution_v2


def classify_runs(parent_run, other_run, out_dir) -> dict:
    """Classify every case of other_run against parent_run. Requires both RI reports to cover all cases."""
    npa = _npa()
    out_dir = Path(out_dir)
    cases = [c["id"] for c in json.loads((Path(parent_run) / "manifest.json").read_text())["cases"]]
    other_cases = [c["id"] for c in json.loads((Path(other_run) / "manifest.json").read_text())["cases"]]
    if cases != other_cases:
        raise ValueError("paired runs have different case lists")
    flags_p = npa.ri_flags(ri_scan(parent_run, out_dir / "ri_parent.json"))
    flags_o = npa.ri_flags(ri_scan(other_run, out_dir / "ri_other.json"))
    missing = [c for c in cases if c not in flags_p or c not in flags_o]
    per_case = {}
    for case in cases:
        per_case[case] = (npa.classify(case, str(parent_run), str(other_run), flags_p, flags_o, None)
                          if case not in missing else {"class": "X", "reason": "RI report does not cover case"})
    counts = Counter(v["class"] for v in per_case.values())
    report = {"parent": str(parent_run), "other": str(other_run), "cases": len(cases),
              "M": counts.get("M", 0), "R": counts.get("R", 0), "X": counts.get("X", 0),
              "ri_missing": missing, "per_case": per_case}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attribution.json").write_text(json.dumps(report, indent=2, default=str))
    return report
