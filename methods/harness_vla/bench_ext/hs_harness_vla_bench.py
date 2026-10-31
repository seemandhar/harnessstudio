"""LIBERO-Pro benchmark for the harness-vla package: RGB-D + camera calibration + the BDDL instruction fix.

Runs inside the vla-eval LIBERO-Pro container (Python 3.8). It only composes two HarnessStudio benchmark
classes, so it adds sensor data only (metric depth and camera calibration; no object poses):
  - harnessstudio.benchmarks.libero_rgbd._RGBDMixin (depth + camera keys, pixel convention validated on LIBERO)
  - harnessstudio.benchmarks.libero_pro_instructions.LIBEROProBDDLInstruction (task/lan instruction fix)

Use in a vla-eval config:
  benchmark: "hs_harness_vla_bench:LIBEROProRGBDInstruction"
  docker.volumes: <harnessstudio-v2>:/hs:ro ; docker.env: PYTHONPATH=/hs:/hs/methods/harness_vla/bench_ext
  params: {depth_cameras: [agentview, robot0_eye_in_hand], ...}
"""
from __future__ import annotations

from harnessstudio.benchmarks.libero_pro_instructions import LIBEROProBDDLInstruction
from harnessstudio.benchmarks.libero_rgbd import _RGBDMixin


class LIBEROProRGBDInstruction(_RGBDMixin, LIBEROProBDDLInstruction):
    """LIBERO-Pro + metric depth/camera calibration, sending the BDDL instruction for task/lan perturbations."""

    def __init__(self, *args, depth_cameras=("agentview", "robot0_eye_in_hand"), **kwargs):
        super().__init__(*args, depth_cameras=tuple(depth_cameras), **kwargs)
