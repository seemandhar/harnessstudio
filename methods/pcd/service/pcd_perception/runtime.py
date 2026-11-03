"""Wire the real frozen models into the engine (paths from the readiness weights manifest)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_ROOT = os.path.join(SERVICE_DIR, "third_party", "sam2_gsam2_5c6166f")
LAMA_ROOT = os.path.join(SERVICE_DIR, "third_party", "lama_ia_3889cc7", "lama")
DEFAULT_MANIFEST = ("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/pcd-port-20261103/readiness/"
                    "weights_manifest.json")


def install_paths() -> None:
    """Put the service package and the two vendored Apache-2.0 trees first on sys.path (nothing is pip-installed)."""
    for p in (LAMA_ROOT, SAM2_ROOT, SERVICE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class WeightPaths:
    gdino_dir: str
    sam2_ckpt: str
    lama_dir: str

    @classmethod
    def from_manifest(cls, path: str = DEFAULT_MANIFEST) -> "WeightPaths":
        with open(path) as f:
            man = json.load(f)
        by = {m["name"]: m for m in man["models"]}
        return cls(gdino_dir=by["grounding-dino-base"]["path"],
                   sam2_ckpt=os.path.join(by["sam2.1-hiera-large"]["path"], "sam2.1_hiera_large.pt"),
                   lama_dir=by["big-lama"]["path"])


def set_authors_torch_flags() -> dict:
    """Process-wide cuDNN flags as the authors' setup_torch_seed sets them (relayed critic edit S-2):
    cudnn.deterministic = True, cudnn.benchmark = False. They only affect cuDNN (GPU); harmless on CPU."""
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch_flags()


def torch_flags() -> dict:
    import torch
    return {"cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}


def build_real_engine(device: str = "cpu", weights: Optional[WeightPaths] = None, classnames=None,
                      aliases=None):
    install_paths()
    set_authors_torch_flags()
    from .engine import CounterfactualEngine
    from .models import SAM2_CONFIG, GroundedSAMDetector, LamaInpainter
    from .sam2_device import build_camera_predictor, load_sam2_state_dict
    from .vocab import CLASSNAMES, GDINO_ALIASES

    weights = weights or WeightPaths.from_manifest()
    sam2_sd = load_sam2_state_dict(weights.sam2_ckpt)
    detector = GroundedSAMDetector(weights.gdino_dir, sam2_sd, device, dict(aliases or GDINO_ALIASES))
    inpainter = LamaInpainter(weights.lama_dir, device, LAMA_ROOT)

    def tracker_factory():
        # The authors rebuild the camera predictor for every episode (contrast_image_generator.py:80-81); a fresh
        # instance also resets SAM2CameraPredictor.frame_idx, which load_first_frame does not reset.
        return build_camera_predictor(SAM2_CONFIG, sam2_sd, device)

    engine = CounterfactualEngine(detector, tracker_factory, inpainter, classnames=classnames or CLASSNAMES)
    return engine


def sam2_cuda_extension_available() -> bool:
    install_paths()
    try:
        from sam2 import _C  # noqa: F401
        return True
    except Exception:
        return False
