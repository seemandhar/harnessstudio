"""Device-parameterised SAM2 camera predictor and its builder.

The vendored real-time camera predictor (third_party/sam2_gsam2_5c6166f/sam2/sam2_camera_predictor.py, git blob
9caed4ad7b7d19bc320567dd6428728af0ed2250 = Gy920/segment-anything-2-real-time@53eb82a = the blob vendored by
Koorye/PCD@31b027e third_party/grounded_sam_2/sam2/sam2_camera_predictor.py) hard-codes CUDA in three methods
used on the PCD path (_init_state :107-111, _get_image_feature :1029, _get_feature :1055). This subclass repeats
those three methods with `self._pcd_device` in place of "cuda"; every other line runs from the pinned blob. With
device "cuda" it is the original code. CPU is used only for the CPU parity check.

`build_camera_predictor` restates the vendored builder build_sam2_camera_predictor (Koorye/PCD third_party/
grounded_sam_2/sam2/build_sam.py:137-177, itself taken from segment-anything-2-real-time) with a `device`
argument and the hydra target pointing at this subclass; the overrides are the same five values.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2  # noqa: F401  (initialises the hydra config module "sam2")
from sam2.build_sam import _load_checkpoint
from sam2.sam2_camera_predictor import SAM2CameraPredictor

POSTPROCESS_OVERRIDES = [
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    "++model.binarize_mask_from_pts_for_mem_enc=true",
    "++model.fill_hole_area=8",
]


class DeviceCameraPredictor(SAM2CameraPredictor):
    _pcd_device = "cuda"

    def _init_state(self, offload_video_to_cpu=False, offload_state_to_cpu=False):
        state = super()._init_state(offload_video_to_cpu=offload_video_to_cpu,
                                    offload_state_to_cpu=offload_state_to_cpu)
        state["device"] = torch.device(self._pcd_device)
        state["storage_device"] = torch.device("cpu") if offload_state_to_cpu else torch.device(self._pcd_device)
        return state

    def _get_image_feature(self, frame_idx, batch_size):
        image, backbone_out = self.condition_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            image = self.condition_state["images"][frame_idx].to(self._pcd_device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            self.condition_state["cached_features"] = {frame_idx: (image, backbone_out)}
        return self._expand(image, backbone_out, batch_size)

    def _get_feature(self, img, batch_size):
        image = img.to(self._pcd_device).float().unsqueeze(0)
        backbone_out = self.forward_image(image)
        return self._expand(image, backbone_out, batch_size)

    def _expand(self, image, backbone_out, batch_size):
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded = {"backbone_fpn": backbone_out["backbone_fpn"].copy(),
                    "vision_pos_enc": backbone_out["vision_pos_enc"].copy()}
        for i, feat in enumerate(expanded["backbone_fpn"]):
            expanded["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded["vision_pos_enc"]):
            expanded["vision_pos_enc"][i] = pos.expand(batch_size, -1, -1, -1)
        features = self._prepare_backbone_features(expanded)
        return (expanded_image,) + features

    # Point/box prompting is not on the PCD GDINO path and still contains hard-coded .cuda() calls.
    def add_new_prompt(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("not used by PCD (mask prompts only)")

    def add_new_points(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("not used by PCD (mask prompts only)")


def device_class(device: str):
    """A subclass bound to one device (hydra instantiates by import path, so bind via class attribute)."""
    DeviceCameraPredictor._pcd_device = str(device)
    return DeviceCameraPredictor


def build_camera_predictor(config_file: str, state_dict: "OrderedDict[str, torch.Tensor]", device: str):
    device_class(device)
    overrides = ["++model._target_=pcd_perception.sam2_device.DeviceCameraPredictor"] + POSTPROCESS_OVERRIDES
    cfg = compose(config_name=config_file, overrides=overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    missing, unexpected = model.load_state_dict(state_dict)
    if missing or unexpected:
        raise RuntimeError(f"SAM2 camera predictor state mismatch: missing={missing} unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    return model


def load_sam2_state_dict(ckpt_path: str):
    """Same read as build_sam._load_checkpoint (torch.load weights_only=True, key 'model')."""
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]


__all__ = ["DeviceCameraPredictor", "build_camera_predictor", "load_sam2_state_dict", "_load_checkpoint"]
