"""Frozen tool models: GroundingDINO-base + SAM2.1-L image predictor (detector), SAM2.1-L camera tracker, Big-LaMa.

Upstream references (Koorye/PCD @ 31b027e):
- contrast_utils/mask_predictors.py:55-56,61     SAM2.1 config 'configs/sam2.1/sam2.1_hiera_l.yaml', checkpoint
                                                 pretrained/sam2.1_hiera_large.pt, GDINO pretrained/grounding-dino-base
- contrast_utils/mask_predictors.py:93-101       build_sam2 + SAM2ImagePredictor; HF AutoProcessor / AutoModelFor-
                                                 ZeroShotObjectDetection in fp32 (no dtype argument), one device
- contrast_utils/mask_predictors.py:103-155      predict(): PIL image of a copy; text query; thresholds box 0.25 /
                                                 text 0.3; boxes in pixels (target_sizes = (H, W)); label rewrite;
                                                 SAM2 on ALL boxes, multimask_output=False; per prompt the highest-
                                                 score box with that label; masks > 0; post-processing per mask
- contrast_utils/inpainters.py:52-109            LaMa: OmegaConf prediction config (out_key 'inpainted', checkpoint
                                                 best.ckpt), training config predict_only; image/255 float, mask x255
                                                 when its max is 1; pad to modulo 8 (reflect); mask = (mask > 0) * 1;
                                                 output cropped, clip(x * 255, 0, 255) -> uint8
- third_party/inpaint_anything/lama saicinpainting/training/trainers/default.py forward (eval mode): masked = img *
                                                 (1 - mask); concat mask; generator; inpainted = mask * pred +
                                                 (1 - mask) * img. Here the generator is built with the same
                                                 make_generator(config, **config.generator) call (base.py:67) and
                                                 loaded from the checkpoint's 'generator.*' keys (strict); the
                                                 discriminator/loss modules that predict_only skips are not built.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

from . import masks as M
from .vocab import alias_to_name, gdino_text, label_to_name

BOX_THRESHOLD = 0.25   # mask_predictors.py:116
TEXT_THRESHOLD = 0.3   # mask_predictors.py:117 (ICLR value; the journal release differs)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # mask_predictors.py:55


class GroundedSAMDetector:
    """GroundingDINO boxes -> SAM2 image-predictor masks -> per-prompt best mask (mask_predictors.py:92-155)."""

    def __init__(self, gdino_dir: str, sam2_state_dict, device: str, aliases: Dict[str, str]):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        self.aliases = aliases
        sam = build_sam2(SAM2_CONFIG, None, device="cpu")  # same overrides as the authors' build_sam2 call
        missing, unexpected = sam.load_state_dict(sam2_state_dict)
        if missing or unexpected:
            raise RuntimeError("SAM2 image model state mismatch")
        sam = sam.to(device).eval()
        self.image_predictor = SAM2ImagePredictor(sam)
        self.processor = AutoProcessor.from_pretrained(gdino_dir)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_dir).to(device)
        self.grounding_model.eval()
        self.last: dict = {}

    @torch.no_grad()
    def __call__(self, image: np.ndarray, prompts: Sequence[str],
                 aliases: Optional[Dict[str, str]] = None) -> List[Optional[np.ndarray]]:
        aliases = self.aliases if aliases is None else aliases   # None: the authors' SIMPLER table
        pil = Image.fromarray(image.copy())
        text = gdino_text(prompts, aliases)
        inputs = self.processor(pil, text=text, return_tensors="pt").to(self.device)
        outputs = self.grounding_model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[pil.size[::-1]])[0]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        reverse = alias_to_name(prompts, aliases)
        labels = [label_to_name(l, reverse) for l in result["labels"]]

        self.image_predictor.set_image(image.copy())
        sam_masks, _, _ = self.image_predictor.predict(point_coords=None, point_labels=None, box=boxes,
                                                       multimask_output=False)
        if sam_masks.ndim == 4:
            sam_masks = sam_masks.squeeze(1)
        chosen: List[Optional[np.ndarray]] = []
        for name in prompts:
            hits = [i for i, l in enumerate(labels) if l == name]
            if hits:
                best = hits[int(scores[hits].argmax())]
                chosen.append(sam_masks[best] > 0)
            else:
                chosen.append(None)
        self.last = {"text": text, "labels": labels, "raw_labels": list(result["labels"]),
                     "scores": [float(s) for s in scores], "boxes": boxes.tolist()}
        return [M.postprocess_detection_mask(m) for m in chosen]


# ------------------------------------------------------------------ LaMa
class _Placeholder:
    """Stands in for non-tensor checkpoint metadata (callbacks, OmegaConf hyper-parameters); never executed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _Placeholder()


_ALLOWED = {("collections", "OrderedDict"), ("collections", "defaultdict"), ("torch._utils", "_rebuild_tensor_v2"),
            ("torch._utils", "_rebuild_parameter"), ("builtins", "dict"), ("builtins", "list"),
            ("builtins", "set"), ("builtins", "tuple"), ("builtins", "int"), ("builtins", "float")}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        base = "builtins" if module == "__builtin__" else module
        if name == "long" and base == "builtins":
            return int
        if (base, name) in _ALLOWED or (base == "torch" and name.endswith("Storage")):
            return super().find_class(module, name)
        return _Placeholder


class _RestrictedPickleModule:
    Unpickler = _RestrictedUnpickler
    load = staticmethod(pickle.load)


def load_lama_generator_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Read best.ckpt without executing arbitrary pickled callables; keep only the generator weights."""
    state = torch.load(ckpt_path, map_location="cpu", pickle_module=_RestrictedPickleModule, weights_only=False)
    sd = state["state_dict"]
    return {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}


def pad_tensor_to_modulo(img: torch.Tensor, mod: int) -> torch.Tensor:
    """saicinpainting/evaluation/data.py:23-40 (Apache-2.0): right/bottom reflect pad to a multiple of mod.
    Restated here because importing saicinpainting.evaluation pulls the training/evaluation closure."""
    _, _, h, w = img.shape
    out_h = h if h % mod == 0 else (h // mod + 1) * mod
    out_w = w if w % mod == 0 else (w // mod + 1) * mod
    return torch.nn.functional.pad(img, pad=(0, out_w - w, 0, out_h - h), mode="reflect")


class LamaInpainter:
    def __init__(self, lama_dir: str, device: str, lama_root: str):
        from saicinpainting.training.modules import make_generator

        predict_config = OmegaConf.load(os.path.join(lama_root, "configs", "prediction", "default.yaml"))
        with open(os.path.join(lama_dir, "config.yaml"), "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        tm = train_config.training_model
        # default.py forward uses only concat_mask / add_noise_kwargs in eval mode; refuse configs that differ.
        if not tm.get("concat_mask", True) or tm.get("add_noise_kwargs", None) is not None:
            raise RuntimeError("unsupported LaMa training_model config")
        gen = make_generator(train_config, **train_config.generator)
        gstate = load_lama_generator_state(os.path.join(lama_dir, "models", predict_config.model.checkpoint))
        missing, unexpected = gen.load_state_dict(gstate, strict=True)
        if missing or unexpected:
            raise RuntimeError("LaMa generator state mismatch")
        for p in gen.parameters():
            p.requires_grad = False
        self.generator = gen.to(device).eval()
        self.device = torch.device(device)
        self.out_key = predict_config.out_key
        self._pad = pad_tensor_to_modulo
        self.mod = 8

    @torch.no_grad()
    def __call__(self, image: np.ndarray, region: np.ndarray) -> np.ndarray:
        assert region.ndim == 2
        mask = region * 255 if np.max(region) == 1 else region
        img = torch.from_numpy(image).float().div(255.0)
        m = torch.from_numpy(mask).float()
        img_b = img.permute(2, 0, 1).unsqueeze(0)
        m_b = m[None, None]
        h, w = img_b.shape[2], img_b.shape[3]
        img_b = self._pad(img_b, self.mod).to(self.device)
        m_b = self._pad(m_b, self.mod).to(self.device)
        m_b = (m_b > 0) * 1
        masked = img_b * (1 - m_b)
        masked = torch.cat([masked, m_b], dim=1)
        predicted = self.generator(masked)
        inpainted = m_b * predicted + (1 - m_b) * img_b
        res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()[:h, :w]
        return np.clip(res * 255, 0, 255).astype("uint8")
