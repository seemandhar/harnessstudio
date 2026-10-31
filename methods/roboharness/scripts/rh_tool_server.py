"""RoboHarness tool server: SAM3 image tools + the multimodal retrieval encoder, over HTTP (stdlib only).

Ported from the authors' RoboHarness repo (MIT, commit 716defc): src/sam3_service.py (SAM3 masks and the four
image tools, unchanged pixel operations) and src/soma_encoder.py (CLIP ViT-L/14 + MiniLM + task hash, weights
0.6/0.3/0.1, 1168-d). One difference in structure, not in behaviour: the original service calls the VLM itself
to detect a bounding box before choosing among SAM3 masks; here the HarnessStudio controller makes that VLM call
(so it is budgeted and traced) and sends the box with the request (`bbox`, [x1, y1, x2, y2] pixels or null). The
mask choice given that box is the original `_get_mask` (Strategy A, then the pure-SAM3 fallback B).

Run it in an environment with torch, transformers, opencv and the `sam3` package, e.g.
    CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python rh_tool_server.py --port 8131 \
        --sam3_weight_path /bigdata/sejain/frozen-harness-bench/models/rpent-sam3/sam3.pt
Endpoints (POST JSON, images as base64 PNG): /visual_overlay /remove_distractor /replace_texture
/replace_background /embed; GET /health.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rh_tool_server")

sam3_predictor = None
encoder = None
LOCK = threading.Lock()  # SAM3 processor state and the encoder are not thread-safe


# ---------------------------------------------------------------- image codecs (sam3_service.py)
def _decode_image(data_url_or_b64: str) -> np.ndarray:
    if not data_url_or_b64:
        raise ValueError("image is empty")
    if "," in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
    raw = base64.b64decode(data_url_or_b64)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(pil)


def _encode_png_b64(img: np.ndarray) -> str:
    pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------- SAM3 (sam3_service.py)
def init_sam3_model(device: str, sam3_weight_path: Optional[str]) -> None:
    global sam3_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    logger.info(f"Loading SAM3 model on {device} from {sam3_weight_path}")
    model = build_sam3_image_model(checkpoint_path=sam3_weight_path, load_from_HF=(sam3_weight_path is None),
                                   device=device, eval_mode=True, enable_segmentation=True,
                                   enable_inst_interactivity=False)
    model = model.to(device).eval()
    sam3_predictor = Sam3Processor(model, device=device)
    logger.info("SAM3 loaded")


def _autocast():
    """SAM3 runs under bf16 autocast (as RPent's SAM3 server); autocast is thread-local, so enter it per request."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else torch.autocast("cpu", enabled=False)


def _np(t) -> np.ndarray:
    t = t.detach().cpu()
    return (t if t.dtype == torch.bool else t.float()).numpy()


def _get_mask(image: np.ndarray, prompt: str, bbox: Optional[list], score_th: float = 0.25) -> Tuple[np.ndarray, float]:
    """Original _get_mask: Strategy A (VLM box picks among SAM3 text masks) then Strategy B (best SAM3 score)."""
    if sam3_predictor is None:
        raise RuntimeError("SAM3 not initialized")
    if not prompt:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), 0.0
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    with _autocast():
        state = sam3_predictor.set_image(pil)

    # --- Strategy A: the VLM box (detected by the controller) filters SAM3's text-prompt masks ---
    if bbox:
        try:
            with _autocast():
                result = sam3_predictor.set_text_prompt(state=state, prompt=prompt)
            masks = _np(result.get("masks", torch.tensor([])))
            scores = _np(result.get("scores", torch.tensor([])))
            if len(masks) > 0:
                best_iou, best_idx = -1.0, -1
                box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                for i, m in enumerate(masks):
                    m_bool = m.squeeze() > 0
                    if not np.any(m_bool):
                        continue
                    ys, xs = np.where(m_bool)
                    mx1, mx2, my1, my2 = xs.min(), xs.max(), ys.min(), ys.max()
                    ix1 = max(bbox[0], mx1); iy1 = max(bbox[1], my1)
                    ix2 = min(bbox[2], mx2); iy2 = min(bbox[3], my2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    metric = inter / (box_area + 1e-6)
                    if metric > best_iou:
                        best_iou, best_idx = metric, i
                if best_idx != -1 and best_iou > 0.1:
                    return (masks[best_idx].squeeze() > 0).astype(np.uint8), float(scores[best_idx])
        except Exception as e:  # noqa: BLE001 - original falls back to pure SAM3
            logger.warning(f"box-guided mask failed, falling back to pure SAM3: {e}")

    # --- Strategy B: pure SAM3 text mode (fallback) ---
    with _autocast():
        result = sam3_predictor.set_text_prompt(state=state, prompt=prompt)
    scores = _np(result.get("scores", torch.tensor([])))
    masks = _np(result.get("masks", torch.tensor([])))
    if len(scores) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8), 0.0
    best = int(np.argmax(scores))
    if scores[best] < score_th:
        return np.zeros(image.shape[:2], dtype=np.uint8), float(scores[best])
    return (masks[best].squeeze() > 0).astype(np.uint8), float(scores[best])


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def visual_overlay(data: dict) -> dict:
    img = _decode_image(data.get("image", ""))
    target = data.get("target_object") or data.get("prompt")
    color = data.get("color", [0, 255, 0])
    alpha = float(data.get("alpha", 0.4))
    mask, score = _get_mask(img, target, data.get("bbox"))
    if not np.any(mask):
        return {"success": False, "image": _encode_png_b64(img), "message": f"mask not found (score={score:.3f})"}
    out = img.copy()
    overlay = np.zeros_like(out)
    overlay[mask == 1] = np.array(color, dtype=np.uint8)
    out = (out * (1 - alpha) + overlay * alpha).astype(np.uint8)
    out[mask == 0] = img[mask == 0]
    return {"success": True, "image": _encode_png_b64(out), "message": "ok", "score": score}


def remove_distractor(data: dict) -> dict:
    img = _decode_image(data.get("image", ""))
    distractor = data.get("object_to_remove") or data.get("prompt")
    mask, score = _get_mask(img, distractor, data.get("bbox"))
    if not np.any(mask):
        return {"success": False, "image": _encode_png_b64(img), "message": f"mask not found (score={score:.3f})"}
    kernel = np.ones((7, 7), np.uint8)
    mask_d = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, mask_d, 5, cv2.INPAINT_TELEA)
    out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return {"success": True, "image": _encode_png_b64(out), "message": "ok", "score": score}


def replace_texture(data: dict) -> dict:
    img = _decode_image(data.get("image", ""))
    target = data.get("target_object")
    texture_b64 = data.get("texture_image")
    if not target or not texture_b64:
        return {"success": False, "image": _encode_png_b64(img), "message": "target_object and texture_image are required"}
    texture = _decode_image(texture_b64)
    mask, score = _get_mask(img, target, data.get("bbox"))
    if not np.any(mask):
        return {"success": False, "image": _encode_png_b64(img), "message": f"mask not found (score={score:.3f})"}
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return {"success": False, "image": _encode_png_b64(img), "message": "bbox not found"}
    x1, y1, x2, y2 = bbox
    h, w = max(1, y2 - y1 + 1), max(1, x2 - x1 + 1)
    tex = np.array(Image.fromarray(texture).resize((w, h))).astype(np.uint8)
    out = img.copy()
    region_mask = mask[y1:y2 + 1, x1:x2 + 1]
    region = out[y1:y2 + 1, x1:x2 + 1]
    region[region_mask == 1] = tex[region_mask == 1]
    out[y1:y2 + 1, x1:x2 + 1] = region
    return {"success": True, "image": _encode_png_b64(out), "message": "ok", "score": score}


def replace_background(data: dict) -> dict:
    img = _decode_image(data.get("image", ""))
    region_prompt = data.get("region_prompt") or data.get("prompt")
    texture_b64 = data.get("texture_image")
    if not region_prompt or not texture_b64:
        return {"success": False, "image": _encode_png_b64(img), "message": "region_prompt and texture_image are required"}
    texture = _decode_image(texture_b64)
    mask, score = _get_mask(img, region_prompt, data.get("bbox"))
    if not np.any(mask):
        return {"success": False, "image": _encode_png_b64(img), "message": f"mask not found (score={score:.3f})"}
    H, W = img.shape[0], img.shape[1]
    th, tw = texture.shape[0], texture.shape[1]
    if th <= 0 or tw <= 0:
        return {"success": False, "image": _encode_png_b64(img), "message": "invalid texture"}
    tiled = np.tile(texture, (int(np.ceil(H / th)), int(np.ceil(W / tw)), 1))[:H, :W, :]
    alpha = float(data.get("alpha", 1.0))
    out = img.copy()
    blended = (img.astype(np.float32) * (1 - alpha) + tiled.astype(np.float32) * alpha).astype(np.uint8)
    out[mask == 1] = blended[mask == 1]
    return {"success": True, "image": _encode_png_b64(out), "message": "ok", "score": score}


# ---------------------------------------------------------------- encoder (soma_encoder.py)
class AdvancedEmbeddingEncoder:
    """Original encoder: [0.6 * CLIP image, 0.3 * MiniLM text, 0.1 * sha256 task hash], each L2-normalized,
    concatenated and L2-normalized again (768 + 384 + 16 = 1168 dims). FP16 on CUDA, as in the original."""

    def __init__(self, vision_model_name: str = "openai/clip-vit-large-patch14",
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda",
                 hash_len: int = 16, weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)):
        from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
        self.device = device if torch.cuda.is_available() else "cpu"
        self.hash_len, self.weights = hash_len, weights
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.clip_model = CLIPModel.from_pretrained(vision_model_name, torch_dtype=dtype).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(vision_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name, torch_dtype=dtype).to(self.device).eval()

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _get_vision_embedding(self, images) -> np.ndarray:
        with torch.no_grad():
            inputs = self.clip_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            vec = self.clip_model.get_image_features(**inputs).float().cpu().numpy()[0]
        return self._l2_normalize(vec)

    @lru_cache(maxsize=128)
    def _get_text_embedding_cached(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.text_tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            token_embeddings = self.text_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            text_features = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return self._l2_normalize(text_features.float().cpu().numpy()[0])

    def _get_hash_embedding(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        h_vals = np.frombuffer(h, dtype=np.uint8)[: self.hash_len].astype(np.float32) / 255.0
        return self._l2_normalize(h_vals)

    def embed(self, image: Image.Image, task_desc: str) -> list:
        v_vec = self._get_vision_embedding([image])
        t_vec = self._get_text_embedding_cached(task_desc)
        h_vec = self._get_hash_embedding(task_desc)
        w_v, w_t, w_h = self.weights
        combined = np.concatenate([v_vec * w_v, t_vec * w_t, h_vec * w_h])
        return self._l2_normalize(combined).tolist()


def embed(data: dict) -> dict:
    if encoder is None:
        raise RuntimeError("encoder not loaded")
    img = Image.fromarray(_decode_image(data.get("image", ""))).convert("RGB")
    return {"success": True, "embedding": encoder.embed(img, str(data.get("text", "")))}


ROUTES = {"/visual_overlay": visual_overlay, "/remove_distractor": remove_distractor,
          "/replace_texture": replace_texture, "/replace_background": replace_background, "/embed": embed}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "sam3_loaded": sam3_predictor is not None, "encoder_loaded": encoder is not None})
        else:
            self._send(404, {"success": False, "message": "not found"})

    def do_POST(self):  # noqa: N802
        fn = ROUTES.get(self.path)
        if fn is None:
            self._send(404, {"success": False, "message": "not found"})
            return
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                out = fn(data)
            self._send(200, out)
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self.path} failed: {e}", exc_info=True)
            self._send(500, {"success": False, "message": str(e)})

    def log_message(self, fmt, *args):  # quiet access log
        pass


def main() -> None:
    global encoder
    ap = argparse.ArgumentParser(description="RoboHarness tool server (SAM3 tools + retrieval encoder)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sam3_weight_path", default=os.environ.get("SAM3_CHECKPOINT_PATH"))
    ap.add_argument("--no_sam3", action="store_true", help="encoder only (memory building)")
    # Same weights as the original's hub ids; a local snapshot dir can be given when the hub cache has no refs.
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--minilm", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    if not args.no_sam3:
        init_sam3_model(args.device, args.sam3_weight_path)
    encoder = AdvancedEmbeddingEncoder(vision_model_name=args.clip, text_model_name=args.minilm, device=args.device)
    logger.info(f"encoder: clip={args.clip} minilm={args.minilm}")
    logger.info(f"RoboHarness tool server on {args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
