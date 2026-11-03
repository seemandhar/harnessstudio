# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval==0.8.0",
#     "torch==2.3.1",
#     "transformers==4.44.2",
#     "timm==0.9.10",
#     "tokenizers==0.19.1",
#     "pillow",
#     "numpy<2",
#     "accelerate",
#     "opencv-python-headless",
#     "pyyaml>=6",
# ]
# ///
"""HarnessStudio OpenVLA server, mode ``simpler_pcd_authors`` (EXT simpler-seeded, docs/extensions/simpler-seeded.md).

It reproduces the OpenVLA inputs and decoding of the PCD authors' SimplerEnv wrapper (Koorye/PCD@31b027e,
simpler_env/policies/openvla/openvla_model.py and contrast_policies/openvla_contrast.py), as a subclass of the
openvla-contrast logits server. ``openvla_hs.py`` and ``openvla_logits_hs.py`` stay byte-identical. The mode is
fixed per process.

Declared differences from the parents (pinned by tests/test_openvla_simpler_server.py):
  [a] prompt: the raw instruction, no OpenVLA template (openvla_model.py:102); token 29871 is appended to the ids
      if absent (openvla_model.py:108-109).
  [b] image: the first image, uint8 HxWx3 at full resolution, cv2.resize to 224x224 with INTER_AREA
      (openvla_model.py:173-175); no JPEG round trip, no crop.
  [S-1] generate is called with input_ids (+29871) and pixel_values only, never with the processor's attention_mask
      (openvla_model.py:114-121, openvla_contrast.py:116-123). predict_action(**inputs) would pass a mask one token
      shorter than the ids, which crashes under eager attention.
  [load] attn_implementation="eager", bf16, low_cpu_mem_usage=True, trust_remote_code from a local snapshot
      (openvla_model.py:43-50); asserts the Llama backbone's attention class is LlamaAttention [S-5c].
  [S-2] torch.backends.cudnn.deterministic = True and cudnn.benchmark = False, set process-wide at start-up, as the
      authors' setup_torch_seed does on every policy.reset (simpler_env/policies/__init__.py:7-15).
  [c] actions: float64 de-tokenization + q01/q99 un-normalization, the same formula as openvla_model.py:209-218.
      No -sign(2g-1) gripper map (the SimplerSeededBenchmark applies the authors' relative sticky gripper). The
      transport still casts to float32 (policy_ext/protocol.py:61), which is disclosed.
  [S-4] optional float64 pre-cast action log (``action_log`` = a run-local JSONL path that must NOT exist yet: one
      fresh log per server launch). It records the request sequence number, the (session, episode) ids, the path, the
      tokens and the float64 bytes. It is behaviour-free.
  [S-5a] action spec: POSITION_DELTA, ROTATION_EULER (raw roll/pitch/yaw), GRIPPER_RAW; observation params {}.
  [A-7] fewer generated tokens than action dims (EOS) raises SimplerDecodeShort, an infrastructure event; the
      authors' baseline would crash on it as well.
  Refused in this mode: samplers / decoding hooks (the SCALE path keeps the template prompt), jpeg_roundtrip,
  center_crop.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import GRIPPER_RAW, POSITION_DELTA, ROTATION_EULER, DimSpec

from harnessstudio.policies.openvla_logits_hs import READOUT, HarnessOpenVLALogitsServer, _recorders

logger = logging.getLogger(__name__)

MODE = "simpler_pcd_authors"
EMPTY_TOKEN = 29871                      # openvla_model.py:108-109 / modeling_prismatic.py:512-515
IMAGE_SIZE = (224, 224)                  # OPENVLA_CONFIG image_size (properties.py)
EAGER_ATTENTION_CLASS = "LlamaAttention"


class SimplerDecodeShort(RuntimeError):
    """generate returned fewer tokens than action dimensions (EOS); an infrastructure event, never an action."""


class HarnessOpenVLASimplerServer(HarnessOpenVLALogitsServer):
    harness_capabilities = HarnessOpenVLALogitsServer.harness_capabilities - {"decoding_hooks"}

    def __init__(self, model_path: str, unnorm_key: str = "fractal20220817_data", *, chunk_size: int = 1,
                 action_ensemble: str = "newest", action_log: Optional[str] = None,
                 jpeg_roundtrip: bool = False, center_crop: bool = False,
                 sampler_prefixes: Sequence[str] = ("hs_",), **kwargs: Any) -> None:
        if jpeg_roundtrip or center_crop:
            raise ValueError(f"mode {MODE}: jpeg_roundtrip/center_crop are not part of the authors' preprocessing")
        if not Path(model_path).is_dir():
            raise ValueError(f"mode {MODE}: model_path must be a local pinned snapshot directory, got {model_path!r}")
        # OpenVLAModelServer.__init__ (vla_eval/model_servers/openvla.py:44-73) with the [load] difference; the
        # parent constructor is NOT called because it would load the model with the default (SDPA) attention.
        PredictModelServer.__init__(self, chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path, self.unnorm_key = model_path, unnorm_key
        self.jpeg_roundtrip, self.center_crop = False, False
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        torch.backends.cudnn.deterministic = True     # [S-2]
        torch.backends.cudnn.benchmark = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_path, attn_implementation="eager", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).to(self._device)
        self._check_model()
        # HarnessOpenVLAServer.__init__ state (policies/openvla_hs.py:99-105)
        self._infer_lock = threading.RLock()
        self._sampler_prefixes = tuple(sampler_prefixes)
        self._decoders: dict = {}
        self._states: dict = {}
        self._init_log(action_log)

    # ---- start-up checks and the [S-4] log ------------------------------------------------------------------
    def _check_model(self) -> None:
        import torch

        attn = type(self._model.language_model.model.layers[0].self_attn).__name__
        if getattr(self._model.config, "_attn_implementation", None) != "eager" or attn != EAGER_ATTENTION_CLASS:
            raise RuntimeError(f"mode {MODE}: expected eager attention ({EAGER_ATTENTION_CLASS}), got "
                               f"{self._model.config._attn_implementation!r} / {attn}")
        if not (torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark):
            raise RuntimeError(f"mode {MODE}: cuDNN must be deterministic with benchmark off")
        self.get_action_stats()   # the unnorm key must exist

    def _init_log(self, action_log: Optional[str]) -> None:
        self._action_log = Path(action_log) if action_log else None
        self._log_lock = threading.Lock()
        self._log_seq = 0
        self._log_ctx = threading.local()
        if self._action_log is not None:
            # one fresh log per server launch: seq restarts at 1 in every process, so appending to an earlier
            # launch's file would make (seq, session, episode) ambiguous (review critic-simpler-diff edit 6)
            if self._action_log.exists():
                raise FileExistsError(f"mode {MODE}: action_log {self._action_log} exists; use a fresh path per launch")
            self._action_log.parent.mkdir(parents=True, exist_ok=True)

    def _log_action(self, path: str, tokens: np.ndarray, action: np.ndarray) -> None:
        if self._action_log is None:
            return
        ctx = getattr(self._log_ctx, "ctx", None)
        a = np.ascontiguousarray(action, dtype=np.float64)
        with self._log_lock:
            self._log_seq += 1
            rec = {"seq": self._log_seq, "session": getattr(ctx, "session_id", None),
                   "episode": getattr(ctx, "episode_id", None), "path": path,
                   "tokens": [int(t) for t in tokens], "action_f64": a.reshape(-1).tolist(),
                   "action_f64_hex": a.tobytes().hex()}
            with open(self._action_log, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # ---- interface specs [S-5a] -----------------------------------------------------------------------------
    def get_observation_params(self) -> dict[str, Any]:
        return {}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": POSITION_DELTA, "rotation": ROTATION_EULER, "gripper": GRIPPER_RAW}

    def get_action_stats(self) -> Mapping[str, Any]:
        return self._model.get_action_stats(self.unnorm_key)

    # ---- inputs [a] [b] -------------------------------------------------------------------------------------
    def _preprocess_image(self, obs: Mapping[str, Any]) -> Any:
        import cv2
        from PIL import Image as PILImage

        images = obs.get("images", {})
        img = next(iter(images.values())) if isinstance(images, Mapping) else images
        img = np.asarray(img)
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"mode {MODE}: expected a uint8 HxWx3 image, got {img.dtype} {img.shape}")
        return PILImage.fromarray(cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA))

    def _prompt(self, obs: Mapping[str, Any]) -> str:
        return str(obs.get("task_description", ""))

    def _inputs(self, obs: Mapping[str, Any]) -> dict:
        raise ValueError(f"mode {MODE}: samplers / decoding hooks are not offered (the SCALE path is template-prompted)")

    # ---- decode [S-1] ---------------------------------------------------------------------------------------
    def _generate(self, obs: Mapping[str, Any], logits_processor=None, stopping_criteria=None) -> np.ndarray:
        """The authors' generate call: ids (+29871) and pixel_values only, greedy, max_new_tokens = action dims."""
        import torch

        inputs = self._processor(self._prompt(obs), self._preprocess_image(obs)).to(self._device, dtype=torch.bfloat16)
        input_ids, pixel_values = inputs["input_ids"], inputs["pixel_values"]
        if not torch.all(input_ids[:, -1] == EMPTY_TOKEN):
            input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([EMPTY_TOKEN]).long(), dim=0)
                                   .to(input_ids.device)), dim=1)
        action_dim = int(self._model.get_action_dim(self.unnorm_key))
        extra = {}
        if logits_processor is not None:
            extra["logits_processor"] = logits_processor
        if stopping_criteria is not None:
            extra["stopping_criteria"] = stopping_criteria
        generated = self._model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=action_dim,
                                         do_sample=False, **extra)
        tokens = generated[0, input_ids.shape[1]:].cpu().numpy()
        if len(tokens) != action_dim:
            raise SimplerDecodeShort(f"generate returned {len(tokens)} tokens for {action_dim} action dims")
        return tokens

    def _actions(self, predicted_action_token_ids: np.ndarray) -> np.ndarray:
        """float64 de-tokenization + un-normalization (openvla_model.py:209-218); no gripper map [c]."""
        model = self._model
        discretized_actions = model.vocab_size - np.asarray(predicted_action_token_ids)
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
        normalized_actions = model.bin_centers[discretized_actions]
        action_norm_stats = self.get_action_stats()
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        return np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                        normalized_actions)

    # ---- native path ----------------------------------------------------------------------------------------
    def predict(self, obs, ctx):
        with self._infer_lock:
            tokens = self._generate(obs)
            action = self._actions(tokens)
            self._log_ctx.ctx = ctx
            self._log_action("native", tokens, action)
            return {"actions": action}

    # ---- HarnessStudio path ---------------------------------------------------------------------------------
    def sample_chunks(self, obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping],
                      readouts: Sequence[str], ctx) -> list:
        if sampler:
            raise ValueError(f"mode {MODE}: samplers / decoding hooks are not offered")
        self._log_ctx.ctx = ctx
        if readouts:   # 'logits' -> parent logits path over the overridden _predict_recorded; others refused there
            return super().sample_chunks(obs, seeds, sampler, readouts, ctx)
        with self._infer_lock:
            out = []
            for seed in seeds:   # greedy: the same native decode for every seed (compute paid per seed)
                tokens = self._generate(obs)
                action = self._actions(tokens)
                self._log_action("native_seeded", tokens, action)
                out.append((action[None, :], {"seed": int(seed)}))
            return out

    def _predict_recorded(self, obs):
        from transformers import LogitsProcessorList, StoppingCriteriaList

        logits_rec, token_rec = _recorders()
        tokens = self._generate(obs, logits_processor=LogitsProcessorList([logits_rec]),
                                stopping_criteria=StoppingCriteriaList([token_rec]))
        action = self._actions(tokens)
        self._log_action(READOUT, tokens, action)
        return action, logits_rec, token_rec


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(HarnessOpenVLASimplerServer)
