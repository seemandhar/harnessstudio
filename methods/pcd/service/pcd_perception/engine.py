"""Per-episode counterfactual state machine and activation evidence.

Upstream references (Koorye/PCD @ 31b027e):
- parallel_inference.py:183,198,223  generator.reset() per episode; generate(obs, instruction) on the reset frame and
                                     after every env step (the frame the policy sees next)
- contrast_image_generator.py:76-91  a new instruction (or the first call after reset) re-parses the objects and
                                     REBUILDS the predictor (fresh detector + fresh camera tracker); then masks ->
                                     inpaint every call, whatever the masks are
- contrast_image_generator.py:116-124 prompts = targets + ["robot"]; robot mask = masks[prompts.index("robot")]
- mask_predictors.py:448-480         TrackingPredictorV2: until tracking starts, detect on every frame; the first
                                     frame where ANY prompt (robot included) has a mask loads the tracker's first
                                     frame and adds each non-None mask with obj_id = CLASSNAMES.index(prompt) + 1,
                                     and returns the detection masks; afterwards track() only (no re-detection), a
                                     mask is (logits > 0) at video resolution, empty -> None, absent object -> None
- inpainters.py:24-33                LaMa on dilate(mask | robot); paste dilate(mask) & ~robot
Tracked masks are NOT post-processed (only detection masks are, mask_predictors.py:155 vs :470-480).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import masks as M
from . import parsers
from .vocab import CLASSNAMES, KEEP_OBJECTS, parse_objects


@dataclass
class EpisodeState:
    instruction: str
    targets: List[str]
    prompts: List[str]
    tracker: object = None
    tracking: bool = False
    step: int = 0
    first_detection_step: Optional[int] = None
    first_detection_found: Dict[str, bool] = field(default_factory=dict)
    tracker_none_steps: Dict[str, List[int]] = field(default_factory=dict)
    mask_present_steps: Dict[str, int] = field(default_factory=dict)
    detect_attempts: int = 0
    rebuilds: int = 0


class CounterfactualEngine:
    """Tools are injected so the same state machine runs with real models or with fakes (unit tests).

    detector(image, prompts) -> list of bool masks or None (post-processed, one per prompt)
    tracker_factory() -> object with load_first_frame(image), add_new_mask(frame_idx, obj_id, mask),
                         track(image) -> (obj_ids, mask_logits tensor/array [N, 1, H, W])
    inpainter(image, region) -> uint8 image of the same shape
    """

    def __init__(self, detector: Callable, tracker_factory: Callable, inpainter: Callable,
                 classnames: Optional[List[str]] = None, get_all_parts: bool = False):
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.inpainter = inpainter
        self.classnames = list(classnames or CLASSNAMES)
        self.get_all_parts = get_all_parts
        self.state: Optional[EpisodeState] = None
        self.parser = parsers.SIMPLER   # authors' parser; other modes are frozen tables chosen at reset
        self._spec = None
        self.calls = {"gdino_sam2_detect": 0, "sam2_first_frame": 0, "sam2_add_mask": 0, "sam2_track": 0,
                      "lama": 0, "tracker_builds": 0}

    # -------------------------------------------------------------- episode control
    def reset(self, instruction: Optional[str] = None, parser: Optional[str] = None) -> dict:
        """Equivalent of ContrastImageGenerator.reset(); the instruction (if given) is parsed eagerly to fail fast.
        parser: None keeps the current mode (default "simpler", the authors' SIMPLER parser); "libero_goal" selects
        the frozen Setting-B table (parsers/libero_goal.json). The mode stays until the next reset that names one."""
        self.state = None
        if parser is not None:
            self._spec = None if parser == parsers.SIMPLER else parsers.load(parser)
            self.parser = parser
        if instruction is not None:
            self._start(instruction)
        return self.episode_summary()

    def _classes(self):
        return self._spec.classnames if self._spec is not None else self.classnames

    def _keep(self):
        return self._spec.keep_objects if self._spec is not None else KEEP_OBJECTS

    def _detect(self, image, prompts):
        if self._spec is None:
            return self.detector(image, prompts)   # SIMPLER: unchanged call (detector's own authors' aliases)
        return self.detector(image, prompts, aliases=self._spec.aliases)

    def _start(self, instruction: str) -> None:
        if self._spec is None:
            targets = parse_objects(instruction, self.get_all_parts)
        else:
            targets = self._spec.parse(instruction)
        prompts = targets + list(self._keep())
        for p in prompts:  # TrackingPredictorV2 would raise ValueError at add_new_mask (mask_predictors.py:463)
            if p not in self._classes():
                raise ValueError(f"prompt '{p}' is not in the class list (mask_predictors.py:11-14)")
        prev = self.state
        self.calls["tracker_builds"] += 1
        self.state = EpisodeState(instruction=instruction, targets=targets, prompts=prompts,
                                  tracker=self.tracker_factory(),
                                  first_detection_found={t: False for t in targets},
                                  tracker_none_steps={t: [] for t in targets},
                                  mask_present_steps={t: 0 for t in targets})
        if prev is not None:  # mid-episode instruction change (e.g. the apple-drawer subtask switch)
            self.state.step = prev.step
            self.state.rebuilds = prev.rebuilds + 1

    # -------------------------------------------------------------- one frame
    def step(self, image: np.ndarray, instruction: str) -> dict:
        t0 = time.perf_counter()
        if self.state is None or instruction != self.state.instruction:
            self._start(instruction)
        st = self.state
        h, w = image.shape[:2]
        timing = {}
        phase = "track" if st.tracking else "detect"
        if not st.tracking:
            t = time.perf_counter()
            obj_masks = self._detect(image, st.prompts)
            self.calls["gdino_sam2_detect"] += 1
            st.detect_attempts += 1
            timing["detect_ms"] = (time.perf_counter() - t) * 1e3
            if any(m is not None for m in obj_masks):
                t = time.perf_counter()
                st.tracker.load_first_frame(image)
                self.calls["sam2_first_frame"] += 1
                for m, p in zip(obj_masks, st.prompts):
                    if m is not None:
                        st.tracker.add_new_mask(0, self._classes().index(p) + 1, m)
                        self.calls["sam2_add_mask"] += 1
                st.tracking = True
                st.first_detection_step = st.step
                for name in st.targets:
                    st.first_detection_found[name] = obj_masks[st.prompts.index(name)] is not None
                timing["tracker_init_ms"] = (time.perf_counter() - t) * 1e3
                phase = "first_detection"
        else:
            t = time.perf_counter()
            obj_ids, logits = st.tracker.track(image)
            self.calls["sam2_track"] += 1
            timing["track_ms"] = (time.perf_counter() - t) * 1e3
            by_name = {}
            for idx, obj_id in enumerate(obj_ids):
                lg = logits[idx]
                lg = lg.cpu().numpy() if hasattr(lg, "cpu") else np.asarray(lg)
                m = (lg > 0).squeeze(0)
                by_name[self._classes()[obj_id - 1]] = m if m.any() else None
            obj_masks = [by_name.get(p, None) for p in st.prompts]

        n = len(st.targets)
        target_masks = obj_masks[:n]
        keep_masks = obj_masks[n:n + len(self._keep())]
        robot = obj_masks[st.prompts.index("robot")]
        union = M.target_union((h, w), target_masks, keep_masks)
        inpaint_region, paste_region = M.inpaint_regions(union, robot)
        t = time.perf_counter()
        inpainted = self.inpainter(image, inpaint_region)
        self.calls["lama"] += 1
        timing["lama_ms"] = (time.perf_counter() - t) * 1e3
        cf = M.compose(image, inpainted, paste_region)

        per_target = {}
        for name, m in zip(st.targets, target_masks):
            present = m is not None
            if present:
                st.mask_present_steps[name] += 1
            none_tracked = phase == "track" and not present
            if none_tracked:
                st.tracker_none_steps[name].append(st.step)
            per_target[name] = {"mask_present": present,
                                "area_fraction": float(m.sum()) / (h * w) if present else 0.0,
                                "tracker_none": none_tracked}
        timing["total_ms"] = (time.perf_counter() - t0) * 1e3
        out = {
            "step": st.step,
            "phase": phase,
            "targets": st.targets,
            "per_target": per_target,
            "robot_mask_present": robot is not None,
            "robot_area_fraction": float(robot.sum()) / (h * w) if robot is not None else 0.0,
            "paste_area_fraction": float(paste_region.sum()) / (h * w),
            "inpaint_area_fraction": float(inpaint_region.sum()) / (h * w),
            "cf_differs": bool((cf != image).any()),
            "cf_sha256": hashlib.sha256(np.ascontiguousarray(cf).tobytes()).hexdigest(),
            "timing_ms": timing,
            "episode": self.episode_summary(),
            "_cf": cf,
            "_masks": {"targets": target_masks, "robot": robot, "union": union, "paste": paste_region,
                       "inpaint": inpaint_region, "inpainted_raw": inpainted},
        }
        st.step += 1
        return out

    # -------------------------------------------------------------- evidence
    def episode_summary(self) -> dict:
        st = self.state
        if st is None:
            return {"started": False, "parser": self.parser, "calls": dict(self.calls)}
        robot_only = (st.first_detection_step is not None and not any(st.first_detection_found.values()))
        never_masked = [t for t in st.targets if st.first_detection_step is not None
                        and not st.first_detection_found[t]]
        return {
            "started": True,
            "parser": self.parser,
            "parser_sha256": self._spec.sha256 if self._spec is not None else None,
            "instruction": st.instruction,
            "targets": st.targets,
            "prompts": st.prompts,
            "steps_seen": st.step,
            "tracking": st.tracking,
            "first_detection_step": st.first_detection_step,
            "detect_attempts": st.detect_attempts,
            "first_detection_found": dict(st.first_detection_found),
            "robot_only_first_detection": robot_only,
            "targets_never_masked": never_masked,
            "mask_present_steps": dict(st.mask_present_steps),
            "tracker_none_steps": {k: list(v) for k, v in st.tracker_none_steps.items()},
            "instruction_rebuilds": st.rebuilds,
            "calls": dict(self.calls),
        }
