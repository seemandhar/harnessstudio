"""PCD controller (HarnessStudio v2 controller method, `null: shadow`, on the EXT openvla-contrast `logits` readout).

Per environment step (Koorye/PCD @ 31b027e, simpler_env/.../parallel_inference.py:198,208,223: the counterfactual is
built from the current frame, then the policy steps once):
  1. perception: `/step(frame, instruction)` on the method's own service session -> counterfactual frame ô at the
     input resolution (+ mask evidence). `/reset(instruction)` once at episode start (generator reset per episode,
     parallel_inference.py:181-183); a changed instruction mid-episode is passed on and the service rebuilds its
     tracker (contrast_image_generator.py:76-81).
  2. stream A: `ctx.policy.sample(1, readouts=["logits"])` on the real observation: the native greedy decode
     (bitwise the native action) + per-position raw float32 logits.
  3. stream B: the same request with the public image override `images[first_key] = ô` (vla-eval's OpenVLA server
     reads only the FIRST image, model_servers/openvla.py:88-89). Separate batch-1 requests, never batched.
  4. combine (hs_pcd.combine.pcd_code_combine, openvla_contrast.py:40-58): (1+a)*s - a*c on positions 0..5 when B
     has 7 rows, gripper from A, the authors' <7-row fallbacks, full-width argmax; alpha = 0.8 (frozen).
  5. execute the combined tokens through the server's bin table (bitwise the server's de-tokenization).
Null (`null: shadow`): steps 1-4 run unchanged on the shadow policy session and the method's own perception session
(on the native trajectory), and the framework executes the native action (decodes per step: baseline 1, candidate 2,
null 3; perception: candidate 1 = null 1).

Evidence (SETTING.md v2 edit 3): a `pcd` trace event per step (`env_step`, perception phase, per-target mask presence
and area, `any_target_present` (any non-robot target with area > 0; Addendum 2 E-4), tracker None targets, cf_differs, contrast skipped (B < 7 rows), stream-A fallback (A < 7 rows), per-dim
changed tokens vs the native token, non-action-vocabulary selections, top1-top2 logit gaps of s and of the combined
scores), and a running summary in accounting.extra["pcd"] (copied into the episode record), incl.
`target_active_steps` and `mechanism_inactive` (E-4: active in fewer than 50% of the steps). Tiled real|counterfactual
frames go to `tile_dir/<session>/` (the run's artifact directory, never harnessstudio/).

Never read: reward, success, simulator state, seeds, case ids or the arm (INVARIANTS 2). The perception session id is
a uuid4 generated here; the trace records it so the evaluator can map tiles to episodes.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .combine import pcd_code_final, table_action, token_stats, top2_gap
from .perception import PerceptionClient, StubPerception

PARSERS = ("simpler", "libero_goal")   # the service's /reset `parser` modes (SETTING.md Addendum 4)
URL_ENV = "HS_PCD_PERCEPTION_URL"     # deployment detail: overrides the card's perception_url
TILE_ENV = "HS_PCD_TILE_DIR"          # deployment detail: overrides the card's tile_dir


def _first_image(obs: Mapping[str, Any], image_key: Optional[str]):
    images = obs.get("images")
    if not isinstance(images, Mapping) or not images:
        raise ValueError("PCD needs a public 'images' dict (the policy reads its first image)")
    key = image_key or next(iter(images))
    if key not in images:
        raise ValueError(f"image key {key!r} not in the public images {sorted(images)}")
    if image_key and next(iter(images)) != key:
        raise ValueError(f"image key {key!r} is not the first image: the policy would not see the counterfactual")
    return key, np.asarray(images[key])


def any_target_present(ev: Mapping[str, Any]) -> bool:
    """SETTING.md Addendum 2 E-4: a step is active when ANY non-robot target mask has area fraction > 0 (the
    service's per_target excludes the robot, service/pcd_perception/engine.py:146-163); multi-target safe."""
    return any(float((t or {}).get("area_fraction") or 0.0) > 0.0 for t in (ev.get("per_target") or {}).values())


def mechanism_inactive(active_steps: int, steps: int) -> Optional[bool]:
    """E-4: the episode is mechanism-inactive if a target mask is present in fewer than 50% of its steps."""
    return None if steps <= 0 else bool(active_steps < 0.5 * steps)


def _tile(real: np.ndarray, cf: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(real, dtype=np.uint8), np.asarray(cf, dtype=np.uint8)], axis=1)  # real | cf


class PCDController:
    def __init__(self, alpha: float = 0.8, perception_url: str = "http://127.0.0.1:8141",
                 perception_timeout: float = 600.0, image_key: Optional[str] = None,
                 instruction_source: str = "observation", parser: str = "service", tile_dir: Optional[str] = None,
                 tile_every: int = 10, require_tiles: bool = True, action_dim: int = 7,
                 token_dim: Optional[int] = None, perception_parser: str = "simpler") -> None:
        self.alpha = float(alpha)
        self.perception_url = str(perception_url)
        self.perception_timeout = float(perception_timeout)
        self.image_key = image_key
        if instruction_source != "observation":
            raise ValueError("instruction_source must be 'observation' (the public instruction, ctx.instruction)")
        if parser != "service":
            raise ValueError("parser must be 'service': the perception service owns the frozen parser/alias table")
        self.instruction_source, self.parser = instruction_source, parser
        if perception_parser not in PARSERS:
            raise ValueError(f"perception_parser must be one of {PARSERS}, not {perception_parser!r}")
        self.perception_parser = perception_parser
        self.tile_dir = tile_dir
        self.tile_every = int(tile_every)
        if self.tile_every < 1:
            raise ValueError("tile_every must be >= 1")
        self.require_tiles = bool(require_tiles)
        self.action_dim = int(action_dim)
        self.token_dim = None if token_dim is None else int(token_dim)

    # ---- deployment ------------------------------------------------------------------------------------------
    def _client(self, ctx):
        # Tier 1 runs HarnessStudio's synthetic `fake` benchmark (contract control_mode "fake", a public contract
        # property): only there is the stub used. Every other benchmark needs the real service; errors raise.
        if getattr(ctx.embodiment, "control_mode", None) == "fake":
            return StubPerception()
        return PerceptionClient(os.environ.get(URL_ENV, self.perception_url), timeout=self.perception_timeout)

    def _tiles(self, ctx, client) -> Optional[Path]:
        root = os.environ.get(TILE_ENV, self.tile_dir or "")
        if not root:
            if self.require_tiles and client.kind != "stub":
                raise RuntimeError(f"PCD needs a tile directory ({TILE_ENV} or tile_dir; SETTING.md v2 edit 3)")
            return None
        path = Path(root).resolve()
        hs_root = Path(__file__).resolve().parents[3]          # .../harnessstudio (methods/pcd/hs_pcd/controller.py)
        if path == hs_root or hs_root in path.parents:
            raise RuntimeError(f"tiles must go to the run's artifact directory, never under {hs_root}")
        out = path / client.session
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ---- episode ----------------------------------------------------------------------------------------------
    async def run(self, ctx) -> None:
        client = self._client(ctx)
        # The summary exists before any service call, so a refused episode still carries extra.pcd (with `error`).
        summary = ctx.accounting.extra.setdefault("pcd", {
            "perception": client.kind, "session": client.session, "steps": 0, "alpha": self.alpha,
            "first_detection_step": None, "mask_present_steps": {}, "area_fraction_sum": {},
            "tracker_none_steps": {}, "cf_differs_steps": 0, "contrast_skipped": 0, "a_fallback": 0,
            "changed_tokens_per_dim": [0] * self.action_dim, "changed_steps": 0,
            "non_action_vocab_per_dim": [0] * self.action_dim, "perception_ms": 0.0, "tiles": 0,
            "instruction_changes": 0, "target_active_steps": 0, "mechanism_inactive": None,
            "perception_parser": self.perception_parser, "service_parser": None,
            "service_parser_sha256": None, "error": None})
        env_step = 0
        last_ep: dict = {}
        try:
            tiles = self._tiles(ctx, client)
            health = await asyncio.to_thread(client.health)
            instruction = ctx.instruction
            reset = await asyncio.to_thread(client.reset, instruction, self.perception_parser)
            ctx.accounting.tool_calls += 1
            served = (reset or {}).get("parser")
            summary["service_parser"], summary["service_parser_sha256"] = served, (reset or {}).get("parser_sha256")
            if served is None and self.perception_parser != "simpler":
                raise RuntimeError(f"the perception service did not confirm parser {self.perception_parser!r} "
                                   "(a service without parser modes would silently run the SIMPLER table)")
            if served is not None and served != self.perception_parser:
                raise RuntimeError(f"perception service parser {served!r} != requested {self.perception_parser!r}")
            ctx.record("pcd_start", session=client.session, perception=client.kind,
                       service_version=health.get("version"), service_device=health.get("device"),
                       sam2_C=health.get("sam2_C"), service_weights=health.get("weights"),
                       tile_dir=str(tiles) if tiles else None, alpha=self.alpha)
            while not ctx.ended:
                obs = ctx.observe()
                key, frame = _first_image(obs, self.image_key)
                if ctx.instruction != instruction:
                    instruction = ctx.instruction
                    summary["instruction_changes"] += 1
                cf, ev = await asyncio.to_thread(client.step, frame, instruction)
                ctx.accounting.tool_calls += 1
                cf = np.asarray(cf)
                if cf.shape != frame.shape or cf.dtype != np.uint8:
                    raise RuntimeError(f"counterfactual {cf.shape}/{cf.dtype} != frame {frame.shape}/uint8")
                images = dict(obs["images"])
                images[key] = cf
                [real] = await ctx.policy.sample(1, readouts=["logits"])
                [cfp] = await ctx.policy.sample(1, obs={"images": images}, readouts=["logits"])
                s, c = real.extras["logits"], cfp.extras["logits"]
                tokens, info, final = pcd_code_final(s, c, self.alpha, self.action_dim, self.token_dim)
                action = table_action(real.extras, tokens)
                ts = token_stats(tokens, real.extras["logits_tokens"], real.extras)
                self._account(summary, ev, info, ts)
                last_ep = ev.get("episode") or last_ep
                ctx.record("pcd", env_step=env_step, phase=ev.get("phase"), per_target=ev.get("per_target"),
                           any_target_present=any_target_present(ev),
                           robot_mask_present=ev.get("robot_mask_present"), cf_differs=ev.get("cf_differs"),
                           paste_area_fraction=ev.get("paste_area_fraction"), cf_sha256=ev.get("cf_sha256"),
                           contrast_skipped=not info["contrasted"], a_fallback=info["a_fallback"],
                           a_rows=info["a_rows"], b_rows=info["b_rows"], tokens=[int(t) for t in tokens],
                           native_tokens=[int(t) for t in real.extras["logits_tokens"]],
                           changed_dims=ts["changed_dims"], non_action_vocab_dims=ts["non_action_vocab_dims"],
                           gap_s=[round(g, 5) for g in top2_gap(np.asarray(s)[:-1])],
                           gap_f=[round(g, 5) for g in top2_gap(final[:-1])],
                           perception_ms=(ev.get("timing_ms") or {}).get("total_ms"))
                if tiles is not None and (env_step % self.tile_every == 0 or ev.get("phase") == "first_detection"):
                    await asyncio.to_thread(self._save_tile, tiles, env_step, frame, cf)
                    summary["tiles"] += 1
                env_step += 1
                await ctx.execute(action[None, :])
        except Exception as exc:   # recorded, then re-raised: the framework marks the episode invalid (method_error)
            summary["error"] = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            summary["service_episode"] = {k: last_ep.get(k) for k in (
                "first_detection_step", "robot_only_first_detection", "targets_never_masked", "mask_present_steps",
                "tracker_none_steps", "detect_attempts", "instruction_rebuilds", "calls") if k in last_ep}
            # Synchronous on purpose: the framework may cancel the controller task when the benchmark ends the
            # episode, and an await inside `finally` would be cancelled before the session is closed.
            client.close()

    @staticmethod
    def _account(summary: dict, ev: dict, info: dict, ts: dict) -> None:
        summary["steps"] += 1
        summary["target_active_steps"] += int(any_target_present(ev))
        summary["mechanism_inactive"] = mechanism_inactive(summary["target_active_steps"], summary["steps"])
        ep = ev.get("episode") or {}
        if ep.get("first_detection_step") is not None and summary["first_detection_step"] is None:
            summary["first_detection_step"] = ep["first_detection_step"]
        for name, t in (ev.get("per_target") or {}).items():
            summary["mask_present_steps"][name] = summary["mask_present_steps"].get(name, 0) + int(bool(t.get("mask_present")))
            summary["area_fraction_sum"][name] = summary["area_fraction_sum"].get(name, 0.0) + float(t.get("area_fraction") or 0.0)
            summary["tracker_none_steps"][name] = summary["tracker_none_steps"].get(name, 0) + int(bool(t.get("tracker_none")))
        summary["cf_differs_steps"] += int(bool(ev.get("cf_differs")))
        summary["contrast_skipped"] += int(not info["contrasted"])
        summary["a_fallback"] += int(info["a_fallback"])
        for d in ts["changed_dims"]:
            if d < len(summary["changed_tokens_per_dim"]):
                summary["changed_tokens_per_dim"][d] += 1
        summary["changed_steps"] += int(bool(ts["changed_dims"]))
        for d in ts["non_action_vocab_dims"]:
            if d < len(summary["non_action_vocab_per_dim"]):
                summary["non_action_vocab_per_dim"][d] += 1
        summary["perception_ms"] += float((ev.get("timing_ms") or {}).get("total_ms") or 0.0)

    @staticmethod
    def _save_tile(directory: Path, env_step: int, real: np.ndarray, cf: np.ndarray) -> None:
        from PIL import Image
        Image.fromarray(_tile(real, cf)).save(directory / f"step{env_step:04d}_real-cf.png")
