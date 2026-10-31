"""FastHarness controller (fast_harness/engine.py @ dfe070f) on HarnessStudio's EpisodeContext.

Every chunk: fresh policy proposal -> scheduler decides review (with a reason) or the zero-call fast path ->
the reviewer (pinned LLM) judges the last executed chunk, names the stage and approves / edits / replaces
the next chunk -> execute -> update the episode's experience. Errors start the bounded recovery state
machine (repair in place, return to the latest verified waypoint, then a reviewer-selected earlier one).

Experience across episodes comes only from the frozen `experience` artifact built by `hs prepare`;
within an episode the store is a fresh copy, updated from this episode's reviews and never written back.
Every store mutation is traced (`fh_mem` events) so the Preparer can rebuild the same store offline.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from typing import Any, Optional

import anyio
import numpy as np

from harnessstudio.core.errors import BudgetExceeded, RegistrationError

from . import kinematics as kin
from . import reviewer as rv
from .memory import Memory
from .scheduler import (Checkpoint, Config, MotionMonitor, Observation, Proposal, Recovery, Scheduler,
                        Transition)

FEATURES_VERSION = "rgb8-state-v1"
_FIRST = ('{"stage":"build-tower","last_outcome":"unknown","error_type":"none","evidence":"No execution yet.",'
          '"response":{"mode":"student","steps":6,"reason":"Start with the fresh proposal.",'
          '"edit":{"left":{"delta_position":[0,0,0],"delta_rotation_vector":[0,0,0],"gripper":"keep"},'
          '"right":{"delta_position":[0,0,0],"delta_rotation_vector":[0,0,0],"gripper":"keep"}},'
          '"target":{"left":{"position":[0,0,0],"quaternion_wxyz":[1,0,0,0],"gripper_closed":false},'
          '"right":{"position":[0,0,0],"quaternion_wxyz":[1,0,0,0],"gripper_closed":false}},'
          '"assessment":{"task_progress":{"verified_completed":[],"currently_attempting":"start","remaining":["all"]},'
          '"current_subgoal":"start","execution_status":"not_started","execution_evidence":"none",'
          '"expected_next_intent":"approach","predicted_next_intent":"approach","intent_status":"aligned",'
          '"intent_evidence":"proposal moves toward the goal"}},'
          '"checkpoint":null,"recovery_status":"continue","selected_checkpoint_id":null}')
_LATER = (_FIRST.replace('"last_outcome":"unknown"', '"last_outcome":"ok"')
          .replace('"execution_status":"not_started"', '"execution_status":"progressing"'))


def _box8(image: np.ndarray) -> list:
    """8x8 area-averaged RGB in [0, 1] (FastHarness: PIL BOX resize to 8x8)."""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = a[..., :3]
    rows = np.array_split(np.arange(a.shape[0]), 8)
    cols = np.array_split(np.arange(a.shape[1]), 8)
    out = [a[np.ix_(r, c)].reshape(-1, 3).mean(axis=0) / 255.0 for r in rows for c in cols]
    return [float(v) for px in out for v in px]


class _LoggedMemory:
    """Forwards to Memory and traces every mutation for the Preparer (`fh_mem` events)."""

    def __init__(self, memory: Memory, ctx) -> None:
        self._m, self._ctx = memory, ctx

    def __getattr__(self, name):
        return getattr(self._m, name)

    def learn_stage(self, stage, features):
        self._ctx.record("fh_mem", op="learn_stage", namespace=self._m.namespace, stage=stage, features=list(features))
        return self._m.learn_stage(stage, features)

    def record(self, **kw):
        args = dict(kw)
        args["obs_step"] = args.pop("step")
        args["record_kind"] = args.pop("kind")
        args["features"] = list(args["features"])
        self._ctx.record("fh_mem", op="record", namespace=self._m.namespace, **args)
        return self._m.record(**kw)

    def recovered(self, case_ids, evidence):
        self._ctx.record("fh_mem", op="recovered", namespace=self._m.namespace, case_ids=sorted(case_ids),
                         evidence=evidence)
        return self._m.recovered(case_ids, evidence)

    def record_span(self, stage, chunks):
        self._ctx.record("fh_mem", op="record_span", namespace=self._m.namespace, stage=stage, chunks=chunks)
        return self._m.record_span(stage, chunks)


def install_completion_tokens_operator(ctx, timeout: float) -> list:
    """TEMPORARY SHIM (core request #1 in docs/results/fast_harness.md).

    HarnessStudio's OpenAICompatibleOperator always sends `max_tokens`, which OpenAI's gpt-6.1-sol rejects
    ("use max_completion_tokens"). For operators on the OpenAI preset this swaps in a subclass that sends
    the same cap as `max_completion_tokens`. Pinning (/models check, Astra refusal), budgets, accounting and
    logging stay in the unchanged LLMTool / base class. Scripted (tier 1) operators are untouched.
    """
    from harnessstudio.tools import llm as hs_llm

    class CompletionTokensOperator(hs_llm.OpenAICompatibleOperator):
        def complete(self, messages, max_tokens):
            self.verify()
            payload = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens,
                       "temperature": self.temperature}
            if self.seed is not None:
                payload["seed"] = int(self.seed)
            for k, v in self.extra_body.items():
                payload.setdefault(k, v)
            body = self._request("/chat/completions", payload)
            if any(w in str(body.get("model", "")).lower() for w in hs_llm.FORBIDDEN_MODEL_WORDS):
                raise RuntimeError("Provider answered with a forbidden model")
            return body["choices"][0]["message"].get("content") or "", body.get("usage") or {}

    tool = ctx.tools.get("llm")
    swapped = []
    for name, op in list(getattr(tool, "_ops", {}).items()):
        if type(op) is hs_llm.OpenAICompatibleOperator and op.base_url == hs_llm.PRESETS["openai"]:
            tool._ops[name] = CompletionTokensOperator(op.model, op.base_url, op.api_key_env, op.temperature,
                                                       timeout=timeout, seed=op.seed, extra_body=op.extra_body)
            swapped.append(name)
    return swapped


class FastHarnessController:
    TIER1_LLM_RESPONSES = [_FIRST] + [_LATER] * 63

    def __init__(self, max_interval: int = 6, min_samples: int = 4, cooldown_successes: int = 3,
                 max_unreviewed_steps: int = 60, match_threshold: float = 0.06, ambiguity_margin: float = 0.01,
                 error_radius: float = 0.04, audit_probability: float = 0.05, stall_chunks: int = 3,
                 attempt_chunks: int = 8, recovery_chunks: int = 24, max_checkpoints: int = 64,
                 max_episode_chunks: int = 1000, confidence_cap: float = 1.0, contact_gating: bool = True,
                 contact_low_z: float = 0.95, contact_descent_m: float = 0.05, commit_gating: bool = True,
                 commit_cap: int = 6, commit_min_confidence: float = 0.9, persist_error_review: bool = True,
                 memory_decay: float = 1.0, cold_full_review: bool = True, always_review: bool = False,
                 max_proposal_steps: int = 15, schema_retries: int = 2, transport_retries: int = 3,
                 transport_wait: float = 15.0, llm_timeout: float = 240.0, image_max_side: Optional[int] = None,
                 max_completion_tokens: int = 4096, joint_step_threshold_rad: float = 0.35,
                 first_joint_step_threshold_rad: float = 0.5, gripper_open_value: float = 1.0,
                 gripper_closed_value: float = 0.0, seed: int = 0, operator: str = "reviewer") -> None:
        self.p = dict(locals())
        self.p.pop("self")
        self.cfg = Config(max_interval=max_interval, min_samples=min_samples, cooldown_successes=cooldown_successes,
                          max_unreviewed_steps=max_unreviewed_steps, match_threshold=match_threshold,
                          ambiguity_margin=ambiguity_margin, error_radius=error_radius,
                          audit_probability=audit_probability, stall_chunks=stall_chunks,
                          attempt_chunks=attempt_chunks, recovery_chunks=recovery_chunks,
                          max_checkpoints=max_checkpoints, max_episode_chunks=max_episode_chunks,
                          always_review=always_review, confidence_cap=confidence_cap, contact_gating=contact_gating,
                          contact_low_z=contact_low_z, contact_descent_m=contact_descent_m,
                          commit_gating=commit_gating, commit_cap=commit_cap,
                          commit_min_confidence=commit_min_confidence, persist_error_review=persist_error_review,
                          seed=seed)

    # -- helpers ---------------------------------------------------------------------------
    def namespace(self, instruction: str, model: str) -> str:
        """FastHarness keys experience by task + policy + features + reviewer + scheduler config."""
        keys = ("max_interval", "min_samples", "cooldown_successes", "match_threshold", "ambiguity_margin",
                "error_radius", "memory_decay", "contact_gating", "commit_gating", "persist_error_review",
                "max_proposal_steps", "gripper_open_value")
        identity = {"instruction": instruction, "features": FEATURES_VERSION, "teacher": model,
                    "config": {k: self.p[k] for k in keys}}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def _features(self, images: dict, state: Optional[np.ndarray]) -> tuple:
        feats = []
        for camera in rv.camera_order(images):
            feats.extend(_box8(images[camera]))
        if state is not None:
            for i, v in enumerate(state):
                if self._robodojo and i in (6, 13):
                    feats.append(min(1.0, max(0.0, float(v) / self.p["gripper_open_value"])))
                else:
                    feats.append(min(1.0, max(0.0, (float(v) + math.pi) / (2 * math.pi))))
        return tuple(round(f, 6) for f in feats)

    def _observe(self, ctx) -> Observation:
        pub = ctx.observe()
        images = {k: np.asarray(v) for k, v in (pub.get("images") or {}).items()}
        state = pub.get("state")
        state = None if state is None else np.asarray(state, dtype=np.float64).reshape(-1)
        payload = {"instruction": ctx.instruction, "remaining_steps": ctx.remaining}
        if state is not None:
            payload["current_state"] = [round(float(v), 5) for v in state]
            if self._robodojo:
                payload["current_eef"] = kin.current_eef(state, self.p["gripper_open_value"])
        return Observation("episode", ctx.steps, self._features(images, state), payload, images)

    def _diagnostics(self, rows: np.ndarray, state: Optional[np.ndarray]) -> tuple:
        """RoboDojoBackend._diagnostics: joint jumps -> review; nonfinite or gripper outside [0, 1] -> blocked."""
        warnings, blocked = [], False
        if not np.isfinite(rows).all():
            return ["Invalid action shape or nonfinite values"], True, {"shape": list(rows.shape), "finite": False}
        diag = {"shape": list(rows.shape), "finite": True, "status": "computed"}
        if self._robodojo:
            joints = [i for i in range(14) if i not in (6, 13)]
            successive = float(np.max(np.abs(np.diff(rows[:, joints], axis=0)))) if len(rows) > 1 else 0.0
            diag["max_joint_step_rad"] = round(successive, 5)
            if successive > self.p["joint_step_threshold_rad"]:
                warnings.append(f"Actual successive joint jump={successive:.6g} rad exceeds "
                                f"{self.p['joint_step_threshold_rad']:.6g} rad")
            if state is not None:
                first = float(np.max(np.abs(rows[0, joints] - state[joints])))
                if first > self.p["first_joint_step_threshold_rad"]:
                    warnings.append(f"Measured state to first action joint jump={first:.6g} rad exceeds "
                                    f"{self.p['first_joint_step_threshold_rad']:.6g} rad")
            grip = rows[:, [6, 13]] / self.p["gripper_open_value"]
            diag["opening_range"] = [[round(float(grip[:, i].min()), 4), round(float(grip[:, i].max()), 4)]
                                     for i in range(2)]
            if (grip < -1e-6).any() or (grip > 1 + 1e-6).any():
                warnings.append("Actual gripper opening is outside [0,1]")
                blocked = True
        return warnings, blocked, diag

    async def _infer(self, ctx, observation: Observation, index: int) -> Proposal:
        if "seeded_sampling" in ctx.policy.capabilities:
            chunk = (await ctx.policy.sample(1))[0].actions
        else:
            chunk = await ctx.policy.act()
        chunk = np.array(chunk, dtype=np.float32)
        raw_grip = None
        if self._robodojo:
            # RoboDojo's take_action clips every gripper command to [0, 1] (src/eval_client/eval_env.py:389);
            # clip the proposal the same way so the reviewer and the validator see the command that executes
            # (pi0.5 overshoots slightly; FastHarness's validator blocks out-of-range gripper commands).
            g = [kin.GRIPPER_INDEX["left"], kin.GRIPPER_INDEX["right"]]
            raw_grip = [[round(float(chunk[:, i].min()), 4), round(float(chunk[:, i].max()), 4)] for i in g]
            open_v = self.p["gripper_open_value"]
            chunk[:, g] = np.clip(chunk[:, g], 0.0, open_v)
        max_steps = max(1, min(self.p["max_proposal_steps"], ctx.remaining, len(chunk)))
        rows = chunk[:max_steps]
        state = observation.payload.get("current_state")
        state = None if state is None else np.asarray(state, dtype=np.float64)
        warnings, blocked, diag = self._diagnostics(rows.astype(np.float64), state)
        if raw_grip is not None:
            diag["raw_opening_range_before_env_clip"] = raw_grip
        payload = {"action_diagnostics": diag}
        if self._robodojo:
            payload["current_eef"] = observation.payload.get("current_eef")
            payload["student_eef_trajectory"] = [
                {arm: kin.eef_pose(arm, r[kin.ARM_SLICES[arm]], r[kin.GRIPPER_INDEX[arm]] / self.p["gripper_open_value"])
                 for arm in kin.ARMS} for r in rows.astype(np.float64)]
        else:
            payload["student_actions"] = [[round(float(v), 4) for v in r] for r in rows]
        return Proposal(f"proposal-{index:04d}", observation, payload, chunk, max_steps, tuple(warnings), blocked)

    async def _llm(self, ctx, messages: list) -> Optional[str]:
        for attempt in range(self.p["transport_retries"] + 1):
            try:
                return await ctx.tools["llm"](messages, operator=self.p["operator"],
                                              max_tokens=self.p["max_completion_tokens"])
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - transport: a review has no side effects, resend
                ctx.record("fh_transport_error", attempt=attempt + 1, error=f"{type(exc).__name__}: {exc}"[:300])
                if attempt >= self.p["transport_retries"]:
                    return None
                await anyio.sleep(self.p["transport_wait"])
        return None

    async def _review(self, ctx, request: dict) -> Optional[rv.Review]:
        """One review with bounded schema retries (TimedRetryReviewer schema_retries); None if all fail."""
        attempt_request = dict(request, retry_feedback=None)
        for attempt in range(self.p["schema_retries"] + 1):
            started = time.monotonic()
            text = await self._llm(ctx, rv.messages(attempt_request, self.p["image_max_side"]))
            if text is None:
                ctx.record("fh_review_failed", reason="transport", attempt=attempt + 1)
                return None
            try:
                review = rv.parse_review(text)
                rv.validate(review, request, takeover=self._robodojo)
            except rv.ReviewError as err:
                ctx.record("fh_review_rejected", code=err.code, path=list(err.path), attempt=attempt + 1,
                           seconds=round(time.monotonic() - started, 3))
                attempt_request = dict(request, retry_feedback=rv.retry_feedback(err))
                continue
            self.metrics["teacher_calls"] += 1
            return review
        ctx.record("fh_review_failed", reason="schema_retries_exhausted")
        return None

    # -- the episode -----------------------------------------------------------------------
    async def run(self, ctx) -> None:
        self._robodojo = kin.roboDojo_like(ctx.embodiment.action_dim, ctx.embodiment.control_mode)
        swapped = install_completion_tokens_operator(ctx, self.p["llm_timeout"])
        tool = ctx.tools["llm"]
        op = tool._ops.get(self.p["operator"])
        model = getattr(op, "model", "scripted")
        extra = getattr(op, "extra_body", None)
        if extra is not None and extra.get("response_format") != rv.RESPONSE_FORMAT:
            raise RegistrationError("The operator's response_format must be the reviewer's strict Review schema")
        namespace = self.namespace(ctx.instruction, model)
        memory = Memory(namespace, decay=self.p["memory_decay"])
        source = "empty"
        path = ctx.artifacts.get("experience")
        if path:
            data = json.loads(open(path, encoding="utf-8").read())
            if namespace in data.get("memories", {}):
                memory = Memory.from_json(data["memories"][namespace])
                source = "artifact"
        cold = memory.empty()
        cfg = Config(**{**asdict(self.cfg), "episode_number": 1 if (cold and self.p["cold_full_review"]) else None})
        ctx.record("fh_start", namespace=namespace, memory_source=source, cold=cold, memory=memory.summary(),
                   robodojo=self._robodojo, operator_shim=swapped, model=model)
        self.memory = _LoggedMemory(memory, ctx)
        self.metrics = dict(chunks=0, teacher_calls=0, automatic_chunks=0, degraded_chunks=0, recovery_events=0,
                            checkpoints=0, passthrough_chunks=0)
        await self._loop(ctx, cfg)

    async def _loop(self, ctx, cfg: Config) -> None:
        memory = self.memory
        scheduler, monitor = Scheduler(cfg), MotionMonitor(cfg.stall_chunks)
        recovery = Recovery(cfg.attempt_chunks, cfg.recovery_chunks)
        checkpoints: list = []
        observation = self._observe(ctx)
        last_review_step, previous_stage = None, None
        chunks_since_review, force_review, stage_run = 0, False, 0
        previous: Optional[Transition] = None
        unreviewed: list = []
        passthrough = False
        index = 0
        while not ctx.ended:
            if self.metrics["chunks"] >= cfg.max_episode_chunks:
                ctx.stop("fast_harness_episode_decision_budget")
                return
            match = memory.match(observation.features, cfg.match_threshold, cfg.ambiguity_margin)
            index += 1
            proposal = await self._infer(ctx, observation, index)
            if passthrough:
                reason, gap, step_limit = None, 1, cfg.max_unreviewed_steps
            else:
                reason, gap, step_limit = scheduler.reason(
                    memory, match, observation, proposal, last_review_step=last_review_step,
                    chunks_since_review=chunks_since_review, previous_stage=previous_stage,
                    recovery_active=recovery.active, force_review=force_review, stalled=monitor.stalled)
                if cfg.episode_number == 1:
                    reason = reason or "first_episode_review"
            ctx.record("fh_schedule", obs_step=observation.step, stage=match.stage, reason=reason, interval=gap,
                       step_limit=step_limit, match_distance=match.distance if math.isfinite(match.distance) else None,
                       warnings=list(proposal.warnings), passthrough=passthrough, **self.metrics)
            stage = match.stage
            source = "harness_auto"
            response = None
            if reason:
                source = "teacher"
                account_previous = True
                for _ in range(8):
                    options = recovery.candidates(checkpoints) if recovery.active else tuple(checkpoints)
                    if recovery.target and recovery.target not in options:
                        options = (*options, recovery.target)
                    request = {"observation": observation, "proposal": proposal, "previous": previous,
                               "stage_hint": stage, "known_stages": memory.stages(),
                               "errors": memory.errors(stage), "checkpoints": tuple(options),
                               "recovery": recovery.context(), "unreviewed": tuple(unreviewed)}
                    review = await self._review(ctx, request)
                    if review is None:   # degraded: unreviewed student prefix, review again next chunk
                        source, force_review = "reviewer_failure_fallback", True
                        self.metrics["degraded_chunks"] += 1
                        response = {"mode": "student", "steps": min(proposal.max_steps, cfg.max_unreviewed_steps)}
                        break
                    ctx.record("fh_review", obs_step=observation.step, reason=reason, stage=review.stage,
                               last_outcome=review.last_outcome, error_type=review.error_type,
                               evidence=review.evidence, mode=review.response["mode"], steps=review.response["steps"],
                               execution_status=review.response["assessment"]["execution_status"],
                               intent_status=review.response["assessment"]["intent_status"],
                               checkpoint=review.checkpoint, recovery_status=review.recovery_status,
                               selected_checkpoint_id=review.selected_checkpoint_id, recovery=recovery.context())
                    stage = review.stage
                    was_active = recovery.active
                    if account_previous:
                        if review.response["assessment"]["intent_status"] == "misaligned":
                            case_id = memory.record(episode=observation.episode_id, step=observation.step,
                                                    kind="intent", stage=review.stage, features=observation.features,
                                                    outcome="error", evidence=review.evidence, student=False,
                                                    error_type="misaligned_intent")
                            if case_id:
                                recovery.error_cases.add(case_id)
                        self._remember(ctx, review, previous, observation, was_active, recovery, checkpoints, cfg)
                        account_previous = False
                        unreviewed.clear()
                    last_review_step, chunks_since_review = observation.step, 0
                    memory.learn_stage(stage, observation.features)
                    try:
                        if was_active:
                            changed = recovery.update(review, checkpoints)
                            if recovery.phase == "normal":
                                memory.recovered(recovery.error_cases, review.evidence)
                                recovery.error_cases.clear()
                        elif (review.last_outcome == "error"
                              or review.response["assessment"]["intent_status"] == "misaligned"
                              or review.response["assessment"]["execution_status"] == "failed"):
                            recovery.start(observation.step, "Resolve the observed problem and resume: "
                                           + review.response["assessment"]["expected_next_intent"]
                                           + ". Original evidence: " + review.evidence)
                            changed = True
                            self.metrics["recovery_events"] += 1
                        else:
                            changed = False
                    except ValueError as exc:  # ineligible checkpoint / inconsistent recovery: FastHarness raises
                        ctx.record("fh_recovery_error", error=str(exc))
                        recovery.phase = "exhausted"
                        changed = True
                    if recovery.phase == "exhausted":
                        passthrough = True
                        ctx.record("fh_recovery_exhausted_passthrough", obs_step=observation.step)
                        response = {"mode": "student", "steps": proposal.max_steps}
                        source = "recovery_exhausted_passthrough"
                        break
                    if changed:
                        ctx.record("fh_recovery_transition", obs_step=observation.step, recovery=recovery.context())
                        reason = "recovery_replan"
                        continue
                    response = dict(review.response, steps=min(review.response["steps"], cfg.max_unreviewed_steps))
                    force_review = (review.last_outcome == "unknown"
                                    or response["assessment"]["intent_status"] == "uncertain")
                    break
                else:
                    passthrough = True
                    response = {"mode": "student", "steps": proposal.max_steps}
                    source = "recovery_exhausted_passthrough"
            else:
                if passthrough:
                    steps = proposal.max_steps
                    source = "recovery_exhausted_passthrough"
                else:
                    steps = min(proposal.max_steps, step_limit - (observation.step - last_review_step))
                if steps <= 0:
                    raise RuntimeError("automatic_review_budget")
                response = {"mode": "student", "steps": steps}
            steps = int(response["steps"])
            if not 1 <= steps <= proposal.max_steps:
                raise RuntimeError("response_horizon")
            if response["mode"] == "student":
                actions = proposal.actions[:steps]
                info = None
            else:
                actions, info = kin.takeover_actions(observation.payload["current_state"], response, steps,
                                                     self.p["gripper_open_value"], self.p["gripper_closed_value"])
            ctx.record("fh_execute", obs_step=observation.step, request_id=proposal.request_id, source=source,
                       mode=response["mode"], steps=steps, takeover=info)
            executed = await ctx.execute(actions)
            if executed == 0:
                return
            after = self._observe(ctx)
            previous = Transition(observation, after, proposal.request_id, stage, response["mode"], source, executed)
            unreviewed.append(previous)
            monitor.observe(previous)
            recovery.executed()
            self.metrics["chunks"] += 1
            self.metrics["automatic_chunks"] += int(source == "harness_auto")
            self.metrics["passthrough_chunks"] += int(source == "recovery_exhausted_passthrough")
            if source == "reviewer_failure_fallback":
                previous_stage, stage_run = None, 0
            elif cfg.commit_gating:
                if previous_stage is not None and stage != previous_stage:
                    memory.record_span(previous_stage, stage_run)
                    stage_run = 1
                else:
                    stage_run += 1
            previous_stage = None if source == "reviewer_failure_fallback" else stage
            observation = after
            chunks_since_review += 1

    def _remember(self, ctx, review, previous, observation, recovery_active, recovery, checkpoints, cfg) -> None:
        if previous is None:
            return
        case_id = self.memory.record(episode=previous.before.episode_id, step=previous.before.step, kind="execution",
                                     stage=previous.stage or review.stage, features=previous.before.features,
                                     outcome=review.last_outcome, evidence=review.evidence,
                                     student=previous.mode == "student" and not recovery_active,
                                     error_type=review.error_type)
        if case_id:
            recovery.error_cases.add(case_id)
        if review.checkpoint and review.last_outcome == "ok" and not recovery_active:
            point = Checkpoint(f"{observation.episode_id}:{observation.step}", observation, review.stage,
                               review.checkpoint["goal"], tuple(review.checkpoint["prerequisites"]))
            if not checkpoints or checkpoints[-1].id != point.id:
                checkpoints.append(point)
                checkpoints[:] = checkpoints[-cfg.max_checkpoints:]
                self.metrics["checkpoints"] += 1
                ctx.record("fh_checkpoint", id=point.id, stage=point.stage, goal=point.goal, obs_step=observation.step)
