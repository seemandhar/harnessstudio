"""GR00T N1.5 Human300 (checkpoint-120000) on RoboCasa365 panda_omron (spec §6).

Top-level constants are importable without NumPy/torch. Server-side hooks (below the
declarations) import torch lazily.
"""
from __future__ import annotations

from ..core.capabilities import caps

NAME = "gr00t-n1.5"
BENCHMARK = "robocasa365"
POLICY_ID = "GR00T-N1.5-Human300-checkpoint120000"
CHUNK_LEN = 16
ACTION_DIM = 12
CONTROL_HZ = 20.0
DENOISING_STEPS = 4
CAPABILITIES = caps("action_chunk", "seeded_sampling", "isolated_streams", "denoise_step",
                    "velocity_readout", "proprio", "variable_execute")
PUBLIC_KEYS = (
    "annotation.human.task_description", "state.base_position", "state.base_rotation",
    "state.end_effector_position_relative", "state.end_effector_rotation_relative",
    "state.gripper_qpos", "video.robot0_agentview_left", "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
ACTION_SHAPES = {"action.end_effector_position": 3, "action.end_effector_rotation": 3,
                 "action.gripper_close": 1, "action.base_motion": 4, "action.control_mode": 1}


def _layout():
    offset, layout = 0, {}
    for key in sorted(ACTION_SHAPES):
        layout[key] = (offset, offset + ACTION_SHAPES[key])
        offset += ACTION_SHAPES[key]
    return layout


ACTION_LAYOUT = _layout()
PROPRIO_EEF = ("state.end_effector_position_relative", None)  # (public key, slice or None)
CHILD_PYTHON = "envs/gr00t-n15/bin/python"  # relative to the RobotPolicyBench root
BACKEND_ADAPTER = "backends/gr00t_robocasa365/adapter.py"
CANONICAL_TABLE = "robocasa365_atomic_seen_dev_r2.json"  # data/canonical_tables; seeded-reset divergence table
# Tier-2 case: seed 108, the non-subset calibration case of ChunkTrust's bitwise K0 proofs.
TIER2_CASE = {"group": "atomic_seen", "task": "CloseToasterOvenDoor", "trial": 1}


# ---- server side (runs inside the GR00T policy server; imports torch lazily) ----

class NoOpSampler:
    identity = {"module": "noop"}

    def on_step(self, step, x_t, v_t, ctx):
        return None

    def extras(self):
        return {}


def _seeded(fn, seed):
    """Protocol N1: dedicated torch stream for one call; global RNG restored afterwards (= crn.seeded_call)."""
    import torch
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return fn()


def seeded_sample(policy, observation, seed, *, samplers, effective=True, ctx=None):
    """One native get_action under a Protocol-N1 seed, with Sampler hooks between velocity and update.

    The action_encoder pre-hook captures x_t; the action_decoder forward hook gives each Sampler
    v_t = output[:, -horizon:]. A replacement is spliced back only when ``effective`` (candidate);
    the null computes it and discards it. With no samplers this is exactly native get_action.
    """
    if not samplers:
        return _seeded(lambda: policy.get_action(observation), seed)
    head = policy.model.action_head
    horizon = int(head.action_horizon)
    state = {"step": 0, "x": None}

    def pre(module, inputs):
        state["x"] = inputs[0]

    def post(module, inputs, output):
        v_t = output[:, -horizon:]
        replacement = None
        for sampler in samplers:
            proposed = sampler.on_step(state["step"], state["x"], v_t, ctx)
            if proposed is not None:
                replacement = proposed
        state["step"] += 1
        if replacement is None or not effective:
            return None
        if tuple(replacement.shape) != tuple(v_t.shape) or replacement.dtype != v_t.dtype:
            raise ValueError("Sampler replacement must match v_t shape and dtype")
        out = output.clone()
        out[:, -horizon:] = replacement
        return out

    handles = [head.action_encoder.register_forward_pre_hook(pre),
               head.action_decoder.register_forward_hook(post)]
    try:
        result = _seeded(lambda: policy.get_action(observation), seed)
    finally:
        for handle in handles:
            handle.remove()
    if state["step"] != int(head.num_inference_timesteps):
        raise RuntimeError(f"expected {head.num_inference_timesteps} denoising steps, saw {state['step']}")
    return result
