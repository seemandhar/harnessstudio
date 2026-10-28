"""Pi0.5 (openpi, pi05_pretrain_human300 / multitask_learning/75000) on RoboCasa365.

Native protocol (openpi-robocasa @5a6beda9): three public cameras resized to 224 + a 16-value state +
the original instruction; 12-D actions, replan every 5 steps; horizon ceil(1.5 * official horizon).
Top-level constants import without NumPy/JAX.
"""
from __future__ import annotations

from ..core.capabilities import caps

NAME = "pi0.5"
BENCHMARK = "robocasa365"
POLICY_ID = "pi0.5-pretrain-human300-75000"
CHUNK_LEN = 5
ACTION_DIM = 12
CONTROL_HZ = 20.0
OPENPI_COMMIT = "5a6beda9ff99da30b4e1b59320f6a32971d7c397"
MODEL_CONFIG = "pi05_pretrain_human300"
# Seeded sampling: openpi Policy.infer splits Policy._rng each call; setting it to
# jax.random.key(seed % 2**32) before a sample makes that sample seeded and isolated.
# No DENOISE_STEP / VELOCITY_READOUT: the flow loop is inside a jitted JAX function.
CAPABILITIES = caps("action_chunk", "seeded_sampling", "isolated_streams", "proprio", "variable_execute")
CAMERAS = ("video.robot0_agentview_left", "video.robot0_eye_in_hand", "video.robot0_agentview_right")
STATE = ("state.end_effector_position_relative", "state.end_effector_rotation_relative",
         "state.base_position", "state.base_rotation", "state.gripper_qpos")
PUBLIC_KEYS = ("observation/image", "observation/right_image", "observation/state",
               "observation/wrist_image", "prompt")
ACTION_LAYOUT = {"action.end_effector_position": (0, 3), "action.end_effector_rotation": (3, 6),
                 "action.gripper_close": (6, 7), "action.base_motion": (7, 11), "action.control_mode": (11, 12)}
PROPRIO_EEF = ("observation/state", (0, 3))  # state16 = eef_pos(3) | eef_rot | base_pos | base_rot | gripper
CHILD_PYTHON = "envs/gr00t-n15/bin/python"   # simulator worker interpreter, relative to RobotPolicyBench
BACKEND_ADAPTER = "backends/pi05_robocasa365/adapter.py"
CANONICAL_TABLE = None  # no reset-divergence table yet; runs pass --canonical-table none
TIER2_CASE = {"group": "composite_unseen", "task": "CuttingToolSelection", "trial": 0}  # shortest unseen horizon
