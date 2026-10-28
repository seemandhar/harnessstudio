"""The RTC paper's frozen Kinetix flow policies (real-time-chunking-kinetix @9296f31d, gs://rtc-assets/bc/31).

Kinetix extension (framework-extension-proposal-v2, rtc-repro-20261028). Top-level constants import without
NumPy/JAX; the runtime lives in harnessstudio.backends.kinetix_rtc12 and runs in the authors' pinned uv env.

Protocol parameters inference delay d and execute horizon s (Capability.ASYNC_EXECUTION) are identical for
every arm and enter the protocol hash. Only the sampler slot is supported in this version (SUPPORTED_SLOTS).
"""
from __future__ import annotations

import os
from pathlib import Path

from ..core.capabilities import caps

NAME = "kinetix-flow"
BENCHMARK = "kinetix-rtc12"
POLICY_ID = "rtc-kinetix-bc-flow-mlpmixer-epoch32"
CHUNK_LEN = 8               # H; the per-protocol execute horizon s becomes the episode's native cadence
ACTION_DIM = 6              # LARGE_ENV_PARAMS: 4 motor + 2 thruster bindings, every level
OBS_DIM = 679               # Kinetix-Symbolic-Continuous-v1 with LARGE_ENV_PARAMS
CONTROL_HZ = 30.0           # dt 1/60 with frame_skip 2
NUM_FLOW_STEPS = 5
MAX_TIMESTEPS = 256
ACTION_NOISE_STD = 0.1
CAPABILITIES = caps("action_chunk", "seeded_sampling", "differentiable_denoiser", "async_execution")
SUPPORTED_SLOTS = frozenset({"sampler"})
PUBLIC_KEYS = ("kinetix/instruction", "kinetix/obs", "kinetix/prev_chunk")
INSTRUCTION = "kinetix"     # constant: modules never see the level or case
LEVELS = (
    "worlds/l/grasp_easy.json", "worlds/l/catapult.json", "worlds/l/cartpole_thrust.json",
    "worlds/l/hard_lunar_lander.json", "worlds/l/mjc_half_cheetah.json", "worlds/l/mjc_swimmer.json",
    "worlds/l/mjc_walker.json", "worlds/l/h17_unicycle.json", "worlds/l/chain_lander.json",
    "worlds/l/catcher_v3.json", "worlds/l/trampoline.json", "worlds/l/car_launch.json",
)
UPSTREAM_COMMIT = "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b"
KINETIX_COMMIT = "cf7453ea103fa0b77348af1a39f689c658161613"
UPSTREAM_SHA256 = {
    "src/eval_flow.py": "c91383ae9a2710ab1f94705d7d9194eb4fb0a00b4477ef5debe5d74d800ec258",
    "src/model.py": "f59e9479c4fe4386ca027ba1a5c0b4940552adfae8cdb38d177cefec1427482e",
    "src/train_expert.py": "47b72a15303f6030db425480a616c2de1c0e8fad1fdd6a70958cca73a6a19b70",
}
CHECKPOINT_EPOCH_DIR = "31"
BC_MANIFEST_SHA256 = "07400b16915b15e3371b44c0b8e58fefd1a1cf31f4a0f38177a31d1ec5639816"
NOISE_MODE = "reference_rng"
_DEFAULT_ROOT = "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtc-repro-20261028/paper-repro/rtc"


def kinetix_root() -> Path:
    """Folder holding upstream/ (pinned checkout), env/ (authors' uv env) and assets/bc/."""
    return Path(os.environ.get("HARNESSSTUDIO_KINETIX_ROOT") or _DEFAULT_ROOT).resolve()


# Absolute child interpreter (tier 1's plugin import probe joins it onto the bench root; an absolute
# path wins in pathlib, so the probe runs in the authors' env).
CHILD_PYTHON = str(kinetix_root() / "env" / "bin" / "python")
BACKEND_ADAPTER = "backends/kinetix_rtc12/launcher.py"
CANONICAL_TABLE = None
TIER2_CASE = {"level": "worlds/l/grasp_easy.json", "inference_delay": 1, "execute_horizon": 1, "rollouts": 64}
