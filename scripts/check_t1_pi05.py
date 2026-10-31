"""T1 checks for a HarnessStudio seeded policy server on a synthetic LIBERO observation.

    python scripts/check_t1_pi05.py ws://localhost:8106
"""
import sys
import time

import anyio
import numpy as np

from harnessstudio.context.policy import PolicyHandle
from harnessstudio.profiles import load_profile


def obs():
    rng = np.random.default_rng(0)
    return {"images": {"agentview": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
                       "wrist": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)},
            "states": rng.normal(size=8).astype(np.float64), "controller_states": np.zeros(8),
            "task_description": "pick up the black bowl and place it on the plate"}


async def main(url):
    h = PolicyHandle(url, load_profile("pi05-libero-hs"))
    await h.open({"task": {"name": "t1-check", "episode_idx": 0}})
    o = obs()
    t = time.monotonic(); a1 = (await h.sample(o, [11]))[0].actions; t1 = time.monotonic() - t
    a2 = (await h.sample(o, [11]))[0].actions
    t = time.monotonic(); trio = await h.sample(o, [22, 11, 33]); t3 = time.monotonic() - t
    b = (await h.sample(o, [22]))[0].actions
    native = await h.act(o)
    print("chunk shape", a1.shape, "dtype", a1.dtype)
    print("a. same seed twice bitwise identical:", np.array_equal(a1, a2))
    print("b. isolated (seed 11 inside [22,11,33] == alone):", np.array_equal(trio[1].actions, a1),
          "| seed 22:", np.array_equal(trio[0].actions, b))
    print("d. different seeds differ:", not np.array_equal(trio[0].actions, trio[2].actions))
    print("e. native request returns an action of shape", native.shape)
    print(f"timing: K=1 {t1*1000:.0f} ms, K=3 {t3*1000:.0f} ms")
    np.save("/tmp/claude-1003/-home-sejain/64151508-5f55-4394-aa9f-29e7307406cd/scratchpad/pi05_seed11.npy", a1)
    await h.close({})

anyio.run(main, sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8106")
