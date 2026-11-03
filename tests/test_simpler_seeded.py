"""EXT simpler-seeded, benchmark half: SimplerSeededBenchmark against a fake SimplerEnv (CPU; no SAPIEN, no GPU).

Ported from the run-local prototype tests (runs/pcd-port-20261103/simpler_ext/tests/test_simpler_seeded.py). The fake
simpler_env modules are installed only while this module runs (setUpModule/tearDownModule). Tests that step the env
need transforms3d (the vla-eval SIMPLER image and the SIMPLER server envs have it) and are skipped elsewhere; the
authoritative run is inside the pinned image with vla-eval 0.8.0 mounted (docs/extensions/simpler-seeded.md section 6).
"""
import asyncio
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

HAS_T3D = importlib.util.find_spec("transforms3d") is not None
needs_t3d = unittest.skipUnless(HAS_T3D, "needs transforms3d (run in the SIMPLER image or a SIMPLER server env)")

# ---------------- fake simpler_env (installed before the benchmark module imports it lazily) ----------------
LOG = {"make": [], "reset": [], "actions": []}


class _Unwrapped:
    def __init__(self, env):
        self.env = env

    def get_language_instruction(self):
        return self.env.instruction


class FakeEnv:
    """TimeLimit-like fake: truncated at `horizon`; terminated (=success) on the steps in `success_steps` (1-based)."""

    horizon = 80
    success_steps = ()
    instruction_change_at = None
    robot_uid = "google_robot_static"

    def __init__(self, task):
        self.task, self.t, self.seed, self.instruction = task, 0, None, "pick coke can"
        self.unwrapped = _Unwrapped(self)
        self.closed = False
        LOG["make"].append(task)

    def _obs(self):
        rng = np.random.default_rng((self.seed or 0) * 1000 + self.t)
        return {"image": {"overhead_camera": {"rgb": rng.integers(0, 255, (512, 640, 3), dtype=np.uint8)}}}

    def reset(self, **kw):
        LOG["reset"].append(dict(kw))
        self.seed, self.t = kw.get("seed"), 0
        return self._obs(), {}

    def step(self, action):
        LOG["actions"].append(np.array(action, copy=True))
        self.t += 1
        if self.instruction_change_at is not None and self.t == self.instruction_change_at:
            self.instruction = "pick something else"
        terminated = self.t in self.success_steps
        truncated = self.t >= self.horizon
        return self._obs(), 0.0, terminated, truncated, {"success": terminated}

    def close(self):
        self.closed = True


def _fake_modules():
    def make(task, **kw):
        assert not kw, kw
        return FakeEnv(task)

    mods = {name: types.ModuleType(name) for name in ("simpler_env", "simpler_env.utils", "simpler_env.utils.env",
                                                      "simpler_env.utils.env.observation_utils")}
    mods["simpler_env"].make = make
    mods["simpler_env.utils.env.observation_utils"].get_image_from_maniskill2_obs_dict = (
        lambda env, obs, camera_name=None: obs["image"]["overhead_camera"]["rgb"])
    return mods


_SAVED = {}
_PATCHES = []


def setUpModule():
    for name, mod in _fake_modules().items():
        _SAVED[name] = sys.modules.get(name)
        sys.modules[name] = mod
    # the fake env "renders" nowhere: the per-reset GPU isolation check is exercised in TestRenderIsolation only
    from harnessstudio.benchmarks.simpler_seeded import SimplerSeededBenchmark as B
    _PATCHES.append(mock.patch.object(B, "_check_render_isolation", return_value={"stub": True}))
    for p in _PATCHES:
        p.start()


def tearDownModule():
    for p in _PATCHES:
        p.stop()
    for name, mod in _SAVED.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


from harnessstudio.benchmarks.simpler_seeded import (  # noqa: E402
    GoogleAuthorsPostprocess, SimplerSeededBenchmark, seeds_sha256)
from vla_eval.orchestrator import _merge_observation_params  # noqa: E402

_ORIG_CHECK = SimplerSeededBenchmark.__dict__["_check_render_isolation"]   # before setUpModule stubs it
from vla_eval.runners.sync_runner import SyncEpisodeRunner  # noqa: E402

DRAWER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 26, 27, 29, 31, 33, 34, 35, 36, 37, 38,
          39, 41]
RNG = np.random.default_rng(7)


def raw_seq(n):
    """Raw OpenVLA-like actions (float64): small arm deltas, gripper switching between ~1 (open) and ~0 (closed)."""
    arm = RNG.normal(0, 0.05, (n, 6))
    g = np.where((np.arange(n) // 20) % 2 == 0, 0.996, 0.004)[:, None]
    return np.concatenate([arm, g], axis=1)


class FakeConn:
    def __init__(self, actions):
        self.actions, self.i, self.started, self.ended = list(actions), 0, None, None

    async def start_episode(self, payload):
        self.started = payload

    async def act(self, obs):
        a = self.actions[self.i]
        self.i += 1
        return {"actions": a}

    async def end_episode(self, result):
        self.ended = result


def run_episode(bench, task, actions, max_steps):
    conn = FakeConn(actions)
    res = asyncio.run(SyncEpisodeRunner().run_episode(bench, task, conn, max_steps=max_steps))
    return res, conn


class TestCases(unittest.TestCase):
    def test_one_task_per_seed_and_unique_case_keys(self):
        b = SimplerSeededBenchmark("google_robot_close_drawer", DRAWER)
        tasks = b.get_tasks()
        self.assertEqual([t["task_id"] for t in tasks], [f"seed{s}" for s in DRAWER])
        # HarnessStudio v2 case key = (contract, name#task_id, episode_idx)  (server/harness_server.py:131-140)
        keys = {("simpler_seeded", f"{t['name']}#{t['task_id']}", 0) for t in tasks}
        self.assertEqual(len(keys), len(DRAWER))
        self.assertTrue(all(set(t) == {"name", "task_name", "task_id"} for t in tasks))

    def test_constructor_validation(self):
        for bad in ([], [1, 1], [-1]):
            with self.assertRaises(ValueError):
                SimplerSeededBenchmark("google_robot_pick_coke_can", bad)
        with self.assertRaises(ValueError):
            SimplerSeededBenchmark("widowx_stack_cube", [0])
        with self.assertRaises(TypeError):
            SimplerSeededBenchmark("google_robot_pick_coke_can", [0], seed=0)  # no **kwargs

    def test_server_seed_observation_param_is_dropped(self):
        merged = _merge_observation_params(SimplerSeededBenchmark,
                                           {"task_name": "google_robot_pick_coke_can", "seeds": [0, 1]}, {"seed": 0})
        self.assertNotIn("seed", merged)
        SimplerSeededBenchmark(**merged)

    def test_specs_and_metric_keys(self):
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0])
        self.assertEqual({k: v.format for k, v in b.get_action_spec().items()},
                         {"position": "delta_xyz", "rotation": "euler_xyz", "gripper": "raw"})
        self.assertEqual(b.get_metric_keys(), {"success": "mean", "final_success": "mean"})

    def test_metadata(self):
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", list(range(32)))
        m = b.get_metadata()
        self.assertEqual(m["protocol"], "pcd_authors_31b027e_seeded")
        self.assertEqual(m["seeds_sha256"], seeds_sha256(range(32)))
        self.assertEqual(m["success_mode"], "accumulate")
        self.assertEqual(m["official_horizon"], 80)


class TestReset(unittest.TestCase):
    def setUp(self):
        for v in LOG.values():
            v.clear()
        FakeEnv.success_steps, FakeEnv.instruction_change_at, FakeEnv.horizon = (), None, 80

    def test_fresh_env_and_exact_reset_kwargs(self):
        b = SimplerSeededBenchmark("google_robot_close_drawer", [16, 5])
        for t in b.get_tasks():
            asyncio.run(b.start_episode(t))
        self.assertEqual(LOG["make"], ["google_robot_close_drawer"] * 2)
        self.assertEqual(LOG["reset"], [{"seed": 16}, {"seed": 5}])   # no options, seed from task_id

    def test_reset_refusals(self):
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0, 1])
        with self.assertRaises(ValueError):
            b.reset({"name": "google_robot_pick_coke_can", "task_id": "seed2"})
        with self.assertRaises(ValueError):
            b.reset({"name": "google_robot_pick_coke_can", "task_id": "seed1", "episode_idx": 3})
        with self.assertRaises(ValueError):
            b.reset({"name": "google_robot_pick_coke_can"})

    def test_observation(self):
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [3])
        asyncio.run(b.start_episode(b.get_tasks()[0]))
        obs = asyncio.run(b.get_observation())
        self.assertEqual(obs["images"]["primary"].shape, (512, 640, 3))
        self.assertEqual(obs["task_description"], "pick coke can")


@needs_t3d
class TestEpisode(unittest.TestCase):
    def setUp(self):
        for v in LOG.values():
            v.clear()
        FakeEnv.success_steps, FakeEnv.instruction_change_at, FakeEnv.horizon = (), None, 80

    def test_no_early_stop_any_and_final_success(self):
        FakeEnv.success_steps = (10, 11, 12)
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0])
        res, conn = run_episode(b, b.get_tasks()[0], raw_seq(80), max_steps=80)
        self.assertEqual(res["steps"], 80)                     # ran the full horizon after success
        self.assertEqual(len(LOG["actions"]), 80)
        self.assertEqual(res["metrics"], {"success": True, "final_success": False, "first_success_step": 9,
                                          "steps": 80})
        self.assertEqual(conn.started["task"]["task_id"], "seed0")

    def test_success_read_from_info(self):
        FakeEnv.success_steps = (5,)
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0])
        res, _ = run_episode(b, b.get_tasks()[0], raw_seq(80), max_steps=80)
        self.assertEqual(res["metrics"]["first_success_step"], 4)

    def test_success_on_last_step_and_never(self):
        FakeEnv.success_steps = (80,)
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0])
        res, _ = run_episode(b, b.get_tasks()[0], raw_seq(80), max_steps=80)
        self.assertEqual((res["metrics"]["success"], res["metrics"]["final_success"]), (True, True))
        FakeEnv.success_steps = ()
        res, _ = run_episode(b, b.get_tasks()[0], raw_seq(80), max_steps=80)
        self.assertEqual(res["metrics"], {"success": False, "final_success": False, "first_success_step": -1,
                                          "steps": 80})

    def test_truncation_governs_even_with_larger_max_steps(self):
        FakeEnv.horizon = 113
        b = SimplerSeededBenchmark("google_robot_close_drawer", [41])
        res, _ = run_episode(b, b.get_tasks()[0], raw_seq(200), max_steps=200)
        self.assertEqual(res["steps"], 113)

    def test_env_receives_authors_postprocessed_actions_and_state_resets(self):
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0, 1])
        acts = raw_seq(80)
        for task in b.get_tasks():
            LOG["actions"].clear()
            run_episode(b, task, acts, max_steps=80)
            ref = GoogleAuthorsPostprocess()
            want = [ref(a) for a in acts]
            self.assertTrue(all(x.tobytes() == y.tobytes() for x, y in zip(LOG["actions"], want)))
            self.assertEqual(LOG["actions"][0][6], 0.0)           # first-step relative gripper is 0 every episode
            self.assertTrue(any(abs(x[6]) > 0.5 for x in LOG["actions"]))  # sticky close/open fired

    def test_instruction_change_resets_gripper_state(self):
        FakeEnv.instruction_change_at = 30
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0])
        acts = raw_seq(80)
        run_episode(b, b.get_tasks()[0], acts, max_steps=80)
        self.assertEqual(LOG["actions"][30][6], 0.0)               # step after the change starts with previous=None

    def test_rejects_non_raw_actions(self):
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0])
        asyncio.run(b.start_episode(b.get_tasks()[0]))
        with self.assertRaises(ValueError):
            b.step({"actions": np.zeros((2, 7))})


if __name__ == "__main__":
    unittest.main()


class TestRenderIsolation(unittest.TestCase):
    """[S-6] render: gpu must see exactly the leased GPU and a Vulkan ICD pinned to the NVIDIA driver."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.nv = os.path.join(self.tmp.name, "nvidia_icd.json")
        with open(self.nv, "w") as f:
            f.write('{"ICD": {"library_path": "libGLX_nvidia.so.0"}}')
        self.lvp = os.path.join(self.tmp.name, "lvp_icd.json")
        with open(self.lvp, "w") as f:
            f.write('{"ICD": {"library_path": "libvulkan_lvp.so"}}')
        self.real = mock.patch.object(SimplerSeededBenchmark, "_check_render_isolation",
                                      _ORIG_CHECK)
        self.real.start()

    def tearDown(self):
        self.real.stop()
        self.tmp.cleanup()

    def check(self, uuids, env, gpu_uuid="GPU-abc"):
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0], gpu_uuid=gpu_uuid)
        out = subprocess.CompletedProcess([], 0, stdout="\n".join(uuids) + "\n", stderr="")
        with mock.patch("harnessstudio.benchmarks.simpler_seeded.subprocess.run", return_value=out), \
                mock.patch.dict(os.environ, env, clear=False):
            for k in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES"):
                if k not in env:
                    os.environ.pop(k, None)
            return b._check_render_isolation()

    def test_pass(self):
        rec = self.check(["GPU-abc"], {"VK_ICD_FILENAMES": self.nv, "VK_DRIVER_FILES": self.nv,
                                       "NVIDIA_VISIBLE_DEVICES": "6"})
        self.assertEqual(rec["gpu_uuids"], ["GPU-abc"])

    def test_refusals(self):
        good = {"VK_ICD_FILENAMES": self.nv, "VK_DRIVER_FILES": self.nv}
        cases = [(["GPU-abc", "GPU-def"], good), (["GPU-def"], good), ([], good),
                 (["GPU-abc"], {}), (["GPU-abc"], {"VK_ICD_FILENAMES": self.lvp}),
                 (["GPU-abc"], {"VK_ICD_FILENAMES": self.nv + os.pathsep + self.lvp})]
        for uuids, env in cases:
            with self.assertRaises(RuntimeError, msg=(uuids, env)):
                self.check(uuids, env)
        with self.assertRaises(RuntimeError):
            self.check(["GPU-abc"], good, gpu_uuid=None)

    def test_reset_runs_the_check_in_gpu_mode(self):
        b = SimplerSeededBenchmark("google_robot_pick_coke_can", [0], gpu_uuid="GPU-abc")
        with mock.patch.object(SimplerSeededBenchmark, "_check_render_isolation", return_value={"ok": 1}) as m:
            asyncio.run(b.start_episode(b.get_tasks()[0]))
        self.assertEqual(m.call_count, 1)
        self.assertEqual(b.get_metadata()["render_isolation"], {"ok": 1})


SIMPLER_FILES = {
    "harnessstudio/benchmarks/simpler_seeded.py", "harnessstudio/benchmarks/__init__.py",
    "harnessstudio/contracts/simpler_seeded.yaml", "harnessstudio/policies/openvla_simpler_hs.py",
    "harnessstudio/profiles/openvla-simpler-logits.yaml", "tests/test_simpler_seeded.py",
    "tests/test_openvla_simpler_server.py", "tests/test_openvla_logits.py", "docs/extensions/simpler-seeded.md",
    "docs/extensions/README.md", "docs/writing-a-method.md", "docs/superpowers/specs/2026-10-31-harnessstudio-v2-design.md",
}
BASE = "da817ba"   # ext2/openvla-contrast head this branch sits on (cut from 1fc5b10, rebased)


def test_simpler_branch_touches_only_its_files():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ok = subprocess.run(["git", "-C", root, "merge-base", "--is-ancestor", BASE, "HEAD"], capture_output=True)
    if ok.returncode:
        raise unittest.SkipTest(f"{BASE} is not an ancestor of HEAD (merged or rebased): scope is pinned by review")
    changed = subprocess.run(["git", "-C", root, "diff", "--name-only", BASE], capture_output=True, text=True,
                             check=True).stdout.split()
    assert set(changed) <= SIMPLER_FILES, sorted(set(changed) - SIMPLER_FILES)
