"""EXT openvla-residual-hook (docs/extensions/openvla-residual-hook.md): CPU tests.

Three groups:
- static (any env): capability/profile/contract, official grids (VM + VA base) against the scripts, Close Drawer VM
  refusal, byte-identity of the parents at main f7dcd0f (review critic-ext-1 R6), branch-scoped diff scope;
- benchmark with a fake SimplerEnv (stepping needs transforms3d; skipped elsewhere): build kwargs, reset options, no
  seed, `states`, Place-in-Drawer subtask switch modelled like the real env (R13a), official horizons (R13c);
- server (needs torch + transformers + opencv; skipped elsewhere): STAR-off == parent bitwise, activation record (R1),
  READS view (R2), lock + exception safety (R3), runtime shape assertion (R4), refusals, decode-short record (R9).
  The server is built with __new__ around a tiny REAL LlamaForCausalLM (32 decoder layers, OpenVLA vocabulary sizes)
  driven by the real transformers `generate`, as tests/test_openvla_simpler_server.py does.

Run the server group in a SIMPLER OpenVLA server env, e.g.
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree> <env python> -B -m unittest \
        tests.test_ext_openvla_residual_hook -v
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "toys"))

from harnessstudio.core import capabilities  # noqa: E402
from harnessstudio.contracts import load_contract  # noqa: E402
from harnessstudio.profiles import load_profile  # noqa: E402

HAS_T3D = importlib.util.find_spec("transforms3d") is not None
HAS_TORCH = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "cv2"))
needs_t3d = unittest.skipUnless(HAS_T3D, "needs transforms3d (SIMPLER image or a SIMPLER server env)")
needs_torch = unittest.skipUnless(HAS_TORCH, "needs torch + transformers + opencv (a SIMPLER OpenVLA server env)")

# ------------------------------------------------------------------------------------------------- R6 pins
MAIN_BASE = "f7dcd0fa0ab20147fa9e96b29e8417d50bbb5c62"
PINNED = {   # sha256 of the file at main f7dcd0f; every one must stay byte-identical on this branch
    "harnessstudio/policies/openvla_hs.py": "e7c6a4c302fece60b338b9a6c14c7ac0cea4dea2057ff72b74c0abfa0aab174b",
    "harnessstudio/policies/openvla_logits_hs.py": "ddf04264cb476f2b9a33f3a2fb4d986d83f555b9275064794ea70a844e456f8d",
    "harnessstudio/policies/openvla_simpler_hs.py": "21885a2aaef2206156edace72c9faada25c1ca97d3fbba509a5f4516caf17643",
    "harnessstudio/benchmarks/simpler_seeded.py": "31d45dd7e5c09a185229942469427943eb8ebfdea912719266ce73480fc6ce52",
    "harnessstudio/contracts/simpler_seeded.yaml": "d644d7e3d09551966ba08cbd81ae05ca8ee2b3a549ff189b4892844c90828d65",
    "harnessstudio/profiles/openvla-simpler-logits.yaml": "8a8847c90714a76697a3744dc6324f69db79dc114f1035a94f67b6c05cb3a623",
    "harnessstudio/policy_ext/fake.py": "99eca111d081f700cd2a9608cc013abddce3ca6c67a112f204b2826c3955a837",
    "harnessstudio/policy_ext/protocol.py": "4955649344effa9b4365c62578940f4d5569c9e7f2cabb11a45e3108111ac411",
    "harnessstudio/policy_ext/mixin.py": "b72ba1d14c4ee713389da2d115d5b927db8b42f4d4a40239ac7ab4e4a3924bc0",
    "harnessstudio/context/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "harnessstudio/context/bridge.py": "b3bb30a92627a233ee092a3586418134d3ae70869d28c9b888effc5ffadd0a4a",
    "harnessstudio/context/episode.py": "11f403b8178c9ff71a4e7ff899401d008e7f594876873b5af24de6524fe03601",
    "harnessstudio/context/policy.py": "cab674de2c2d150ed8616bf6984c0fd16306092a415d6f66b721fb719386a9ab",
    "harnessstudio/context/trace.py": "f60b9345902414f2869734b98ac1f125b809ccf619714302187e0aa3fe9c8697",
}
EXT_FILES = {
    "harnessstudio/policy_ext/residual_hooks.py", "harnessstudio/policies/openvla_simpler_star_hs.py",
    "harnessstudio/benchmarks/simpler_vm.py", "harnessstudio/contracts/simpler_vm.yaml",
    "harnessstudio/profiles/openvla-simpler-star.yaml", "harnessstudio/profiles/fake-t3.yaml",
    "harnessstudio/core/capabilities.py", "harnessstudio/core/method.py", "tests/test_ext_openvla_residual_hook.py", "tests/toys/hs_ext_residual_toy.py",
    "docs/extensions/openvla-residual-hook.md", "docs/extensions/README.md", "docs/writing-a-method.md",
    "tests/test_openvla_logits.py",     # one amended assertion (test_new_profiles), flagged in the EXT doc section 6
}
EXT_BRANCH = "ext2/openvla-residual-hook"


def _sha(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(*args):
    return subprocess.run(["git", "--no-optional-locks", "-C", ROOT, *args], capture_output=True, text=True)


class TestStatic(unittest.TestCase):
    def test_r6_parents_byte_identical(self):
        bad = {p: _sha(p) for p, h in PINNED.items() if _sha(p) != h}
        self.assertEqual(bad, {})
        here = sorted(os.listdir(os.path.join(ROOT, "harnessstudio", "context")))
        self.assertEqual([f for f in here if f.endswith(".py")],
                         sorted(p.rsplit("/", 1)[1] for p in PINNED if p.startswith("harnessstudio/context/")))

    def test_capability_profiles_contract(self):
        self.assertIn("residual_hooks", capabilities.CAPABILITIES)
        self.assertIn("residual_hooks", capabilities.check(load_profile("openvla-simpler-star").capabilities))
        self.assertTrue({"residual_hooks", "logits_readout"} <= set(load_profile("fake-t3").capabilities))
        for name in ("fake-t0", "fake-t1", "fake-t2", "openvla-simpler-logits", "openvla-libero-hs"):
            self.assertNotIn("residual_hooks", load_profile(name).capabilities, name)
        c = load_contract("simpler_vm")
        self.assertEqual(c.default_horizon, 80)                          # R13b
        from harnessstudio.benchmarks import simpler_vm as V
        self.assertEqual(dict(c.horizons), {f"google_robot_{g}": h for g, h in V.HORIZONS.items()})
        self.assertEqual(set(c.public_keys), {"images", "task_description", "states"})
        self.assertEqual(c.filter_public({"images": 1, "task_description": "x", "states": [0] * 8, "task_name": "t",
                                          "base_pose": 0, "tcp_pose": 0}).keys(), {"images", "task_description", "states"})

    def test_sampler_registration_accepts_residual_hooks_only_when_declared(self):
        from harnessstudio.core.errors import RegistrationError
        from harnessstudio.core.method import MethodConfig, ModuleRef, validate
        base = dict(name="toy-residual", version="1", paper="none", fidelity="faithful",
                    slots={"proposer": ModuleRef("hs_ext_residual_toy:ToyHook", {}),
                           "sampler": ModuleRef("hs_ext_residual_toy:ToyHook", {})})
        req = {"act", "chunk", "seeded_sampling", "isolated_streams"}
        validate(MethodConfig(**base, required=frozenset(req | {"residual_hooks"})))
        with self.assertRaisesRegex(RegistrationError, "residual_hooks"):
            validate(MethodConfig(**base, required=frozenset(req)))

    def test_tier1_route_a_star_like_method_gets_fake_t3(self):
        from harnessstudio.verify import tier1
        cfg = types.SimpleNamespace(required=frozenset({"act", "chunk", "seeded_sampling", "isolated_streams",
                                                        "residual_hooks"}))
        self.assertEqual(tier1.default_profile(cfg), "fake-t3")
        for req, want in (({"act"}, "fake-t0"), ({"act", "chunk", "seeded_sampling", "isolated_streams"}, None)):
            got = tier1.default_profile(types.SimpleNamespace(required=frozenset(req)))
            if want:
                self.assertEqual(got, want)
            self.assertNotEqual(got, "fake-t3")

    def test_grid_sizes_distinct_and_refusal(self):
        from harnessstudio.benchmarks import simpler_vm as V
        sizes = {g: len(V.enumerate_grid(g)) for g in V.GRIDS}
        self.assertEqual(sizes, {"vm_pick_coke_can": 300, "vm_move_near_v0": 240, "vm_open_drawer": 108,
                                 "vm_place_in_drawer": 108, "va_open_drawer": 27, "va_close_drawer": 27,
                                 "va_place_in_drawer": 27, "va_move_near": 60, "va_pick_coke_can": 75})
        for g in V.GRIDS:
            keys = [json.dumps({k: v for k, v in c.items() if k != "index"}, sort_keys=True) for c in V.enumerate_grid(g)]
            self.assertEqual(len(set(keys)), len(keys), g)
        with self.assertRaisesRegex(ValueError, "refused"):
            V.enumerate_grid("vm_close_drawer")
        with self.assertRaisesRegex(ValueError, "refused"):
            V.SimplerVMBenchmark("vm_close_drawer")
        params = inspect.signature(V.SimplerVMBenchmark.__init__).parameters.values()
        self.assertFalse(any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params))

    def test_vm_values_match_scripts(self):
        from harnessstudio.benchmarks import simpler_vm as V
        coke = V.enumerate_grid("vm_pick_coke_can")
        self.assertEqual(coke[0]["build_kwargs"], {"lr_switch": True, "urdf_version": "None"})
        self.assertEqual((coke[0]["obj"]["init_xy"], coke[24]["obj"]["init_xy"]), ([-0.35, -0.02], [-0.12, 0.42]))
        self.assertEqual(coke[1]["obj"]["init_xy"], [-0.35, 0.09])        # x outer, y inner (evaluator loop)
        self.assertTrue(coke[25]["build_kwargs"]["upright"])
        self.assertEqual(coke[75]["build_kwargs"]["urdf_version"], "recolor_tabletop_visual_matching_1")
        self.assertEqual(coke[0]["overlay"], "google_coke_can_real_eval_1")
        mn = V.enumerate_grid("vm_move_near_v0")
        self.assertEqual((mn[59]["obj"], mn[60]["build_kwargs"]["urdf_version"]),
                         ({"episode_id": 59}, "recolor_tabletop_visual_matching_1"))
        self.assertEqual(mn[0]["env_name"], "MoveNearGoogleBakedTexInScene-v0")
        dr = V.enumerate_grid("vm_open_drawer")
        self.assertEqual(list(dr[0]["build_kwargs"])[0], "shader_dir")
        self.assertEqual(dr[0]["build_kwargs"], {"shader_dir": "rt", "station_name": "mk_station_recolor",
                                                 "light_mode": "simple", "disable_bad_material": True,
                                                 "urdf_version": "recolor_cabinet_visual_matching_1"})
        self.assertEqual((dr[9]["env_name"], dr[1]["robot_xy"], dr[2]["robot_yaw"]),
                         ("OpenMiddleDrawerCustomInScene-v0", [0.765, -0.182], -0.06))
        pl = V.enumerate_grid("vm_place_in_drawer")
        self.assertEqual((pl[0]["build_kwargs"]["model_ids"], pl[9]["overlay"], pl[8]["obj"]["init_xy"]),
                         ("baked_apple_v2", "open_drawer_b0", [-0.02, 0.08]))

    def test_va_base_values_match_scripts(self):
        from harnessstudio.benchmarks import simpler_vm as V
        od, cd = V.enumerate_grid("va_open_drawer"), V.enumerate_grid("va_close_drawer")
        self.assertEqual({c["scene_name"] for c in od + cd}, {"frl_apartment_stage_simple"})
        self.assertEqual({json.dumps(c["build_kwargs"]) for c in od + cd}, {json.dumps({"shader_dir": "rt"})})
        self.assertEqual({c["overlay"] for c in od + cd}, {None})
        self.assertEqual([c["robot_xy"] for c in od[:4]], [[0.65, -0.2], [0.65, 0.0], [0.65, 0.2], [0.75, -0.2]])
        self.assertEqual(od[9]["env_name"], "OpenMiddleDrawerCustomInScene-v0")
        self.assertEqual(cd[0]["env_name"], "CloseTopDrawerCustomInScene-v0")
        pl = V.enumerate_grid("va_place_in_drawer")
        self.assertEqual(pl[0]["build_kwargs"], {"shader_dir": "rt", "model_ids": "apple"})
        self.assertEqual((pl[0]["robot_xy"], pl[9]["robot_xy"], pl[8]["obj"]["init_xy"]),
                         ([0.65, -0.2], [0.65, 0.0], [-0.02, 0.08]))
        mn = V.enumerate_grid("va_move_near")
        self.assertEqual((mn[0]["env_name"], mn[0]["build_kwargs"], mn[0]["overlay"], mn[0]["robot_yaw"]),
                         ("MoveNearGoogleInScene-v0", {}, None, -0.09))
        coke = V.enumerate_grid("va_pick_coke_can")
        self.assertEqual((coke[0]["build_kwargs"], coke[25]["build_kwargs"], coke[0]["overlay"]),
                         ({"lr_switch": True}, {"upright": True}, None))

    def test_p8_robot_states_reads_robot_attributes_only(self):
        """P8 ruling: the states function reads only the robot (base pose, TCP link pose, gripper closedness)."""
        import ast
        import inspect as _inspect
        from harnessstudio.benchmarks import simpler_vm as V
        fn = ast.parse(_inspect.getsource(V.robot_states))
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        robot_side = {"unwrapped", "agent", "robot", "pose", "to_transformation_matrix", "get_links", "get_name",
                      "get_pose", "get_gripper_closedness"}
        numeric_side = {"asarray", "float64", "float32", "linalg", "inv", "concatenate", "astype", "isfinite", "all",
                        "shape", "ndarray", "__name__"}          # np.ndarray annotation; type(exc).__name__ in the error
        self.assertEqual(attrs, robot_side | numeric_side, sorted(attrs ^ (robot_side | numeric_side)))
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        self.assertNotIn("raw_obs", names)
        src = _inspect.getsource(V.robot_states).lower()
        for bad in ("obj", "drawer", "contact", "success", "cabinet", "scene", "episode", "reward", "evaluate", "extra"):
            self.assertNotIn(bad, src.replace("get_links", "").replace("link", ""), bad)
        mk = ast.parse(__import__("textwrap").dedent(_inspect.getsource(V.SimplerVMBenchmark.make_obs)))
        calls = {n.func.id for n in ast.walk(mk) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("robot_states", calls)

    def test_robot_quat(self):
        from harnessstudio.benchmarks import simpler_vm as V
        self.assertEqual(V.robot_quat(0.0), [0.0, 0.0, 0.0, 1.0])
        q = V.robot_quat(-0.09)                      # rotation about z by pi - 0.09
        self.assertAlmostEqual(2 * math.atan2(q[3], q[0]), math.pi - 0.09, places=12)
        if HAS_T3D:
            from transforms3d.euler import euler2quat
            from transforms3d.quaternions import qmult
            for yaw in (-0.09, -0.03, -0.025, 0.0, -0.06):
                np.testing.assert_allclose(V.robot_quat(yaw), qmult(euler2quat(0, 0, yaw), [0, 0, 0, 1]), atol=1e-15)

    def test_branch_touches_only_its_files(self):
        if _git("rev-parse", "--verify", "-q", "main").returncode:
            raise unittest.SkipTest("no local main to compare with")
        if _git("merge-base", "--is-ancestor", "HEAD", "main").returncode == 0:
            raise unittest.SkipTest("HEAD is contained in main: no unmerged extension commits")
        branch = _git("symbolic-ref", "-q", "--short", "HEAD").stdout.strip()
        if branch and branch != EXT_BRANCH:
            raise unittest.SkipTest(f"on branch {branch!r}, not {EXT_BRANCH!r}")
        if not branch and _git("merge-base", "--is-ancestor", "HEAD", EXT_BRANCH).returncode:
            raise unittest.SkipTest(f"detached HEAD not on {EXT_BRANCH!r}")
        base = _git("merge-base", "HEAD", "main").stdout.strip()
        changed = set(_git("diff", "--name-only", base).stdout.split())
        changed |= set(_git("ls-files", "--others", "--exclude-standard").stdout.split())
        self.assertLessEqual(changed, EXT_FILES, sorted(changed - EXT_FILES))


# --------------------------------------------------------------------------------------- fake SimplerEnv
def _quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat4(p, q):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = _quat_wxyz_to_mat(q), np.asarray(p, dtype=np.float64)
    return t


class _FakePose:
    def __init__(self, p, q):
        self.p, self.q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64) / np.linalg.norm(q)

    def to_transformation_matrix(self):
        return _mat4(self.p, self.q)


class _FakeLink:
    def __init__(self, name, pose):
        self._name, self._pose = name, pose

    def get_name(self):
        return self._name

    def get_pose(self):
        return self._pose


class _FakeRobot:
    def __init__(self, env):
        self._env = env

    @property
    def pose(self):
        return _FakePose([0.35, 0.20, 0.079], [0.0, 0.0, 0.0, 1.0])      # base rotated by pi about z (VM init)

    def get_links(self):
        t = self._env._elapsed_steps
        tcp = _FakePose([0.4 + 0.001 * t, -0.2, 0.96], [0.16 + 0.001 * t, -0.60, 0.78, -0.03])
        return [_FakeLink("link_base", self.pose), _FakeLink("link_gripper_tcp", tcp)]


class _FakeAgent:
    def __init__(self, env):
        self.robot = _FakeRobot(env)
        self._env = env

    def get_gripper_closedness(self):
        return 0.25 if self._env._elapsed_steps > 2 else 0.0


class FakeVMEnv:
    """Gymnasium-like: TimeLimit at max_episode_steps; PlaceInto* force-advances its subtask when _elapsed_steps >= 100
    BEFORE a step (place_in_closed_drawer_in_scene.py:244-247). Like the pinned image's drawer/place envs, the raw obs
    has NO agent.eef_pos and NO extra.tcp_pose (P8 / D-1); `obs_style` adds them for the equality tests."""

    obs_style = "drawer"     # "drawer" | "tcp_pose" (vla-eval fallback inputs) | "eef_pos" (PCD fork)

    def __init__(self, env_name, **kwargs):
        self.env_name, self.kwargs, self._elapsed_steps, self.subtask = env_name, kwargs, 0, 0
        self.unwrapped = self
        self.agent = _FakeAgent(self)
        self.resets = []

    def get_language_instruction(self):
        if self.env_name.startswith("PlaceInto"):
            return "open top drawer" if self.subtask == 0 else "place apple into top drawer"
        return "pick coke can"

    def _obs(self):
        t = self._elapsed_steps
        agent = {"qpos": np.zeros(11), "base_pose": np.concatenate([self.agent.robot.pose.p, self.agent.robot.pose.q])}
        extra = {}
        tcp = self.agent.robot.get_links()[1].get_pose()
        if self.obs_style == "tcp_pose":
            extra["tcp_pose"] = np.concatenate([tcp.p, tcp.q])
        if self.obs_style == "eef_pos":
            agent["eef_pos"] = fork_eef_pos(self)
        return {"image": np.full((4, 4, 3), t % 255, np.uint8), "agent": agent, "extra": extra}

    def reset(self, **kw):
        self.resets.append(kw)
        self._elapsed_steps, self.subtask = 0, 0
        return self._obs(), {}

    def step(self, action):
        if self.env_name.startswith("PlaceInto") and self._elapsed_steps >= 100:
            self.subtask = 1
        self._elapsed_steps += 1
        trunc = self._elapsed_steps >= self.kwargs["max_episode_steps"]
        return self._obs(), 0.0, False, trunc, {"success": False}

    def close(self):
        pass


def _mat_to_quat_wxyz(m):
    """Independent rotation -> unit quaternion (eigenvector of the Bar-Itzhack K matrix), canonical w >= 0."""
    m = np.asarray(m, dtype=np.float64)
    k = np.array([[m[0, 0] - m[1, 1] - m[2, 2], m[1, 0] + m[0, 1], m[2, 0] + m[0, 2], m[2, 1] - m[1, 2]],
                  [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], m[2, 1] + m[1, 2], m[0, 2] - m[2, 0]],
                  [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]],
                  [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]]]) / 3.0
    vals, vecs = np.linalg.eigh(k)
    x, y, z, w = vecs[:, np.argmax(vals)]
    q = np.array([w, x, y, z])
    return -q if q[0] < 0 else q


def fork_eef_pos(env):
    """The PCD fork's agent.eef_pos (agents/base_agent.py:174-183), re-derived in float64 for the tests."""
    ag = env.unwrapped.agent
    rel = np.linalg.inv(ag.robot.pose.to_transformation_matrix()) @ ag.robot.get_links()[1].get_pose().to_transformation_matrix()
    return np.concatenate([rel[:3, 3], _mat_to_quat_wxyz(rel[:3, :3]), [1.0 - ag.get_gripper_closedness()]])


def _signfree_maxabs(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    qa, qb = a[3:7], b[3:7]
    q = min(np.abs(qa - qb).max(), np.abs(qa + qb).max())
    return max(np.abs(a[:3] - b[:3]).max(), q, abs(a[7] - b[7]))


class TestBenchmarkFakeEnv(unittest.TestCase):
    def setUp(self):
        from harnessstudio.benchmarks import simpler_vm as V
        from vla_eval.benchmarks.base import NullEpisodeRecorder
        self.V, self.made = V, []
        self.tmp = tempfile.TemporaryDirectory()
        ov = os.path.join(self.tmp.name, "real_inpainting")
        os.makedirs(ov)
        for n in ["google_coke_can_real_eval_1", "google_move_near_real_eval_1"] + [o[0] for o in V.DRAWER_OVERLAYS]:
            open(os.path.join(ov, n + ".png"), "wb").close()
        utils = types.ModuleType("simpler_env.utils.env.observation_utils")
        utils.get_image_from_maniskill2_obs_dict = lambda env, obs, camera_name=None: obs["image"]
        self.mods = {n: types.ModuleType(n) for n in ("simpler_env", "simpler_env.utils", "simpler_env.utils.env")}
        self.mods["simpler_env.utils.env.observation_utils"] = utils
        self.patches = [mock.patch.dict(sys.modules, self.mods),
                        mock.patch.object(V, "_gym_make", side_effect=lambda n, **kw: self.made.append(FakeVMEnv(n, **kw))
                                          or self.made[-1]),
                        mock.patch.object(V, "_overlay_dir", return_value=__import__("pathlib").Path(ov)),
                        mock.patch.object(V.SimplerVMBenchmark, "_check_render_isolation", return_value={"stub": 1})]
        for p in self.patches:
            p.start()
        self.Null = NullEpisodeRecorder

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def bench(self, grid, cases):
        b = self.V.SimplerVMBenchmark(grid, cases=cases)
        b._recorder = self.Null()
        return b

    def test_reset_kwargs_options_states(self):
        b = self.bench("vm_pick_coke_can", [0, 7])
        self.assertEqual([t["task_id"] for t in b.get_tasks()], ["vm000", "vm007"])
        obs = b.reset({"task_id": "vm007", "episode_idx": 0})
        env = self.made[-1]
        kw = env.kwargs
        self.assertEqual((env.env_name, kw["max_episode_steps"], kw["control_freq"], kw["sim_freq"]),
                         ("GraspSingleOpenedCokeCanInScene-v0", 80, 3, 513))
        self.assertEqual((kw["rgb_overlay_cameras"], kw["camera_cfgs"], kw["obs_mode"]),
                         (["overhead_camera"], {"add_segmentation": True}, "rgbd"))
        self.assertTrue(kw["rgb_overlay_path"].endswith("google_coke_can_real_eval_1.png"))
        self.assertEqual(list(kw)[:2], ["lr_switch", "urdf_version"])
        opts = env.resets[0]
        self.assertNotIn("seed", opts)
        np.testing.assert_allclose(opts["options"]["obj_init_options"]["init_xy"],
                                   self.V.enumerate_grid("vm_pick_coke_can")[7]["obj"]["init_xy"])
        np.testing.assert_allclose(opts["options"]["robot_init_options"]["init_rot_quat"], [0, 0, 0, 1])
        if HAS_T3D:                     # robot_states (P8) uses transforms3d mat2quat; absent in the shared venv
            o = b.make_obs(obs, {"task_id": "vm007"})
            self.assertEqual((o["states"].shape, o["task_description"]), ((8,), "pick coke can"))
        with self.assertRaises(ValueError):
            b.reset({"task_id": "vm003", "episode_idx": 0})
        with self.assertRaises(ValueError):
            b.reset({"task_id": "va000", "episode_idx": 0})

    def test_va_has_no_overlay(self):
        b = self.bench("va_close_drawer", [4])
        b.reset({"task_id": "va004", "episode_idx": 0})
        kw = self.made[-1].kwargs
        self.assertIsNone(kw["rgb_overlay_path"])
        self.assertNotIn("rgb_overlay_cameras", kw)
        self.assertEqual((list(kw)[0], kw["scene_name"], kw["max_episode_steps"]), ("shader_dir", "frl_apartment_stage_simple", 113))
        self.assertEqual(self.made[-1].env_name, "CloseTopDrawerCustomInScene-v0")

    @needs_t3d
    def test_place_in_drawer_switch_and_horizon(self):
        for grid in ("vm_place_in_drawer", "va_place_in_drawer"):
            b = self.bench(grid, [0])
            b.reset({"task_id": f"{grid[:2]}000", "episode_idx": 0})
            seen, steps, resets = [], 0, []
            orig = b._post.reset
            b._post.reset = lambda: (resets.append(steps), orig())
            while True:
                r = b.step({"actions": np.array([0, 0, 0, 0, 0, 0, 1.0])})
                steps += 1
                seen.append(b.make_obs(r.obs, {})["task_description"])
                if b.check_done(r):
                    break
            self.assertEqual(steps, 200)
            self.assertEqual((seen[99], seen[100]), ("open top drawer", "place apple into top drawer"))   # R13a
            self.assertEqual(resets, [100])     # gripper state reset once, inside the 101st step (counted before +1)
            res = b.get_step_result(r)
            self.assertEqual((res["steps"], res["horizon_ok"]), (200, True))

    @needs_t3d
    def test_horizon_ok_false_when_cut_short(self):
        b = self.bench("vm_move_near_v0", [3])
        b.reset({"task_id": "vm003", "episode_idx": 0})
        r = None
        for _ in range(10):
            r = b.step({"actions": np.zeros(7)})
        res = b.get_step_result(r)
        self.assertEqual((res["steps"], res["horizon_ok"]), (10, False))

    # ---- P8 (review critic-ext-2; evaluator defect D-1) ----------------------------------------------------------
    @needs_t3d
    def test_p8_states_without_eef_pos_or_tcp_pose(self):
        for grid, tid in (("vm_open_drawer", "vm000"), ("vm_place_in_drawer", "vm010"), ("va_close_drawer", "va004"),
                          ("vm_pick_coke_can", "vm007"), ("va_move_near", "va001")):
            b = self.bench(grid, [int(tid[2:])])
            obs = b.reset({"task_id": tid, "episode_idx": 0})
            env = self.made[-1]
            self.assertNotIn("eef_pos", obs["agent"])
            self.assertNotIn("tcp_pose", obs["extra"])
            for _ in range(4):
                o = b.make_obs(obs, {"task_id": tid})
                st = o["states"]
                self.assertEqual((st.shape, st.dtype), ((8,), np.float32))
                self.assertLess(_signfree_maxabs(st, fork_eef_pos(env)), 1e-6)
                self.assertGreaterEqual(float(st[3]), 0.0)
                self.assertAlmostEqual(float(np.linalg.norm(st[3:7].astype(np.float64))), 1.0, places=6)
                obs = b.step({"actions": np.zeros(7)}).obs

    @needs_t3d
    def test_p8_equals_vla_eval_fallback_and_fork_value(self):
        from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark
        for style in ("tcp_pose", "eef_pos"):
            with mock.patch.object(FakeVMEnv, "obs_style", style):
                b = self.bench("vm_pick_coke_can", [3])
                obs = b.reset({"task_id": "vm003", "episode_idx": 0})
                for _ in range(3):
                    base = SimplerEnvBenchmark.make_obs(b, obs, {"task_id": "vm003"})["states"]   # vla-eval's value
                    ours = b.make_obs(obs, {"task_id": "vm003"})["states"]
                    self.assertLess(_signfree_maxabs(ours, base), 1e-6, style)
                    obs = b.step({"actions": np.zeros(7)}).obs

    @needs_t3d
    def test_p8_raises_when_robot_state_unavailable(self):
        b = self.bench("vm_open_drawer", [0])
        obs = b.reset({"task_id": "vm000", "episode_idx": 0})
        env = self.made[-1]
        with mock.patch.object(_FakeAgent, "get_gripper_closedness", side_effect=AttributeError("no gripper")):
            with self.assertRaises(self.V.RobotStateUnavailable):
                b.make_obs(obs, {"task_id": "vm000"})
        with mock.patch.object(_FakeRobot, "get_links", return_value=[]):
            with self.assertRaises(self.V.RobotStateUnavailable):
                b.make_obs(obs, {"task_id": "vm000"})
        self.assertIsNotNone(env)


# --------------------------------------------------------------------------------------------- server
if HAS_TORCH:
    import torch
    from transformers import BatchFeature, LlamaConfig, LlamaForCausalLM

V_LOGITS, VOCAB, NB, AD, EMPTY, N_IMG, N_LAYERS = 32064, 32000, 256, 7, 29871, 4, 32
FRACTAL = {"mask": [True] * 6 + [False], "q01": [-0.22, -0.15, -0.23, -0.35, -0.42, -0.44, 0.0],
           "q99": [0.18, 0.15, 0.22, 0.59, 0.35, 0.45, 1.0]}


class _StubProcessor:
    def __call__(self, prompt, pil_image):
        digest = hashlib.sha256(prompt.encode()).digest()
        ids = [1] + [100 + b for b in digest[:9]]
        return BatchFeature({"input_ids": torch.tensor([ids], dtype=torch.long),
                             "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                             "pixel_values": torch.tensor(np.asarray(pil_image, dtype=np.float32))[None]})


class _StubVLA:
    """Prismatic's generate shape (image ids spliced after BOS) around a tiny real eager Llama with 32 layers."""

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        cfg = LlamaConfig(vocab_size=V_LOGITS, hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS,
                          num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                          bos_token_id=1, eos_token_id=2, pad_token_id=32000)
        cfg._attn_implementation = "eager"
        self.lm = LlamaForCausalLM(cfg).eval()
        with torch.no_grad():
            self.lm.lm_head.weight[VOCAB - NB: VOCAB] *= 6.0
            for p in self.lm.model.layers.parameters():
                p.mul_(3.0)                                         # non-trivial late-layer updates
        self.lm.generation_config.eos_token_id = None
        self.lm.generation_config.pad_token_id = 32000
        self.language_model = self.lm
        self.vocab_size = VOCAB
        self.config = types.SimpleNamespace(n_action_bins=NB, _attn_implementation="eager")
        bins = np.linspace(-1.0, 1.0, NB)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self.short = False

    def get_action_dim(self, key):
        return AD

    def get_action_stats(self, key):
        return FRACTAL

    def generate(self, input_ids=None, pixel_values=None, **kw):
        d = hashlib.sha256(pixel_values.to(torch.float32).numpy().tobytes()).digest()
        full = torch.cat([input_ids[:, :1], torch.tensor([[200 + b for b in d[:N_IMG]]]), input_ids[:, 1:]], dim=1)
        if self.short:
            kw = dict(kw, max_new_tokens=3)
        out = self.lm.generate(input_ids=full, **kw)
        return torch.cat([input_ids, out[:, full.shape[1]:]], dim=1)


def make_server(log=None, seed=0):
    from vla_eval.model_servers.predict import PredictModelServer
    from harnessstudio.policies import openvla_simpler_star_hs as R
    srv = R.HarnessOpenVLASimplerStarServer.__new__(R.HarnessOpenVLASimplerStarServer)
    PredictModelServer.__init__(srv, chunk_size=1, action_ensemble="newest")
    srv.model_path, srv.unnorm_key = "/stub", "fractal20220817_data"
    srv.jpeg_roundtrip = srv.center_crop = False
    srv._device = torch.device("cpu")
    srv._processor, srv._model = _StubProcessor(), _StubVLA(seed)
    srv._infer_lock = threading.RLock()
    srv._sampler_prefixes, srv._decoders, srv._states, srv._hooks = ("hs_",), {}, {}, {}
    srv._init_log(log)
    return srv


def obs(seed=0, text="open top drawer"):
    rng = np.random.default_rng(seed)
    return {"images": {"primary": rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)}, "task_description": text,
            "states": np.array([0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0], np.float32) + seed * 1e-3,
            "task_name": "google_robot_vm_open_drawer", "base_pose": np.zeros(7, np.float32)}


TOY = {"factory": "hs_ext_residual_toy:ToyHook", "params": {"start_from_last": 11, "gain": 0.5}}
CTX = types.SimpleNamespace(session_id="sess", episode_id="ep1")


def _records(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


@needs_torch
class TestServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.tmp.name, "actions.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _no_hooks(self, srv):
        for layer in srv._layers():
            self.assertFalse(layer._forward_hooks or layer._forward_pre_hooks)

    def test_init_requires_action_log(self):
        from harnessstudio.policies import openvla_simpler_star_hs as R
        with self.assertRaisesRegex(ValueError, "action_log is required"):
            R.HarnessOpenVLASimplerStarServer("/does/not/matter")

    def test_star_off_is_parent_bitwise(self):
        from harnessstudio.policies import openvla_simpler_hs as S
        srv = make_server(self.log)
        parent = S.HarnessOpenVLASimplerServer.__new__(S.HarnessOpenVLASimplerServer)
        parent.__dict__.update({k: v for k, v in srv.__dict__.items()})
        for o in (obs(1), obs(2)):
            (a, _), = srv.sample_chunks(o, [5], None, (), CTX)
            (b, _), = S.HarnessOpenVLASimplerServer.sample_chunks(parent, o, [5], None, (), CTX)
            self.assertEqual((a.dtype, a.tobytes()), (b.dtype, b.tobytes()))
        self._no_hooks(srv)

    def test_hooked_request_record_and_shape(self):
        srv = make_server(self.log)
        (native, _), = srv.sample_chunks(obs(3), [9], None, (), CTX)
        (star, ex), = srv.sample_chunks(obs(3), [9], TOY, (), CTX)
        self.assertEqual(ex, {"seed": 9})
        self.assertNotEqual(native.tobytes(), star.tobytes())          # the toy hook changes the decode
        self._no_hooks(srv)
        rec = [r for r in _records(self.log) if r["path"] == "residual_hook"]
        self.assertEqual(len(rec), 1)
        r = rec[0]
        self.assertEqual((r["seeds"], r["session"], r["episode"]), ([9], "sess", "ep1"))
        from harnessstudio.context.trace import fingerprint
        self.assertEqual(r["input_sha256"], fingerprint(obs(3)))
        self.assertEqual(r["hook"], {"factory": TOY["factory"], "params": TOY["params"]})
        self.assertEqual(r["info"]["keys"], ["states"])
        sh = r["shape"]
        self.assertEqual((sh["forwards"], sh["seq_lens"][1:], sh["layer_calls"], sh["mixed_calls"]),
                         (AD, [1] * (AD - 1), [N_LAYERS] * AD, [11] * AD))
        self.assertGreater(sh["seq_lens"][0], 1)
        self.assertEqual(sh["correction_start_layer"], 22)
        self.assertEqual(np.frombuffer(bytes.fromhex(r["action_f64_hex"]), np.float64).tobytes(),
                         np.asarray(star, np.float64).reshape(-1).tobytes())

    def test_r2_view_has_only_declared_keys(self):
        srv = make_server(self.log)
        srv.sample_chunks(obs(4), [1], TOY, (), CTX)
        hook = next(iter(srv._hooks.values()))
        self.assertEqual([sorted(v) for v in hook.seen], [["states"]])
        np.testing.assert_array_equal(hook.seen[0]["states"], obs(4)["states"])
        bad = {"factory": "hs_ext_residual_toy:ToyHook", "params": {"reads": ["states", "task_name"]}}
        with self.assertRaisesRegex(ValueError, "may read only"):
            srv.sample_chunks(obs(4), [1], bad, (), CTX)

    def test_shadow_toy_is_native_bitwise(self):
        srv = make_server(self.log)
        shadow = {"factory": "hs_ext_residual_toy:ToyHook", "params": {"start_from_last": 11, "gain": 0.5, "shadow": True}}
        (a, _), = srv.sample_chunks(obs(5), [2], None, (), CTX)
        (b, _), = srv.sample_chunks(obs(5), [2], shadow, (), CTX)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_refusals(self):
        srv = make_server(self.log)
        with self.assertRaisesRegex(ValueError, "readouts"):
            srv.sample_chunks(obs(), [1], TOY, ("logits",), CTX)
        with self.assertRaisesRegex(ValueError, "exactly one seed"):
            srv.sample_chunks(obs(), [1, 2], TOY, (), CTX)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            srv.sample_chunks(obs(), [1], {"factory": "os.path:join"}, (), CTX)
        self._no_hooks(srv)

    def test_r4_shape_violation_raises(self):
        from harnessstudio.policies import openvla_simpler_star_hs as R
        srv = make_server(self.log)
        orig = srv._generate

        def twice(o):
            orig(o)
            return orig(o)
        srv._generate = twice
        with self.assertRaises(R.ResidualHookShapeError):
            srv.sample_chunks(obs(6), [1], TOY, (), CTX)
        self._no_hooks(srv)
        recs = _records(self.log) if os.path.exists(self.log) else []
        self.assertFalse([r for r in recs if r.get("path") == "residual_hook"])   # no record for a refused shape

    def test_r3_exception_leaves_no_hooks_and_native_thread_stays_native(self):
        srv = make_server(self.log)
        ref = [srv.sample_chunks(obs(10 + i), [3], None, (), CTX)[0][0].tobytes() for i in range(4)]
        got, errors = [], []

        def native_worker():
            try:
                for _ in range(3):
                    got.append([srv.sample_chunks(obs(10 + i), [3], None, (), CTX)[0][0].tobytes() for i in range(4)])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        t = threading.Thread(target=native_worker)
        t.start()
        for i in range(4):
            srv.sample_chunks(obs(20 + i), [1], TOY, (), types.SimpleNamespace(session_id="s2", episode_id="e"))
        t.join()
        self.assertEqual(errors, [])
        self.assertTrue(all(g == ref for g in got))
        boom = mock.patch.object(srv._model.lm, "generate", side_effect=RuntimeError("boom"))
        with boom, self.assertRaisesRegex(RuntimeError, "boom"):
            srv.sample_chunks(obs(7), [1], TOY, (), CTX)
        self._no_hooks(srv)

    def test_r9_decode_short_record_then_raise(self):
        from harnessstudio.policies.openvla_simpler_hs import SimplerDecodeShort
        srv = make_server(self.log)
        srv._model.short = True
        for sampler, path in ((None, "native_seeded"), (TOY, "residual_hook")):
            with self.assertRaises(SimplerDecodeShort):
                srv.sample_chunks(obs(8), [4], sampler, (), CTX)
            r = [x for x in _records(self.log) if x.get("event") == "decode_short"][-1]
            self.assertEqual((r["path"], r["seeds"], r["session"]), (path, [4], "sess"))
        self._no_hooks(srv)

    def test_episode_state_is_per_episode(self):
        srv = make_server(self.log)
        for _ in range(3):
            srv.sample_chunks(obs(9), [1], TOY, (), CTX)
        key = next(iter(srv._hooks))
        self.assertEqual(srv._states[("sess", "ep1", key)]["requests"], 3)
        asyncio.run(srv.on_episode_start({}, CTX))
        self.assertNotIn(("sess", "ep1", key), srv._states)

    def test_p9_activity_recorded_and_shape_checked(self):
        from harnessstudio.policies import openvla_simpler_star_hs as R
        srv = make_server(self.log)
        ok = {"factory": "hs_ext_residual_toy:ToyHook", "params": {"start_from_last": 11, "gain": 0.5, "activity": "ok"}}
        (a_on, _), = srv.sample_chunks(obs(11), [1], ok, (), CTX)
        rec = [r for r in _records(self.log) if r["path"] == "residual_hook"][-1]
        er = rec["activity"]["edit_ratio"]
        self.assertEqual((len(er), {len(r) for r in er}, rec["activity"]["edit_ratio_layers"]),
                         (AD, {11}, list(range(22, 33))))
        (a_off, _), = srv.sample_chunks(obs(11), [1], TOY, (), CTX)       # same hook arithmetic, no activity()
        self.assertEqual(a_on.tobytes(), a_off.tobytes())
        self.assertIsNone([r for r in _records(self.log) if r["path"] == "residual_hook"][-1]["activity"])
        bad = {"factory": "hs_ext_residual_toy:ToyHook", "params": {"start_from_last": 11, "gain": 0.5, "activity": "bad"}}
        with self.assertRaises(R.ResidualHookShapeError):
            srv.sample_chunks(obs(11), [1], bad, (), CTX)
        self._no_hooks(srv)

    def test_star_hook_through_server_if_available(self):
        if importlib.util.find_spec("hs_star") is None:
            raise unittest.SkipTest("hs_star not on sys.path (run-local method package)")
        srv = make_server(self.log)
        from hs_star.server_hook import DEFAULTS
        sampler = {"factory": "hs_star.server_hook:StarResidualHook", "params": dict(DEFAULTS)}
        for t in range(12):
            o = obs(30)
            o["states"] = np.array([0.4 + 0.01 * t, 0.002 * t * t, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0], np.float32)
            srv.sample_chunks(o, [t], sampler, (), CTX)
        rec = [r for r in _records(self.log) if r["path"] == "residual_hook"]
        self.assertEqual(len(rec), 12)
        betas = [r["info"]["beta_t"] for r in rec]
        self.assertTrue(all(abs(b - 0.40) < 1e-12 for b in betas[:7]))
        self.assertTrue(all(0.25 <= b <= 0.55 for b in betas))
        self.assertEqual({r["shape"]["mixed_per_forward"] for r in rec}, {11})
        self.assertIn("w_t_f64", rec[-1]["info"])
        er = [r["activity"]["edit_ratio"] for r in rec]                        # P9 on the real STAR mixer
        self.assertTrue(all(len(e) == AD and all(len(row) == 11 for row in e) for e in er))
        # layer 22: warm start m = beta*dh + (1-beta)*dh, equal to dh only up to rounding (critic-ext-2 P9 said
        # "exactly 0"; in floating point it is O(eps): documented in the EXT doc)
        self.assertTrue(all(row[0] is not None and row[0] < 1e-5 for e in er for row in e))
        self.assertEqual(rec[0]["activity"]["edit_ratio_layers"], list(range(22, 33)))
        # STAR bitwise equal with activity logging on vs off (test-only toggle)
        from hs_star import star_core
        o = obs(31)
        srv2 = make_server(self.log + ".2")
        (on, _), = srv2.sample_chunks(o, [0], sampler, (), types.SimpleNamespace(session_id="x", episode_id="y"))
        orig = star_core.StarLayerMixer.__init__

        def off_init(self, *a, **k):
            k["record_activity"] = False
            orig(self, *a, **k)
        srv3 = make_server(self.log + ".3")
        with mock.patch.object(star_core.StarLayerMixer, "__init__", off_init):
            (off, _), = srv3.sample_chunks(o, [0], sampler, (), types.SimpleNamespace(session_id="x", episode_id="y"))
        self.assertEqual(on.tobytes(), off.tobytes())
        self.assertIsNone(_records(self.log + ".3")[-1]["activity"])


if __name__ == "__main__":
    unittest.main()
