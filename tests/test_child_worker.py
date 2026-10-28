"""Worker + server end to end on CPU with a fake GR00T policy and a fake MultiStep env (GR00T env only)."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None

KEYS = ("annotation.human.task_description", "state.base_position", "state.base_rotation",
        "state.end_effector_position_relative", "state.end_effector_rotation_relative",
        "state.gripper_qpos", "video.robot0_agentview_left", "video.robot0_agentview_right",
        "video.robot0_eye_in_hand")
SHAPES = {"action.end_effector_position": 3, "action.end_effector_rotation": 3,
          "action.gripper_close": 1, "action.base_motion": 4, "action.control_mode": 1}


class FakeEnv:
    def __init__(self, horizon):
        self.n_action_steps, self.horizon, self.t = 16, horizon, 0
        self.pos = np.zeros(3, np.float32)

    def obs(self):
        frame = np.full((1, 8, 8, 3), self.t % 200, np.uint8)
        return {"annotation.human.task_description": np.array(["pick the cup"]),
                "state.base_position": np.zeros((1, 3), np.float32),
                "state.base_rotation": np.zeros((1, 4), np.float32),
                "state.end_effector_position_relative": self.pos[None].copy(),
                "state.end_effector_rotation_relative": np.zeros((1, 3), np.float32),
                "state.gripper_qpos": np.zeros((1, 2), np.float32),
                "video.robot0_agentview_left": frame, "video.robot0_agentview_right": frame.copy(),
                "video.robot0_eye_in_hand": frame.copy()}

    def step(self, actions):
        n = min(self.n_action_steps, self.horizon - self.t)
        for i in range(n):
            self.pos = self.pos + 0.05 * actions["action.end_effector_position"][i]
        self.t += n
        return self.obs(), 0.0, self.t >= self.horizon, False, {"rewards": [0.0] * n, "success": [False]}


class FakePolicy:
    def get_action(self, obs):
        return {k: torch.randn(16, w).numpy().astype(np.float32) for k, w in SHAPES.items()}


class LocalRPC:
    """Routes call_endpoint to real host.server.Endpoints (serialization skipped)."""
    def __init__(self, endpoints):
        self.ep = endpoints

    def call_endpoint(self, name, payload):
        return getattr(self.ep, name)(payload)


class CountingVideo:
    frames = 0

    def __init__(self, path):
        pass

    def append(self, public):
        CountingVideo.frames += 1

    def close(self):
        pass


@unittest.skipIf(torch is None, "GR00T env only")
class WorkerTest(unittest.TestCase):
    def run_arm(self, tmp, arm, method_name, baseline_run=None):
        from harnessstudio.bench import groot_native as gw
        from harnessstudio.bench.evaluation import validate_episode
        from harnessstudio.backends.gr00t_robocasa365 import worker
        from harnessstudio.core.manifest import method_manifest
        from harnessstudio.core.registry import get
        from harnessstudio.host.server import Endpoints
        config, origin = get(method_name)
        mm = method_manifest(config, origin, policy="gr00t-n1.5")
        protocol = json.loads((Path(worker.__file__).parent / "copied" / "protocol_v2_n1.json").read_text())
        out = Path(tmp) / arm
        manifest = {"harness_config": {"arm": arm, "method": method_name}, "harness_config_sha256": "e" * 64,
                    "method_manifest": mm, "protocol": protocol, "checkpoint_identity": "c" * 64,
                    "protocol_sha256": "d" * 64, "baseline_run": str(baseline_run) if baseline_run else None,
                    "harness_identity": ("gr00t-n15-n1-baseline" if arm == "baseline"
                                         else f"{method_name}@{config.version}/{arm}"),
                    "output": str(out)}
        case = {"id": "FakeTask--0002", "task": "FakeTask", "trial": 2, "horizon": 48, "seed": 109}
        env = FakeEnv(48)
        CountingVideo.frames = 0
        with (Path(tmp) / f"{arm}-calls.jsonl").open("x") as log:
            ep = Endpoints(FakePolicy(), method_slots=mm["method"]["slots"], manifest_sha256=mm["sha256"],
                           log=log, hashes=gw.hashes)
            record = worker.run_episode(env, env.obs(), LocalRPC(ep), manifest, case,
                                        out / "results" / case["id"], video_factory=CountingVideo)
        rows = [json.loads(l) for l in (out / "results" / case["id"] / "trace.jsonl").read_text().splitlines()]
        dispatch = [r for r in rows if r["event"] == "action_dispatch"]
        self.assertEqual(record["status"], "complete", record.get("error"))
        self.assertEqual(record["validity"], "valid")
        self.assertEqual(CountingVideo.frames, record["chunks"])
        self.assertEqual(len(dispatch), record["chunks"])
        validate_episode(record)
        return out, record, dispatch

    def test_three_arms_medoid(self):
        fields = ("input", "action", "noise_seed", "execute_steps", "selected_k")
        with tempfile.TemporaryDirectory() as tmp:
            base_dir, base, brows = self.run_arm(tmp, "baseline", "medoid-bon")
            _, null, nrows = self.run_arm(tmp, "null", "medoid-bon", baseline_run=base_dir)
            _, cand, crows = self.run_arm(tmp, "candidate", "medoid-bon")
        self.assertEqual([[r[f] for f in fields] for r in brows], [[r[f] for f in fields] for r in nrows])
        self.assertIsNone(null["divergence"])
        self.assertEqual(base["policy_calls"], 3)
        self.assertEqual(null["policy_calls"], 24)
        self.assertEqual(cand["policy_calls"], 24)
        self.assertEqual(base["harness"], "gr00t-n15-n1-baseline")
        self.assertEqual(cand["harness"], "medoid-bon@2/candidate")

    def test_public_copy_rejects_bad_schema(self):
        from harnessstudio.bench import groot_native as gw
        obs = FakeEnv(16).obs()
        with self.assertRaises(ValueError):
            gw.public_copy(dict(obs, reward=1.0))
        bad = dict(obs)
        bad.pop("state.gripper_qpos")
        with self.assertRaises(ValueError):
            gw.public_copy(bad)
