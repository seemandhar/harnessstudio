"""HarnessStudio Pi0.5 (openpi) / RoboCasa365 worker: server mode (openpi policy + seeded propose) and
episode mode (robocasa gym env + HarnessStudio pipeline). Native protocol follows openpi-robocasa
@5a6beda9 examples/robocasa (three public cameras resized with pad to 224, state16, original prompt,
12-D actions, replan every 5 steps, horizon ceil(1.5 * official)). Resets are seeded for every trial
(7 + 50 * task_index + trial) so paired arms start identically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
GR00T_BACKEND = HERE.parent / "gr00t_robocasa365"
for _p in (str(GR00T_BACKEND / "copied"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harnessstudio.bench import groot_native as gw  # noqa: E402  (hashing / JSON helpers only)
from harnessstudio.bench.common import read_json, verify_manifest  # noqa: E402
from harnessstudio.core.errors import MethodContractError  # noqa: E402

RUNNER_VERSION = "harnessstudio-pi05-2"
DETERMINISTIC_XLA_FLAG = "--xla_gpu_autotune_level=0"


def require_deterministic_xla():
    """Refuse to serve unless XLA autotuning is off (cross-process bitwise determinism)."""
    if DETERMINISTIC_XLA_FLAG not in os.environ.get("XLA_FLAGS", "").split():
        raise RuntimeError("Pi0.5 server needs XLA_FLAGS=" + DETERMINISTIC_XLA_FLAG
                           + " (XLA autotune makes actions differ across processes)")
CAMERAS = ("video.robot0_agentview_left", "video.robot0_eye_in_hand", "video.robot0_agentview_right")
CAMERA_KEYS = ("observation/image", "observation/wrist_image", "observation/right_image")
STATE = ("state.end_effector_position_relative", "state.end_effector_rotation_relative",
         "state.base_position", "state.base_rotation", "state.gripper_qpos")


def public_request(obs, image_tools, size=224):
    """Native openpi-robocasa public request (identical to the RoboMetaHarness Pi0.5 bridge)."""
    import numpy as np
    request = {}
    for key, name in zip(CAMERAS, CAMERA_KEYS):
        value = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[key]), size, size))
        if value.shape != (size, size, 3) or value.dtype != np.uint8:
            raise ValueError("Pi0.5 native public image preprocessing changed")
        request[name] = value
    state = np.concatenate([np.asarray(obs[key]).reshape(-1) for key in STATE], axis=0)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("Pi0.5 native public state must contain 16 finite values")
    request["observation/state"] = state
    request["prompt"] = str(obs["annotation.human.task_description"])
    if not request["prompt"].strip():
        raise ValueError("Pi0.5 native instruction is empty")
    return request


def validate_actions(actions):
    import numpy as np
    value = np.asarray(actions)
    if value.ndim != 2 or value.shape[0] < 5 or value.shape[1] != 12 or not np.isfinite(value).all():
        raise ValueError("Pi0.5 native action must contain at least five finite 12D rows")
    return value


class Pi05EnvPort:
    def __init__(self, env, image_tools, convert_action):
        self.env, self.image_tools, self.convert_action = env, image_tools, convert_action

    def execute(self, native, start, count):
        from harnessstudio.core.pipeline import StepResult
        actions = validate_actions(native)
        executed, success, obs = 0, False, None
        for row in actions[start:start + count]:
            obs, _, _, _, info = self.env.step(self.convert_action(row))
            executed += 1
            if info["success"]:
                success = True
                break
        return StepResult(public_request(obs, self.image_tools), success, executed, success)


class Pi05Client:
    """PolicyClient over openpi's websocket protocol; HarnessStudio endpoints ride in the request dict."""
    def __init__(self, client, manifest_sha256, before_call, after_call):
        self.client, self.manifest_sha256 = client, manifest_sha256
        self.before, self.after = before_call, after_call

    def _call(self, request):
        self.before()
        started, ok = time.monotonic(), False
        try:
            out = self.client.infer(request)
            ok = True
            return out
        finally:
            self.after(ok, time.monotonic() - started)

    def get_action_seeded(self, observation, seed):
        out = self._call(dict(observation, hs_endpoint="get_action_seeded", hs_seed=int(seed)))
        actions = validate_actions(out["actions"])
        return actions, actions[:5]

    def propose(self, observation, *, arm, decision_index, step_index, seeds, want):
        from harnessstudio.core.runtime import ProposeResult
        out = self._call(dict(observation, hs_endpoint="propose", hs_arm=arm, hs_decision_index=decision_index,
                              hs_step_index=step_index, hs_seeds=[int(s) for s in seeds], hs_want=list(want),
                              hs_manifest=self.manifest_sha256))
        natives = [validate_actions(a) for a in out["natives"]]
        return ProposeResult(natives, [a[:5] for a in natives], dict(out.get("extras") or {}), dict(out.get("cost") or {}))


class FrameVideo(gw.PublicVideo):
    """One pre-action public frame per decision from the resized agent-view image (RI-2 contract)."""
    def append(self, public):
        import numpy as np
        frame = np.asarray(public["observation/image"])
        super().append({"video.robot0_agentview_left": frame[None]})


def run_episode(env, initial, client, image_tools, convert_action, manifest, case, directory, video_factory=None):
    from harnessstudio.core.arms import Arm
    from harnessstudio.core.capabilities import unified_names
    from harnessstudio.core.pipeline import EpisodeSpec, run_episode as studio_episode
    from harnessstudio.eval.loading import method_from_manifest
    from harnessstudio.policies import pi05 as p
    from harnessstudio.runtime import PolicySpec
    from harnessstudio.runtime.contracts import public_hash
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    core = []
    config = manifest["harness_config"]
    record = {"status": "running", "quality_result": False, "runner_version": RUNNER_VERSION, "case_id": case["id"],
              "task": case["task"], "group": case["group"], "trial": case["trial"], "horizon": case["horizon"],
              "harness": manifest["harness_identity"], "arm": config["arm"], "method": config["method"],
              "policy": "pi0.5", "checkpoint_identity": manifest["checkpoint_identity"],
              "checkpoint_sha256": manifest["checkpoint_identity"], "protocol_sha256": manifest["protocol_sha256"],
              "method_manifest_sha256": manifest["method_manifest"]["sha256"], "seed": case["seed"],
              "seed_mode": "seeded_reset", "native_reset_seed": case["seed"], "observation_stride": 5,
              "action_chunk": 5, "control_hz": 20, "steps": 0, "chunks": 0, "success": False,
              "policy_dispatch_attempts": 0, "policy_calls_completed": 0, "policy_seconds": 0.0,
              "training_steps": 0, "manifest_sha256": manifest.get("manifest_digest")}

    def progress():
        gw._write(directory / "progress.json", record)
    trace = (directory / "trace.jsonl").open("x", buffering=1)
    video = (video_factory or FrameVideo)(directory / "episode.mp4")

    def before_call():
        record["policy_dispatch_attempts"] += 1
        progress()

    def after_call(completed, seconds):
        record["policy_calls_completed"] += int(completed)
        progress()
    try:
        method = method_from_manifest(manifest["method_manifest"])
        arm = Arm(config["arm"])
        public = public_request(initial, image_tools)
        record["initial_observation_sha256"] = public_hash(public)
        baseline = None
        if arm is Arm.NULL:
            baseline = json.loads((Path(manifest["baseline_run"]) / "results" / case["id"] / "core-traces.json").read_text())
        spec = EpisodeSpec(case_id=case["id"], horizon=case["horizon"], chunk_len=5, action_dim=12, control_hz=20.0,
                           benchmark="RoboCasa365", public_keys=p.PUBLIC_KEYS,
                           noise_namespace=manifest["protocol"]["noise"]["namespace"],
                           instruction_of=lambda obs: str(obs["prompt"]),
                           with_instruction=lambda obs, text: dict(obs, prompt=text),
                           input_hashes=gw.hashes, action_hashes=lambda a: gw.hashes({"actions": a}),
                           method_manifest_sha256=manifest["method_manifest"]["sha256"],
                           action_layout=p.ACTION_LAYOUT, proprio_eef=p.PROPRIO_EEF)
        policy_spec = PolicySpec(p.POLICY_ID, manifest["checkpoint_identity"], unified_names(p.CAPABILITIES))
        result = studio_episode(method, arm, spec, public, Pi05EnvPort(env, image_tools, convert_action),
                                Pi05Client(client, manifest["method_manifest"]["sha256"], before_call, after_call),
                                policy_spec, trace=lambda row: gw._append(trace, row), baseline_core_traces=baseline,
                                before_decision=video.append)
        core.extend(result.pop("core_traces"))
        gw._write(directory / "unified-manifest.json", dict(result.pop("unified_manifest"), runner_version=RUNNER_VERSION))
        record.update(result)
        record["chunks"] = result["decisions"]
        record.update(status="complete", quality_result=True, validity="valid")
    except MethodContractError:
        record.update(status="complete", quality_result=False, validity="invalid_method_defect", error=traceback.format_exc())
    except BaseException:
        record.update(status="infrastructure_error", quality_result=False, validity="infrastructure", error=traceback.format_exc())
    finally:
        try:
            video.close()
        except BaseException:
            record.update(status="infrastructure_error", quality_result=False, validity="infrastructure",
                          error=traceback.format_exc())
        record["wall_seconds"] = time.monotonic() - started
        trace.close()
        gw._write(directory / "core-traces.json", core)
        progress()
        gw._write(directory / "result.json", record)
    return record


def task_registry():
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    groups = ("atomic_seen", "composite_seen", "composite_unseen")
    return {g: [{"name": t, "horizon": math.ceil(1.5 * get_task_horizon(t))} for t in TASK_SET_REGISTRY[g]]
            for g in groups}


def serve(manifest, port, receipt):
    import asyncio
    import dataclasses
    import jax
    import websockets.asyncio.server as websockets_server
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server as native_server
    from openpi.shared import normalize
    from openpi.training import config as training_config
    require_deterministic_xla()
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError("Pi0.5 policy server must use its assigned CUDA GPU")
    started = time.monotonic()
    stats = normalize.load(Path(manifest["checkpoint_path"]) / "assets")
    cfg = training_config.get_config("pi05_pretrain_human300")
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, base_config=training_config.DataConfig(norm_stats=stats)))
    policy = policy_config.create_trained_policy(cfg, manifest["checkpoint_path"], norm_stats=stats)
    calls = (Path(receipt).parent / "server-calls.jsonl").open("x", buffering=1)
    expected_manifest = manifest["method_manifest"]["sha256"]

    def seeded(observation, seed):
        policy._rng = jax.random.key(int(seed) % (2 ** 32))
        return validate_actions(policy.infer(observation)["actions"])

    class HarnessStudioPolicy:
        count = 0

        def infer(self, request):
            self.count += 1
            before = time.monotonic()
            endpoint = request.pop("hs_endpoint", None)
            try:
                if endpoint == "get_action_seeded":
                    seed = int(request.pop("hs_seed"))
                    out = {"actions": seeded(request, seed)}
                    gw._append(calls, {"call": self.count, "endpoint": endpoint, "noise_seed": seed,
                                       "action": gw.hashes(out), "wall_seconds": time.monotonic() - before})
                    return out
                if endpoint == "propose":
                    arm, seeds = request.pop("hs_arm"), [int(s) for s in request.pop("hs_seeds")]
                    want, sha = request.pop("hs_want"), request.pop("hs_manifest")
                    request.pop("hs_decision_index"), request.pop("hs_step_index")
                    if sha != expected_manifest:
                        raise ValueError("method manifest mismatch between worker and server")
                    if arm not in ("null", "candidate") or not seeds or want:
                        raise ValueError("Pi0.5 propose: arm null/candidate, nonempty seeds, no sampler extras")
                    natives = [seeded(request, s) for s in seeds]
                    gw._append(calls, {"call": self.count, "endpoint": endpoint, "arm": arm, "seeds": seeds,
                                       "actions": [gw.hashes({"a": a}) for a in natives],
                                       "wall_seconds": time.monotonic() - before})
                    return {"natives": natives, "extras": {}, "cost": {"policy_calls": len(seeds),
                                                                       "gpu_seconds": time.monotonic() - before}}
                raise ValueError("unknown or missing hs_endpoint")
            except BaseException:
                gw._append(calls, {"call": self.count, "status": "error", "error": traceback.format_exc()})
                raise

    server = native_server.WebsocketPolicyServer(HarnessStudioPolicy(), host="127.0.0.1", port=port,
                                                 metadata=policy.metadata)

    async def run():
        async with websockets_server.serve(server._handler, "127.0.0.1", port, compression=None, max_size=None,
                                           process_request=native_server._health_check) as live:
            gw._write(receipt, {"status": "ready", "pid": os.getpid(), "port": port,
                                "checkpoint_identity": manifest["checkpoint_identity"],
                                "method_manifest_sha256": expected_manifest, "frozen_weights": True,
                                "load_seconds": time.monotonic() - started,
                                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "jax_devices": [str(d) for d in devices]})
            await live.serve_forever()
    try:
        asyncio.run(run())
    finally:
        calls.close()


def run_case(manifest, case_id, port, placement):
    import gymnasium as gym
    import numpy as np
    import robocasa  # noqa: F401
    import render_context_fix
    import scene_determinism
    from openpi_client import image_tools, websocket_client_policy
    from robocasa.utils.env_utils import convert_action
    case = next(c for c in manifest["cases"] if c["id"] == case_id)
    directory = Path(manifest["output"]) / "results" / case_id
    directory.mkdir(parents=True, exist_ok=True)
    env = client = None
    result = {"status": "infrastructure_error", "quality_result": False, "case_id": case_id}
    try:
        if manifest["placement"].get(case_id) != placement:
            raise ValueError("Case is running outside its frozen placement")
        if case["seed"] != 7 + 50 * case["task_index"] + case["trial"]:
            raise ValueError("Paired reset seed rule changed")
        determinism = scene_determinism.install()
        determinism["render_context_fix"] = render_context_fix.install()
        np.random.seed(case["seed"])
        env = gym.make("robocasa/" + case["task"], split=manifest["split"], seed=7)
        initial, _ = env.reset(seed=case["seed"])
        client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)
        result = run_episode(env, initial, client, image_tools, convert_action, manifest, case, directory)
        result["scene_determinism"] = determinism
        gw._write(directory / "result.json", result)
    except BaseException:
        result = {"status": "infrastructure_error", "quality_result": False, "case_id": case_id,
                  "error": traceback.format_exc()}
        gw._write(directory / "result.json", result)
    finally:
        for closer in (lambda: client and client._ws.close(), lambda: env and env.close()):
            try:
                closer()
            except BaseException:
                pass
        gw._write(directory / "progress.json", result)
    print("RMH_EPISODE:" + json.dumps(result, sort_keys=True, default=str), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("registry", "server", "episode"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--placement")
    args = parser.parse_args(argv)
    if args.mode == "registry":
        print("RMH_REGISTRY:" + json.dumps(task_registry(), sort_keys=True), flush=True)
        return 0
    manifest = verify_manifest(read_json(args.manifest))
    try:
        if args.mode == "server":
            serve(manifest, args.port, args.receipt)
            return 0
        return 0 if run_case(manifest, args.case_id, args.port, args.placement)["status"] == "complete" else 1
    except BaseException:
        error = {"status": "infrastructure_error", "quality_result": False, "error": traceback.format_exc(), "mode": args.mode}
        if args.mode == "server" and args.receipt:
            gw._write(args.receipt, error)
        print("RMH_WORKER_ERROR:" + json.dumps(error, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
