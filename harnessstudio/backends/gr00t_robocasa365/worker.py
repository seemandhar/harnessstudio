"""HarnessStudio GR00T N1.5 / RoboCasa365 worker (server and episode modes).

Generalized from ChunkTrust runner v2.2 ahs_worker.py (runs/chunktrust-beat-20261026/harnesses/
chunktrust_ahs/source/v2_2/ahs_worker.py). Kept unchanged in behaviour: make_env (scene
determinism + render-context fix), prepare_initial (seeded reset, scene witness, canonical-table
pre-action check, re-render), run_case (case checks, seed rule, placement, parity-abort result,
client timeouts, cleanup) and the main() CLI. The episode loop is HarnessStudio's slot pipeline.
Run with the RobotPolicyBench gr00t-n15 Python.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
COPIED = HERE / "copied"
if str(COPIED) not in sys.path:
    sys.path.insert(0, str(COPIED))
from harnessstudio.bench import groot_native as gw  # noqa: E402
from harnessstudio.bench.common import read_json, verify_manifest  # noqa: E402
import evaluator_log  # noqa: E402


INSTRUCTION_KEY = "annotation.human.task_description"


def with_instruction(observation, text):
    """Same observation with the public instruction replaced (shape kept, no truncation)."""
    import numpy as np
    old = np.asarray(observation[INSTRUCTION_KEY])
    new = np.array([text] * old.size, dtype=object if old.dtype.kind == "O" else None).reshape(old.shape)
    return dict(observation, **{INSTRUCTION_KEY: new})
import parity  # noqa: E402
import render_context_fix  # noqa: E402
import scene_determinism  # noqa: E402
from harnessstudio.core.errors import MethodContractError  # noqa: E402

OBSERVATION_STRIDE = 16  # declared common contract constant (native chunk); executed k recorded separately
RUNNER_VERSION = "harnessstudio-gr00t-1"


class GrootEnvPort:
    """Executes native GR00T actions through MultiStepWrapper (spec §5.1). It records NO video:
    the pipeline's before_decision callback appends exactly one pre-action frame per decision."""
    def __init__(self, env):
        self.env = env
        if env.n_action_steps != 16:
            raise ValueError("Native MultiStep n_action_steps changed")

    def execute(self, native, start, count):
        import numpy as np
        from harnessstudio.core.pipeline import StepResult
        actions = native if start == 0 else {k: v[start:] for k, v in native.items()}
        self.env.n_action_steps = count
        try:
            public, _, terminated, truncated, info = self.env.step(actions)
        finally:
            self.env.n_action_steps = 16
        executed = len(info["rewards"])
        success = bool(np.asarray(info["success"]).reshape(-1)[0])
        return StepResult(gw.public_copy(public), success, executed, bool(terminated or truncated))


class GrootClient:
    """PolicyClient over the GR00T zmq RPC."""
    def __init__(self, client, manifest_sha256, before_call, after_call):
        self.client, self.manifest_sha256 = client, manifest_sha256
        self.before, self.after = before_call, after_call

    def _call(self, endpoint, payload):
        self.before()
        started, ok = time.monotonic(), False
        try:
            out = self.client.call_endpoint(endpoint, payload)
            ok = True
            return out
        finally:
            self.after(ok, time.monotonic() - started)

    def get_action_seeded(self, observation, seed):
        native = self._call("get_action_seeded", {"observation": gw.public_copy(observation), "noise_seed": seed})
        return native, gw.flatten_native(native)

    def propose(self, observation, *, arm, decision_index, step_index, seeds, want):
        from harnessstudio.core.runtime import ProposeResult
        out = self._call("propose", {"observation": gw.public_copy(observation), "arm": arm,
                                     "decision_index": decision_index, "step_index": step_index,
                                     "seeds": list(seeds), "want": list(want),
                                     "method_manifest_sha256": self.manifest_sha256})
        natives = out["natives"]
        return ProposeResult(natives, [gw.flatten_native(n) for n in natives], out["extras"], out["cost"])


def run_episode(env, initial, client, manifest, case, directory, *, port_number=0, evaluator=None,
                step_tap=None, video_factory=None):
    """One reset instance after the pre-action check. CPU tests inject a native-shaped env/RPC.

    ``evaluator`` holds evaluator-only facts (scene witness, parity report, placement); it is
    written to the record and never passed to the port, modules or harness.
    """
    from harnessstudio.core.arms import Arm
    from harnessstudio.core.capabilities import unified_names
    from harnessstudio.core.pipeline import EpisodeSpec, run_episode as studio_episode
    from harnessstudio.eval.loading import method_from_manifest
    from harnessstudio.policies import gr00t_n15 as g
    from harnessstudio.runtime import PolicySpec
    evaluator = dict(evaluator or {})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    core_traces = []
    config = manifest["harness_config"]
    protocol = manifest["protocol"]
    record = {"status": "running", "quality_result": False, "runner_version": RUNNER_VERSION,
        "case_id": case["id"], "task": case["task"], "trial": case["trial"], "horizon": case["horizon"],
        "harness": manifest["harness_identity"], "arm": config["arm"], "method": config["method"],
        "harness_config_sha256": manifest["harness_config_sha256"],
        "method_manifest_sha256": manifest["method_manifest"]["sha256"],
        "checkpoint_identity": manifest["checkpoint_identity"],
        # C5 contract fields (evaluation.REQUIRED_FIELDS / CONTRACT_FIELDS)
        "seed": case["seed"], "observation_stride": OBSERVATION_STRIDE,
        "protocol_sha256": manifest["protocol_sha256"], "checkpoint_sha256": manifest["checkpoint_identity"],
        "seed_mode": "seeded_reset", "native_reset_seed": case["seed"],
        "steps": 0, "policy_dispatch_attempts": 0, "policy_calls_completed": 0, "chunks": 0, "success": False,
        "action_chunk": 16, "control_hz": 20, "denoising_steps": 4,
        "noise_model": "N1_crn" if protocol["noise"]["crn"] else "N2_unpaired",
        "initial_scene_sha256": evaluator.get("initial_scene_sha256"),
        "scene_components": evaluator.get("scene_components"),
        "parity": evaluator.get("parity"), "placement": evaluator.get("placement"),
        "initial_packet": evaluator.get("initial_packet"),
        "scene_determinism": evaluator.get("scene_determinism"),
        "port": port_number, "training_steps": 0, "policy_seconds": 0.0,
        "actual_UnifiedHarness_instantiated": False, "manifest_sha256": manifest.get("manifest_digest")}

    def progress():
        gw._write(directory / "progress.json", record)
    trace = (directory / "trace.jsonl").open("x", buffering=1)
    # Every arm records video, one pre-action frame per decision (RI-2, plan clarification 8).
    video = (video_factory or gw.PublicVideo)(directory / "episode.mp4")
    rpc_seconds = [0.0]

    def before_call():
        record["policy_dispatch_attempts"] += 1
        progress()

    def after_call(completed, seconds):
        record["policy_calls_completed"] += int(completed)
        rpc_seconds[0] += seconds
        progress()
    try:
        method = method_from_manifest(manifest["method_manifest"])
        arm = Arm(config["arm"])
        public = gw.public_copy(initial)
        record["initial_observation_sha256"] = gw.hashes(public)["__aggregate__"]  # P2.1 strict
        horizon = case["horizon"]
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("Invalid official horizon")
        baseline = None
        if arm is Arm.NULL:
            baseline = json.loads((Path(manifest["baseline_run"]) / "results" / case["id"]
                                   / "core-traces.json").read_text())
        spec = EpisodeSpec(case_id=case["id"], horizon=horizon, chunk_len=16, action_dim=12, control_hz=20.0,
                           benchmark="RoboCasa365", public_keys=tuple(sorted(gw.PUBLIC_KEYS)),
                           noise_namespace=protocol["noise"]["namespace"], instruction_of=gw.instruction,
                           with_instruction=with_instruction, input_hashes=gw.hashes, action_hashes=gw.hashes,
                           method_manifest_sha256=manifest["method_manifest"]["sha256"],
                           action_layout=g.ACTION_LAYOUT, proprio_eef=g.PROPRIO_EEF)
        policy_spec = PolicySpec(g.POLICY_ID, manifest["checkpoint_identity"], unified_names(g.CAPABILITIES))
        port_client = GrootClient(client, manifest["method_manifest"]["sha256"], before_call, after_call)
        record["actual_UnifiedHarness_instantiated"] = True
        progress()
        result = studio_episode(method, arm, spec, public, GrootEnvPort(env), port_client, policy_spec,
                                trace=lambda row: gw._append(trace, row), baseline_core_traces=baseline,
                                before_decision=video.append)
        core_traces.extend(result.pop("core_traces"))
        gw._write(directory / "unified-manifest.json", dict(result.pop("unified_manifest"),
                  runner_version=RUNNER_VERSION, adapter_exception=(
                      "variable execute_steps via StudioUnifiedHarness.commit; a policy call counts one sample")))
        record.update(result)
        record["chunks"] = result["decisions"]  # RI-2: frames == chunks == action_dispatch rows
        record["rpc_seconds"] = rpc_seconds[0]
        record.update(status="complete", quality_result=True, validity="valid")
    except MethodContractError:
        record.update(status="complete", quality_result=False, validity="invalid_method_defect",
                      error=traceback.format_exc())
    except BaseException:
        record.update(status="infrastructure_error", quality_result=False, validity="infrastructure",
                      error=traceback.format_exc())
    finally:
        try:
            video.close()
        except BaseException:
            record.update(status="infrastructure_error", quality_result=False, validity="infrastructure",
                          error=traceback.format_exc())
        if step_tap is not None:
            try:
                record["evaluator_step_log"] = step_tap.save(directory / "evaluator_steps.npz")
            except BaseException:
                record.update(status="infrastructure_error", quality_result=False, validity="infrastructure",
                              error=traceback.format_exc())
        record["wall_seconds"] = time.monotonic() - started
        trace.close()
        gw._write(directory / "core-traces.json", core_traces)
        progress()
        gw._write(directory / "result.json", record)
    return record


def make_env(task, horizon, split, receipt=None):
    # v2.1: before any robocasa model is built (fail closed if the simulator source differs).
    applied = scene_determinism.install()
    applied["render_context_fix"] = render_context_fix.install()  # v2.2, before any render context exists
    if receipt is not None:
        receipt.update(applied)
    from gr00t.eval import simulation
    config = simulation.SimulationConfig(env_name="robocasa/" + task,
        split=split, n_envs=1, n_episodes=1,
        video=simulation.VideoConfig(video_dir=None),
        multistep=simulation.MultiStepConfig(n_action_steps=16, max_episode_steps=horizon))
    env = simulation._create_single_env(config, 0)
    if env.unwrapped.env.control_freq != 20:
        raise ValueError("Native controller frequency changed")
    return env


def prepare_initial(env, case, manifest, directory, placement):
    """Evaluator-only: reset, scene witness, packet save, pre-action check. No action."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    initial, _ = env.reset(seed=case["seed"])
    public = gw.public_copy(initial)
    witness = parity.scene_witness(env)
    parity.save_packet(directory / "initial_public_reset.npz", public)
    table = manifest.get("canonical_table")
    try:
        public, report = parity.preaction_check(env, public, case["id"], table, hash_fn=gw.hashes,
                                                max_rerenders=manifest["protocol"]["parity"]["max_rerenders"])
    except parity.ParityAbort as abort:
        abort.report.update(initial_scene_sha256=witness["initial_scene_sha256"],
                            initial_observation_sha256=gw.hashes(public)["__aggregate__"])
        raise
    if report.get("rerenders"):
        parity.save_packet(directory / "initial_public.npz", public)
        packet = "initial_public.npz"
    else:
        packet = "initial_public_reset.npz"
    return public, {"initial_scene_sha256": witness["initial_scene_sha256"],
                    "scene_components": witness["components"], "parity": report,
                    "placement": placement, "initial_packet": packet}


def run_case(manifest, case_id, port, placement):
    directory = Path(manifest["output"]) / "results" / case_id
    directory.mkdir(parents=True, exist_ok=True)
    env = client = None
    result = {"status": "infrastructure_error", "quality_result": False, "case_id": case_id}
    try:
        import zmq
        from gr00t.eval import simulation
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        matches = [c for c in manifest["cases"] if c["id"] == case_id]
        if len(matches) != 1:
            raise ValueError("Case must identify exactly one predeclared episode")
        case = matches[0]
        if get_task_horizon(case["task"]) != case["horizon"]:
            raise ValueError("Official task horizon changed")
        if case["seed"] != 7 + 50 * case["task_index"] + case["trial"]:
            raise ValueError("Paired reset seed rule changed")
        if manifest["placement"].get(case_id) != placement:
            raise ValueError("Case is running outside its frozen placement")
        determinism = {}
        env = make_env(case["task"], case["horizon"], manifest["split"], receipt=determinism)
        try:
            public, evaluator = prepare_initial(env, case, manifest, directory, placement)
            evaluator["scene_determinism"] = determinism
        except parity.ParityAbort as abort:
            result = {"status": "infrastructure_error", "quality_result": False, "case_id": case_id,
                      "task": case["task"], "trial": case["trial"], "runner_version": RUNNER_VERSION,
                      "parity_abort": True, "parity": abort.report, "physical_steps_attempted": 0,
                      "policy_calls": 0, "placement": placement, "error": str(abort)}
            gw._write(directory / "result.json", result)
            gw._write(directory / "progress.json", result)
            print("RMH_EPISODE:" + json.dumps(result, sort_keys=True, default=str), flush=True)
            return result
        step_tap = evaluator_log.install(env) if manifest.get("evaluator_step_log", False) else None
        client = simulation.SimulationInferenceClient(host="127.0.0.1", port=port)
        client.socket.setsockopt(zmq.RCVTIMEO, 180000)
        client.socket.setsockopt(zmq.SNDTIMEO, 180000)
        result = run_episode(env, public, client, manifest, case, directory, port_number=port,
                             evaluator=evaluator, step_tap=step_tap)
    except BaseException:
        result = {"status": "infrastructure_error", "quality_result": False,
                  "case_id": case_id, "error": traceback.format_exc()}
        gw._write(directory / "result.json", result)
        gw._write(directory / "progress.json", result)
    finally:
        for name, closer in (("close_error", lambda: env and env.close()),
                             ("client_close_error", lambda: client and client.socket.close(linger=0))):
            try:
                closer()
            except BaseException:
                result.update(status="infrastructure_error", quality_result=False, **{name: traceback.format_exc()})
                gw._write(directory / "result.json", result)
                gw._write(directory / "progress.json", result)
    print("RMH_EPISODE:" + json.dumps(result, sort_keys=True, default=str), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("server", "episode"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18925)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--placement")
    args = parser.parse_args(argv)
    manifest = None
    try:
        manifest = verify_manifest(read_json(args.manifest))
        if args.mode == "server":
            if args.receipt is None:
                parser.error("--receipt is required for server")
            from harnessstudio.host.server import serve
            serve(manifest, args.port, args.receipt)
            return 0
        if args.case_id is None or args.placement is None:
            parser.error("--case-id and --placement are required for episode")
        return 0 if run_case(manifest, args.case_id, args.port, args.placement)["status"] == "complete" else 1
    except BaseException:
        error = {"status": "infrastructure_error", "quality_result": False,
                 "error": traceback.format_exc(), "mode": args.mode}
        if args.mode == "server" and args.receipt:
            gw._write(args.receipt, error)
        elif args.mode == "episode" and manifest and args.case_id:
            directory = Path(manifest["output"]) / "results" / args.case_id
            gw._write(directory / "result.json", error)
            gw._write(directory / "progress.json", error)
        print("RMH_WORKER_ERROR:" + json.dumps(error, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
