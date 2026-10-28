"""Scheduler for the HarnessStudio GR00T worker. Generalized from ChunkTrust runner v2.2 ahs_runner.py.

Same semantics as the original RoboMetaHarness runner run()/worker(): shared leases, one
persistent policy server per measured-idle GPU, isolated episode client per case, round-robin
case assignment, P2.4 parity retry with zero executed actions, P2.5 placement, no rerolls,
deadline, sealed results (completed.json). Differences from v2.2: child processes run the FROZEN
copy of worker.py with the frozen closure first on PYTHONPATH.
"""
from __future__ import annotations

import fcntl
import os
import shutil
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE / "copied") not in sys.path:
    sys.path.insert(0, str(HERE / "copied"))
from harnessstudio.bench.common import (case_state, group_members, now, process_identity,  # noqa: E402
                                                  read_json, sha256, valid_result, verify_manifest,
                                                  write_json)
from harnessstudio import config  # noqa: E402
from harnessstudio.bench.registry import child_environment  # noqa: E402
from harnessstudio.bench.runner_util import (COORDINATOR_POLL_SECONDS, _wait, stop_owned_group,  # noqa: E402
                                                 summarize, topology_and_idle)
import scene_determinism  # noqa: E402

ADAPTER = HERE / "adapter.py"
WORKER_REL = Path("harnessstudio/backends/gr00t_robocasa365/worker.py")


def child_paths(manifest) -> list:
    frozen = Path(manifest["frozen_closure"]["root"])
    return [str(frozen), str(frozen / "plugins")]


def studio_environment(manifest, *, role="episode", **kw):
    """Child environment: frozen closure first, then backend-declared paths (manifest['backend'])."""
    env = child_environment(manifest["policybench"], **kw)
    backend = manifest.get("backend") or {}
    extra = [str(p) for p in backend.get(f"{role}_paths", [])]
    env["PYTHONPATH"] = os.pathsep.join(child_paths(manifest) + extra + [env["PYTHONPATH"]])
    env.update({k: str(v) for k, v in backend.get(f"{role}_env", {}).items()})
    # Credentials: children get none, except the variables a controller method names, and only
    # its episode worker gets those (values come from this process's environment, never a file).
    from harnessstudio.core.llm import credential_names
    allowed = set(credential_names(manifest.get("method_manifest", {}).get("method", {}))) if role == "episode" else set()
    for key in [k for k in env if k.endswith("_API_KEY")]:
        if key not in allowed:
            env.pop(key)
    for key in allowed:  # child_environment strips every *_API_KEY; re-add only the declared ones
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def backend_commands(manifest):
    """(server argv prefix, episode argv prefix). Defaults are the GR00T backend's (older manifests)."""
    backend = manifest.get("backend") or {}
    bench = Path(manifest["policybench"])
    frozen = Path(manifest["frozen_closure"]["root"])
    worker = str(frozen / backend.get("worker_rel", str(WORKER_REL)))
    server_python = backend.get("server_python", str(bench / "envs/gr00t-n15/bin/python"))
    episode_python = backend.get("episode_python", str(bench / "envs/gr00t-n15/bin/python"))
    return [server_python, worker], [episode_python, worker]


def worker(manifest_path, gpu, slot=0):
    manifest = verify_manifest(manifest_path)
    output = Path(manifest["output"])
    state = read_json(output / "state.json")
    attempt = state["attempts"][-1]
    directory = output / "logs" / attempt["id"] / f"gpu-{gpu}-slot-{slot}"
    directory.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server_base, episode_base = backend_commands(manifest)
    env = studio_environment(manifest, role="server", gpu=gpu)
    render = scene_determinism.client_environment(
        studio_environment(manifest, role="episode", egl=state["mapping"][str(gpu)]["egl_index"]))
    server = client = None
    receipt = directory / "server-ready.json"
    record = {"gpu": gpu, "slot": slot, "status": "starting", "identity": process_identity(os.getpid()),
              "episode_children": [], "started_at": now()}
    status_path = directory / "worker.json"
    write_json(status_path, record)
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server_argv = server_base + ["--mode", "server", "--manifest", str(manifest_path),
                              "--port", str(port), "--receipt", str(receipt)]
        with (directory / "server.log").open("x") as log:
            server = subprocess.Popen(server_argv, env=env, cwd=config.workspace(), stdout=log, stderr=subprocess.STDOUT)
        record.update(server_identity=process_identity(server.pid), port=port, server_argv=server_argv)
        write_json(status_path, record)
        load_deadline = min(time.time() + 360, manifest["deadline"] - 15)
        while not receipt.exists():
            if stopped or time.time() >= load_deadline or server.poll() is not None:
                raise RuntimeError("Server startup stopped/timed out; see server.log")
            time.sleep(.25)
        ready = read_json(receipt)
        if (ready.get("pid") != server.pid or ready.get("checkpoint_identity") != manifest["checkpoint_identity"]
                or ready.get("method_manifest_sha256") != manifest["method_manifest"]["sha256"]):
            raise RuntimeError("Readiness receipt does not belong to owned policy server")
        record["status"] = "running"
        write_json(status_path, record)
        key = slot_key(gpu, slot)
        for case in manifest["assignments"][key]:
            if manifest["placement"][case["id"]] != key:
                raise RuntimeError("Frozen placement map disagrees with slot assignment")
            if case_state(output, case, manifest_digest=manifest["manifest_digest"]) != "pending":
                continue
            attempt_no = 0
            while True:
                if stopped or time.time() >= manifest["deadline"] - 30:
                    raise TimeoutError("No time to start another episode")
                verify_manifest(manifest)
                target = output / "results" / case["id"]
                target.mkdir(exist_ok=False, parents=True)
                write_json(target / "started.json", {"at": now(), "case": case, "gpu": gpu, "slot": slot,
                    "placement": key, "parity_attempt": attempt_no,
                    "worker_identity": record["identity"], "manifest_sha256": sha256(manifest_path)})
                client_argv = episode_base + ["--mode", "episode", "--manifest", str(manifest_path),
                                      "--port", str(port), "--case-id", case["id"], "--placement", key]
                with (target / "client.log").open("x") as log:
                    client = subprocess.Popen(client_argv, env=render, cwd=config.workspace(), stdout=log, stderr=subprocess.STDOUT)
                record["episode_children"].append({"case": case["id"], "identity": process_identity(client.pid),
                                                   "argv": client_argv, "parity_attempt": attempt_no})
                write_json(status_path, record)
                _wait(client, server, manifest["deadline"], lambda: stopped)
                verify_manifest(manifest)
                result_path = target / "result.json"
                result = read_json(result_path) if result_path.exists() else {}
                decision = parity_retry_decision(result, attempt_no,
                                                 manifest["protocol"]["parity"]["fresh_process_retries"])
                record.setdefault("parity_events", []).append({"case": case["id"], "attempt": attempt_no,
                                                                "decision": decision, "parity": result.get("parity")})
                write_json(status_path, record)
                if decision == "retry":
                    # Zero actions were executed: preserve the aborted attempt and use a fresh process.
                    keep = output / "parity_aborts" / f"{case['id']}--attempt-{attempt_no}"
                    keep.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(target), str(keep))
                    attempt_no += 1
                    continue
                break
            if decision == "invalid":
                continue  # still unmatched after the retry: case invalid for all arms; phase incomplete
            if client.returncode or not result_path.exists() or not valid_result(result, case):
                raise RuntimeError("Episode infrastructure error; retained without reroll: " + case["id"])
            write_json(target / "completed.json", {"at": now(), "result_sha256": sha256(result_path),
                       "manifest_sha256": sha256(manifest_path), "manifest_digest": manifest["manifest_digest"],
                       "client_exit_code": client.returncode})
        record.update(status="complete", ended_at=now())
    except BaseException as exc:
        record.update(status="infrastructure_error", error=repr(exc), ended_at=now())
    finally:
        for process in (client, server):
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                process.wait()
        write_json(status_path, record)
    return 0 if record["status"] == "complete" else 1


def parity_retry_decision(result, attempts_done, max_retries):
    """P2.4: 'retry' only for a pre-action parity abort with zero executed actions."""
    if not result.get("parity_abort"):
        return "none"
    if result.get("physical_steps_attempted", None) != 0 or result.get("policy_calls", None) != 0:
        raise RuntimeError("parity abort after actions/policy calls is an invariant violation")
    return "retry" if attempts_done < max_retries else "invalid"


def placement_map(assignments):
    return {case["id"]: key for key, cases in assignments.items() for case in cases}


def slot_key(gpu, slot):
    return f"{int(gpu)}.{int(slot)}"


def plan_slots(gpu_ids, workers_per_gpu, n_cases):
    """GPU-major interleave so the first cases spread across GPUs before stacking slots."""
    if type(workers_per_gpu) is not int or workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be a positive integer")
    slots = [slot_key(g, s) for s in range(workers_per_gpu) for g in gpu_ids]
    return slots[:max(1, min(len(slots), n_cases))]


def run(manifest_path, *, resume=False):
    from harnessstudio.bench import resources
    manifest_path = Path(manifest_path)
    manifest = verify_manifest(manifest_path)
    output = Path(manifest["output"])
    with (output / ".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "state.json"
        state = read_json(path) if path.exists() else {"attempts": []}
        if state.get("manifest_digest", manifest["manifest_digest"]) != manifest["manifest_digest"]:
            raise RuntimeError("Resume manifest differs from original execution receipt")
        if state["attempts"] and not resume:
            raise ValueError("Run already started. Use --resume; completed and interrupted scenes are not rerolled.")
        for attempt in state["attempts"]:
            controller = attempt.get("controller")
            if controller and process_identity(controller["pid"]) == controller:
                raise RuntimeError("Prior controller is still alive")
            for record in attempt.get("workers", {}).values():
                if group_members(record["pgid"]):
                    raise RuntimeError("Prior owned group still alive; reconcile before resume")
        summary = summarize(manifest)
        if not summary["pending_cases"]:
            if summary["status"] == "complete":
                summarize(manifest, force_evidence=True)
                return 0
            return 2
        if time.time() >= manifest["deadline"] - 30:
            raise TimeoutError("Original run deadline exhausted; resume does not renew its budget")
        slots = [key for key in manifest["assignments"] if any(
            case_state(output, case, manifest_digest=manifest["manifest_digest"]) == "pending"
            for case in manifest["assignments"][key])]
        ids = sorted({int(key.split(".")[0]) for key in slots})
        inventory, mapping = topology_and_idle(ids, manifest["policybench"])
        lease_dir = config.leases_dir()
        owner = manifest["lease_owner"]
        leases = read_json(lease_dir / "resources.json") if (lease_dir / "resources.json").exists() else {}
        if owner in leases.get("leases", {}):
            resources.release(lease_dir, owner)
        workers_per_gpu = manifest.get("workers_per_gpu", 1)
        resources.acquire(lease_dir, owner, ids, inventory=inventory, instances_per_gpu=workers_per_gpu,
                          calibrated_instance_mib=manifest.get("calibrated_worker_mib"))
        attempt = {"id": f"attempt-{len(state['attempts']):03d}", "started_at": now(),
                   "started_epoch": time.time(), "gpu_ids": ids, "slots": slots,
                   "controller": process_identity(os.getpid()), "workers": {}, "status": "running"}
        state.update(mapping=mapping, status="running", manifest_digest=manifest["manifest_digest"])
        state["attempts"].append(attempt)
        processes = {}
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
        old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            write_json(path, state)
            for key in slots:
                gpu, slot = (int(v) for v in key.split("."))
                directory = output / "logs" / attempt["id"] / f"gpu-{gpu}-slot-{slot}"
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / "launcher.log").open("x") as log:
                    process = subprocess.Popen([sys.executable, str(ADAPTER), "_worker",
                        "--manifest", str(manifest_path), "--gpu", str(gpu), "--slot", str(slot)], cwd=config.workspace(),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes[key] = process
                record = process_identity(process.pid)
                if record is None:
                    raise RuntimeError("Worker exited before receipt")
                attempt["workers"][key] = record
                write_json(path, state)
            last = None
            cleanup_reserve = 15 + 7 * len(processes) + COORDINATOR_POLL_SECONDS
            while any(p.poll() is None for p in processes.values()):
                if stopped or time.time() >= manifest["deadline"] - cleanup_reserve:
                    break
                summary = summarize(manifest)
                current = (summary["completed_episodes"], len(summary["blocked_cases"]))
                if current != last:
                    print(f"{current[0]}/{summary['expected_episodes']} complete, "
                          f"{summary['successes']} successes, {current[1]} interrupted/errors", flush=True)
                    last = current
                time.sleep(COORDINATOR_POLL_SECONDS)
        except BaseException as exc:
            attempt["error"] = repr(exc)
        finally:
            cleanup_errors = []
            for key, process in processes.items():
                try:
                    record = attempt["workers"].get(key)
                    if record:
                        stop_owned_group(record)
                    elif group_members(process.pid):
                        raise RuntimeError("Worker identity missing; reservation retained")
                    process.wait(timeout=5)
                except BaseException as exc:
                    cleanup_errors.append(repr(exc))
            attempt.update(ended_at=now(), wall_seconds=time.time() - attempt["started_epoch"],
                gpu_hours_upper_bound=(time.time() - attempt["started_epoch"]) * len(ids) / 3600,
                worker_hours_upper_bound=(time.time() - attempt["started_epoch"]) * len(slots) / 3600,
                cleanup_errors=cleanup_errors, workers_reaped=not cleanup_errors)
            if not cleanup_errors:
                resources.release(lease_dir, owner)
            attempt["lease_released"] = not cleanup_errors
            summary = summarize(manifest, force_evidence=True)
            attempt["status"] = ("complete" if summary["status"] == "complete" and not cleanup_errors
                                 else "stopped" if stopped else "incomplete")
            state["status"] = attempt["status"]
            state["gpu_hours_upper_bound"] = sum(a.get("gpu_hours_upper_bound", 0) for a in state["attempts"])
            write_json(path, state)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return 0 if state["status"] == "complete" else 2
