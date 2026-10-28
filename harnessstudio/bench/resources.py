"""Conservative GPU reservations for cooperating harness agents.

This module never launches GPU work or kills processes. Reservations use whole
GPUs; ``instances_per_gpu`` describes concurrency inside one owner's reservation,
not permission for unrelated owners to share a GPU. Locks protect cooperating
agents only, so callers must still check inventory immediately before launching.

Leases persist until explicit release, even after the acquiring process exits.
This supports separate CLI calls and avoids reclaiming a GPU whose child jobs
survived their controller. PID identity is diagnostic, never an eviction timer.
"""

from __future__ import annotations

import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile


class ResourceError(RuntimeError):
    """Invalid inventory, unsafe capacity, or a conflicting reservation."""


def _query(fields: str, kind: str) -> list[list[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceError(f"Cannot inspect GPUs with nvidia-smi: {exc}") from exc
    return [row for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True)
            if row]


def snapshot() -> list[dict]:
    """Read GPU capacity and live compute processes without starting GPU work.

    A failed process query is represented explicitly; acquire refuses inventory
    with unknown process occupancy rather than treating missing data as idle.
    Graphics/display memory also counts against ``max_used_mib`` and capacity.
    """
    rows = _query("index,uuid,memory.total,memory.used,utilization.gpu", "gpu")
    inventory = []
    try:
        for row in rows:
            index, uuid, total, used, utilization = [cell.strip() for cell in row]
            inventory.append({
                "index": int(index), "uuid": uuid,
                "total_mib": int(total), "used_mib": int(used),
                "utilization_percent": int(utilization),
                "processes": [], "processes_known": True,
            })
    except (TypeError, ValueError) as exc:
        raise ResourceError("nvidia-smi returned unsupported GPU inventory") from exc
    try:
        processes = _query("gpu_uuid,pid,process_name,used_gpu_memory", "compute-apps")
        by_uuid = {gpu["uuid"]: gpu for gpu in inventory}
        for row in processes:
            uuid, pid, name, used = [cell.strip() for cell in row]
            if uuid in by_uuid:
                by_uuid[uuid]["processes"].append({
                    "pid": int(pid), "name": name,
                    "used_mib": int(used) if used.isdecimal() else None,
                })
    except (ResourceError, TypeError, ValueError) as exc:
        for gpu in inventory:
            gpu["processes_known"] = False
            gpu["process_query_error"] = str(exc)
    return inventory


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ResourceError(f"{name} must be an integer >= {minimum}")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResourceError(f"{name} must be an integer >= {minimum}") from exc
    if str(parsed) != str(value) or parsed < minimum:
        raise ResourceError(f"{name} must be an integer >= {minimum}")
    return parsed


def _owner(owner: str) -> str:
    if not isinstance(owner, str) or not owner.strip() or len(owner) > 256:
        raise ResourceError("owner must be a nonempty run identifier (<=256 characters)")
    return owner


def _identity() -> dict:
    pid = os.getpid()
    proc = Path(f"/proc/{pid}")
    identity = {"pid": pid, "process_start_ticks": None,
                "process_group": os.getpgrp(), "command": [], "cwd": os.getcwd(),
                "boot_id": None}
    try:
        identity["process_start_ticks"] = int(
            (proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
        identity["command"] = [part.decode(errors="replace") for part in
                               (proc / "cmdline").read_bytes().split(b"\0") if part]
        identity["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        pass
    return identity


@contextlib.contextmanager
def _locked(lease_dir):
    directory = Path(lease_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".resources.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            path = directory / "resources.json"
            if path.exists():
                try:
                    state = json.loads(path.read_text())
                except (OSError, ValueError) as exc:
                    raise ResourceError(f"Cannot read lease registry {path}") from exc
                if (not isinstance(state, dict) or state.get("version") != 1
                        or not isinstance(state.get("leases"), dict)):
                    raise ResourceError(f"Invalid lease registry {path}")
            else:
                state = {"version": 1, "leases": {}}
            yield directory, state
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _save(directory: Path, state: dict):
    fd, temporary = tempfile.mkstemp(prefix=".resources-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "resources.json")
        dirfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def acquire(lease_dir, owner, gpu_ids, *, inventory=None, max_used_mib=2048,
            reserve_mib=2048, instances_per_gpu=1, calibrated_instance_mib=None) -> dict:
    """Atomically reserve all requested GPUs or raise ResourceError.

    Every visible compute process is considered external and blocks a new lease,
    even when its memory use is below the threshold. More than one instance per
    GPU requires a measured peak per-instance memory estimate. That estimate plus
    existing memory and reserve must fit on every requested GPU.

    Repeating the identical reservation for the same owner is idempotent and
    does not re-probe running owned jobs. Changed reservations require release.
    """
    owner = _owner(owner)
    try:
        ids = [_integer(value, "gpu_id") for value in gpu_ids]
    except TypeError as exc:
        raise ResourceError("gpu_ids must be a nonempty collection") from exc
    if not ids or len(ids) != len(set(ids)):
        raise ResourceError("gpu_ids must be nonempty and unique")
    ids.sort()
    threshold = _integer(max_used_mib, "max_used_mib")
    reserve = _integer(reserve_mib, "reserve_mib")
    instances = _integer(instances_per_gpu, "instances_per_gpu", 1)
    calibrated = (None if calibrated_instance_mib is None else
                  _integer(calibrated_instance_mib, "calibrated_instance_mib", 1))
    if instances > 1 and calibrated is None:
        raise ResourceError("Multiple instances require calibrated_instance_mib")
    spec = {"gpu_ids": ids, "instances_per_gpu": instances,
            "calibrated_instance_mib": calibrated, "max_used_mib": threshold,
            "reserve_mib": reserve}
    with _locked(lease_dir) as (directory, state):
        existing = state["leases"].get(owner)
        if existing:
            if all(existing.get(key) == value for key, value in spec.items()):
                return existing
            raise ResourceError(f"Owner {owner!r} already has a different lease; release it first")
        for other, lease in state["leases"].items():
            overlap = set(ids) & set(lease.get("gpu_ids", []))
            if overlap:
                raise ResourceError(f"GPU(s) {sorted(overlap)} already reserved by {other!r}")
        observed = snapshot() if inventory is None else inventory
        try:
            by_id = {}
            for gpu in observed:
                index = _integer(gpu["index"], "inventory index")
                if index in by_id:
                    raise ResourceError(f"Duplicate GPU {index} in inventory")
                by_id[index] = gpu
        except (TypeError, KeyError) as exc:
            raise ResourceError("Malformed GPU inventory") from exc
        selected = []
        for index in ids:
            if index not in by_id:
                raise ResourceError(f"GPU {index} is absent from inventory")
            gpu = by_id[index]
            total = _integer(gpu.get("total_mib"), "total_mib", 1)
            used = _integer(gpu.get("used_mib"), "used_mib")
            if not gpu.get("processes_known", True):
                raise ResourceError(f"GPU {index} process occupancy is unknown")
            if gpu.get("processes"):
                raise ResourceError(f"GPU {index} has live external compute processes")
            if used > threshold:
                raise ResourceError(f"GPU {index} is busy: {used} MiB > {threshold} MiB")
            required = reserve + used + instances * (calibrated or 0)
            if required > total:
                raise ResourceError(f"GPU {index} lacks capacity: {required} MiB > {total} MiB")
            selected.append(gpu)
        lease = {"owner": owner, **spec, **_identity(),
                 "created_at": datetime.now(timezone.utc).isoformat(),
                 "inventory_at_acquire": selected}
        state["leases"][owner] = lease
        _save(directory, state)
        return lease


def release(lease_dir, owner) -> dict:
    """Release only the named reservation; never stop any process.

    Callers must ensure their jobs have finished before release. A controller PID
    exiting does not establish that its children stopped. Owner IDs should be
    unique per run and must not be borrowed by another agent.
    """
    owner = _owner(owner)
    with _locked(lease_dir) as (directory, state):
        lease = state["leases"].pop(owner, None)
        if lease is not None:
            _save(directory, state)
        return {"owner": owner, "released": lease is not None, "lease": lease}
