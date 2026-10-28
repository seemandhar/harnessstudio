"""Run summaries, owned-process stop, GPU topology/idle checks.

Vendored from RoboMetaHarness robometaharness/benchmarking/runner.py lines 1-198
(see bench/PROVENANCE.json); scheduling itself lives in the HarnessStudio backend runner.
"""
from collections import OrderedDict
import csv
import fcntl
import io
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .common import (case_state, group_members, now, process_identity, read_json,
                     sha256, stat_fingerprint, valid_result, verify_manifest, verify_manifest_digest, write_json)
from .registry import child_environment


# Process-local metadata cache. Digests remain bound to the immutable summary;
# cache hits only avoid reading unchanged evidence bytes during polling.
EVIDENCE_CACHE_MAX = 100_000
_EVIDENCE_SHA_CACHE = OrderedDict()
COORDINATOR_POLL_SECONDS = 10


def _evidence_sha256(path, *, force=False):
    path = str(path)
    try:
        before = stat_fingerprint(path)
        cached = _EVIDENCE_SHA_CACHE.get(path)
        if not force and cached is not None and cached[0] == before:
            _EVIDENCE_SHA_CACHE.move_to_end(path)
            return cached[1]
        digest = sha256(path)
        after = stat_fingerprint(path)
    except OSError as exc:
        _EVIDENCE_SHA_CACHE.pop(path, None)
        raise RuntimeError('Completed episode evidence missing/unreadable: ' + path) from exc
    if before != after:
        _EVIDENCE_SHA_CACHE.pop(path, None)
        raise RuntimeError('Completed episode evidence changed during hashing: ' + path)
    _EVIDENCE_SHA_CACHE[path] = (after, digest)
    _EVIDENCE_SHA_CACHE.move_to_end(path)
    while len(_EVIDENCE_SHA_CACHE) > EVIDENCE_CACHE_MAX:
        _EVIDENCE_SHA_CACHE.popitem(last=False)
    return digest


def summarize(manifest, *, force_evidence=False):
    with (Path(manifest['output']) / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _summarize(manifest, force_evidence=force_evidence)


def _summarize(manifest, *, force_evidence=False):
    verify_manifest_digest(manifest)
    output = Path(manifest['output'])
    if (output / 'manifest.json').exists():
        disk = verify_manifest_digest(read_json(output / 'manifest.json'))
        if disk['manifest_digest'] != manifest['manifest_digest']:
            raise RuntimeError('Frozen manifest changed during execution')
    previous_path = output / 'summary.json'
    previous = read_json(previous_path) if previous_path.exists() else {}
    if previous.get('manifest_digest', manifest['manifest_digest']) != manifest['manifest_digest']:
        raise RuntimeError('Summary belongs to a different frozen manifest')
    bindings = previous.get('completed_evidence_sha256', {})
    for path, digest in bindings.items():
        if _evidence_sha256(path, force=force_evidence) != digest:
            raise RuntimeError('Completed episode evidence changed: ' + path)
    rows, complete, blocked, pending, running = [], [], [], [], []
    for case in manifest['cases']:
        state = case_state(output, case, manifest_digest=manifest['manifest_digest'])
        directory = output / 'results' / case['id']
        if state == 'complete':
            row = read_json(directory / 'result.json')
            if row.get('checkpoint_identity') != manifest['checkpoint_identity'] or row.get('harness') != manifest['harness']:
                raise RuntimeError('Episode belongs to a different policy/harness: ' + case['id'])
            complete.append(row)
            for path in directory.iterdir():
                if path.is_file() and path.name not in ('progress.json', 'client.log'):
                    name = str(path)
                    if name not in bindings:
                        bindings[name] = _evidence_sha256(path, force=force_evidence)
        elif state == 'blocked':
            blocked.append(case['id'])
            row = read_json(directory / 'result.json') if (directory / 'result.json').exists() else {}
        elif state == 'pending':
            pending.append(case['id'])
            row = {}
        else:
            running.append(case['id'])
            row = read_json(directory / 'progress.json') if (directory / 'progress.json').exists() else {}
        rows.append({'task': case['task'], 'trial': case['trial'], 'group': case['group'],
            'status': state, 'success': row.get('success') if state == 'complete' else None,
            'steps': row.get('steps'), 'horizon': case['horizon'],
            'policy_calls': row.get('policy_calls'), 'policy_seconds': row.get('policy_seconds'),
            'wall_seconds': row.get('wall_seconds'), 'result': str(directory / 'result.json')})
    all_complete = len(complete) == len(manifest['cases'])
    wins = sum(row['success'] for row in complete)
    by_task = {}
    for case in manifest['cases']:
        by_task.setdefault(case['task'], {'group': case['group'], 'expected': 0,
                                         'completed': 0, 'successes': 0})['expected'] += 1
    for row in complete:
        by_task[row['task']]['completed'] += 1
        by_task[row['task']]['successes'] += row['success']
    macro = (sum(row['successes'] / row['completed'] for row in by_task.values()) / len(by_task)
             if all_complete else None)
    result = {'updated_at': now(), 'manifest_digest': manifest['manifest_digest'],
        'status': 'complete' if all_complete else 'incomplete',
        'policy': manifest['policy'], 'benchmark': manifest['benchmark'],
        'harness': manifest['harness'], 'mode': manifest['mode'],
        'groups': manifest['groups'], 'expected_episodes': len(manifest['cases']),
        'completed_episodes': len(complete), 'successes': wins,
        'physical_steps_complete_episodes': sum(row['steps'] for row in complete),
        'descriptive_success_rate': wins / len(complete) if complete else None,
        'macro_task_success_rate': macro,
        'selected_task_coverage_complete': all_complete and manifest['mode'] == 'full',
        'tasks_with_success': sum(row['successes'] > 0 for row in by_task.values()),
        'expected_tasks': len(by_task),
        'official_group_protocol_complete': (all_complete and manifest['mode'] == 'full'
            and manifest.get('official_sequential_reset_equivalent', True)
            and all(row['expected'] == manifest.get('official_trials_per_task', 50)
                    for row in by_task.values())),
        'pending_cases': pending, 'blocked_cases': blocked, 'running_cases': running, 'per_task': by_task,
        'completed_evidence_sha256': bindings,
        'comparison': manifest.get('comparison',
            'standalone, independent unseeded scenes; no matched native/null parity or improvement claim'),
        'note': 'Errors/interrupted episodes are excluded from robot success/failure denominators and remain visible.'}
    write_json(previous_path, result)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = output / '.episodes.csv.tmp'
    temporary.write_text(stream.getvalue())
    os.replace(temporary, output / 'episodes.csv')
    return result


def stop_owned_group(record, grace=5):
    """Session identity, boot and start-time guards prevent unrelated kills."""
    members = group_members(record['pgid'])
    leader = process_identity(record['pid'])
    if members and leader is not None and leader != record:
        raise RuntimeError('Process-group leader PID was reused; reservation retained')
    if any(row['boot_id'] != record['boot_id'] or
           int(row['start_ticks']) < int(record['start_ticks']) for row in members):
        raise RuntimeError('Ambiguous process group; reservation retained')
    for sig, duration in ((signal.SIGTERM, grace), (signal.SIGKILL, 2)):
        if not group_members(record['pgid']):
            return
        try:
            os.killpg(record['pgid'], sig)
        except ProcessLookupError:
            return
        until = time.monotonic() + duration
        while group_members(record['pgid']) and time.monotonic() < until:
            time.sleep(.1)
    if group_members(record['pgid']):
        raise RuntimeError('Owned descendants still alive; reservation retained')


def topology_and_idle(ids, bench):
    """Both typed compute/graphics occupancy and PCI -> EGL mapping are checked."""
    from . import resources
    inventory = resources.snapshot()
    xml = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True,
                         check=True, timeout=20).stdout
    selected_uuids = {row['uuid'] for row in inventory if row['index'] in ids}
    for gpu in ET.fromstring(xml).findall('gpu'):
        if gpu.findtext('uuid') in selected_uuids and gpu.findall('./processes/process_info'):
            raise RuntimeError('Selected GPU is occupied; choose an idle GPU')
    code = 'import json;from gpu_mapping import discover;print(json.dumps(discover()))'
    observed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=child_environment(bench), check=True, timeout=30)
    import json
    mapping = json.loads(observed.stdout)
    pinned = read_json(Path(bench) / 'registry/gpu-mapping.json')['devices']
    if mapping != pinned:
        raise RuntimeError('GPU/EGL mapping changed; refresh and validate RobotPolicyBench registry')
    by_id = {row['index']: row for row in inventory}
    for gpu in ids:
        if gpu not in by_id or str(gpu) not in mapping or mapping[str(gpu)]['uuid'] != by_id[gpu]['uuid']:
            raise ValueError('Unknown or mismatched physical GPU: ' + str(gpu))
    return inventory, mapping


def _wait(process, server, deadline, stopped):
    while process.poll() is None:
        if stopped() or time.time() >= deadline - 15:
            raise TimeoutError('Stop/deadline: preserving incomplete episode without reroll')
        if server is not None and server.poll() is not None:
            raise RuntimeError('Policy server exited; see server.log')
        time.sleep(.25)


