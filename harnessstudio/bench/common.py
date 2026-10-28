"""Small shared helpers, also importable in policy Python 3.10 environments."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def stat_fingerprint(path):
    info = Path(path).stat()
    return {'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'ctime_ns': info.st_ctime_ns, 'ino': info.st_ino, 'dev': info.st_dev}


def verify_manifest_digest(manifest):
    expected = manifest.get('manifest_digest')
    if not isinstance(expected, str) or expected != json_hash({
            key: value for key, value in manifest.items() if key != 'manifest_digest'}):
        raise RuntimeError('Frozen manifest digest changed or missing')
    return manifest


def verify_manifest(manifest):
    if not isinstance(manifest, dict):
        manifest = read_json(manifest)
    verify_manifest_digest(manifest)
    for path, expected in manifest['source_sha256'].items():
        if sha256(path) != expected:
            raise RuntimeError('Frozen source changed: ' + path)
    for path, expected in manifest['checkpoint_files'].items():
        if stat_fingerprint(path) != expected['stat']:
            raise RuntimeError('Frozen checkpoint metadata changed: ' + path)
    return manifest


def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': fields[19], 'pgid': int(fields[2]),
                'sid': int(fields[3]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def group_members(pgid):
    members = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            item = process_identity(int(entry.name))
            if item and item['pgid'] == pgid and item['sid'] == pgid:
                members.append(item)
    return members


def valid_result(row, case):
    return (row.get('status') == 'complete' and row.get('task') == case['task']
            and row.get('trial') == case['trial'] and row.get('horizon') == case['horizon']
            and row.get('quality_result') is True and type(row.get('success')) is bool
            and type(row.get('steps')) is int and 0 < row['steps'] <= case['horizon']
            and (row['success'] or row['steps'] == case['horizon']))


def case_state(output, case, *, manifest_digest=None):
    """Errors and interrupted unseeded scenes are never silently rerolled."""
    directory = Path(output) / 'results' / case['id']
    result = directory / 'result.json'
    if result.exists():
        row = read_json(result)
        seal = directory / 'completed.json'
        if valid_result(row, case) and seal.exists():
            committed = read_json(seal)
            if (committed.get('result_sha256') == sha256(result) and
                    (manifest_digest is None or committed.get('manifest_digest') == manifest_digest)):
                return 'complete'
        if not valid_result(row, case):
            return 'blocked'
    started = directory / 'started.json'
    if started.exists():
        owner = read_json(started).get('worker_identity')
        if owner and process_identity(owner['pid']) == owner:
            return 'running'
    if (directory / 'started.json').exists() or directory.exists():
        return 'blocked'
    return 'pending'
