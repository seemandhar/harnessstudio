"""Verified policy/benchmark pairs and outcome-free case selection.

Vendored from RoboMetaHarness robometaharness/benchmarking/registry.py (bench/PROVENANCE.json).
Paths come from harnessstudio.config; child processes import harnessstudio, not robometaharness.
"""
import os
from pathlib import Path
import subprocess

from .. import config
from .common import read_json, sha256, stat_fingerprint, write_json

DEFAULT_POLICYBENCH = config.policybench()
GROUPS = ('atomic_seen', 'composite_seen', 'composite_unseen')


def canonical_pair(policy, benchmark):
    policy = policy.lower().replace('_', '-').replace(' ', '')
    benchmark = benchmark.lower().replace('_', '').replace('-', '')
    if policy in ('groot', 'gr00t', 'gr00t-n15', 'gr00t-n1.5', 'groot-n1.5'):
        policy = 'gr00t-n1.5'
    if policy in ('awomo-0.5v', 'awomo-05v', 'awomo0.5v'):
        policy = 'awomo-0.5v'
    if policy in ('pi0.5', 'pi05', 'pi-0.5', 'π0.5'):
        policy = 'pi0.5'
    if benchmark in ('robocasa', 'robocasa365', 'robocase'):
        benchmark = 'robocasa365'
    if benchmark != 'robocasa365' or policy not in ('gr00t-n1.5', 'awomo-0.5v', 'pi0.5'):
        raise ValueError('Unsupported pair. Use --list for implemented backends; policies are never substituted.')
    return policy, benchmark


def supported():
    return [{'policy': 'gr00t-n1.5', 'benchmark': 'robocasa365',
             'checkpoint': 'Human300/checkpoint-120000',
             'harnesses': ['unified', 'native', '/path/to/harness.py:factory'],
             'modes': ['small', 'full'], 'groups': list(GROUPS), 'official_trials_per_task': 50},
            {'policy': 'awomo-0.5v', 'benchmark': 'robocasa365',
             'checkpoint': 'Auwomo/Awomo-0.5V-RoboCasa365',
             'harnesses': ['unified', 'native', '/path/to/harness.py:factory'], 'modes': ['small', 'full'],
             'groups': list(GROUPS), 'official_trials_per_task': 30},
            {'policy': 'pi0.5', 'benchmark': 'robocasa365',
             'checkpoint': 'robocasa/pi05_pretrain_human300/multitask_learning/75000',
             'harnesses': ['unified', 'native', '/path/to/harness.py:factory'],
             'modes': ['small', 'full'], 'groups': list(GROUPS),
             'official_trials_per_task': 50,
             'protocol_note': 'isolated later-trial resets differ from published sequential env resets'}]


def child_environment(policybench, *, project=None, gpu=None, egl=None):
    bench = Path(policybench)
    env = dict(os.environ)
    # Saved keys may be explicitly loaded by a reviewed harness plugin; never
    # copy the outer agent's credentials into the baseline policy server.
    for key in list(env):
        if key.endswith('_API_KEY') or key in ('NRP_CACHE_SALT',):
            env.pop(key, None)
    env.update(PYTHONPATH=os.pathsep.join(map(str, [config.SOURCE_ROOT,
        bench / 'policies/isaac-gr00t', bench / 'benchmarks/robocasa365',
        bench / 'benchmarks/robosuite', bench / 'scripts'])),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', NO_ALBUMENTATIONS_UPDATE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
    if egl is None:
        env.pop('MUJOCO_EGL_DEVICE_ID', None)
    else:
        env['MUJOCO_EGL_DEVICE_ID'] = str(egl)
    return env


def load_task_registry(policybench):
    import json
    bench = Path(policybench)
    interpreter = bench / 'envs/gr00t-n15/bin/python'
    if not interpreter.is_file():
        raise ValueError('Missing isolated policy environment: ' + str(interpreter))
    result = subprocess.run([str(interpreter), '-m',
        'harnessstudio.bench.groot_native', '--mode', 'registry'],
        env=child_environment(bench), capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError('Cannot inspect official task registry: ' + result.stderr[-3000:])
    rows = [line.split(':', 1)[1] for line in result.stdout.splitlines()
            if line.startswith('RMH_REGISTRY:')]
    if len(rows) != 1:
        raise RuntimeError('Backend returned no unique official task registry')
    return json.loads(rows[0])


def csv_words(values):
    return [word for value in (values or []) for word in value.split(',') if word]


def normalize_groups(groups):
    """Resolve readable CLI aliases before freezing cases or comparing resumes."""
    if not groups:
        return []
    names = [name.lower().replace('-', '_') for name in groups]
    if 'all' in names:
        if names != ['all']:
            raise ValueError("Use 'all' alone, or list individual task groups")
        return list(GROUPS)
    aliases = {'atomic_seen': 'atomic_seen', 'atomicseen': 'atomic_seen',
               'comseen': 'composite_seen', 'composite_seen': 'composite_seen',
               'comunseen': 'composite_unseen', 'composite_unseen': 'composite_unseen'}
    if any(name not in aliases for name in names):
        raise ValueError('Unknown task group; use atomic_seen, comseen, comunseen or all')
    resolved = [aliases[name] for name in names]
    if len(resolved) != len(set(resolved)):
        raise ValueError('Duplicate task group after resolving aliases')
    return resolved


def choose_cases(registry, *, mode, groups=None, tasks=None, num_tasks=None,
                 episodes_per_task=None, default_trials=50, seeded=False, indexed=False):
    groups = normalize_groups(groups) or (list(GROUPS) if mode == 'full' else ['atomic_seen'])
    if len(groups) != len(set(groups)) or any(g not in registry for g in groups):
        raise ValueError('Unknown or duplicate task group')
    available = {}
    for group in groups:
        for index, row in enumerate(registry[group]):
            name = row.get('name', row.get('task'))
            if name in available:
                raise ValueError('Registry has overlapping selected task groups: ' + name)
            available[name] = {'task': name, 'horizon': row['horizon'], 'group': group}
            if seeded or indexed:
                available[name]['task_index'] = index
    if mode == 'full':
        if tasks or num_tasks is not None:
            raise ValueError('Full mode evaluates every task in selected groups; use small for a subset')
        selected = list(available)
        count = default_trials if episodes_per_task is None else episodes_per_task
        if count < 1:
            raise ValueError('episodes-per-task must be positive')
    else:
        count = 1 if episodes_per_task is None else episodes_per_task
        if count < 1:
            raise ValueError('episodes-per-task must be positive')
        if tasks:
            if num_tasks is not None:
                raise ValueError('Use either --tasks or --num-tasks')
            if len(tasks) != len(set(tasks)) or any(t not in available for t in tasks):
                raise ValueError('Unknown or duplicate task; check selected --groups')
            # Official order is stable regardless of argument ordering.
            selected = [name for name in available if name in tasks]
        else:
            size = 4 if num_tasks is None else num_tasks
            if size < 1:
                raise ValueError('num-tasks must be positive')
            names = list(available)
            size = min(size, len(names))
            indices = [0] if size == 1 else [i * (len(names) - 1) // (size - 1) for i in range(size)]
            selected = [names[i] for i in indices]
        if len(selected) * count > 32:
            raise ValueError('Small mode is capped at 32 episodes; use full for official coverage')
    if not selected:
        raise ValueError('Empty benchmark selection')
    return groups, [dict(available[name], trial=trial, id=f'{name}--{trial:04d}',
                        **({'seed': 7 + 50 * available[name]['task_index'] + trial} if seeded else {}))
                    for name in selected for trial in range(count)]


def checkpoint_inventory(bench, *, rehash=False):
    """Hash once, cache by inode/size/nanosecond mtime+ctime, check release hashes."""
    import fcntl
    bench = Path(bench)
    model = bench / 'models/gr00t-n1.5-robocasa-human300'
    cachefile = config.workspace() / '.state/benchmark-checkpoints.json'
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    with cachefile.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = read_json(cachefile) if cachefile.exists() else {}
        files = sorted(p for p in model.rglob('*') if p.is_file() and '.cache' not in p.parts)
        if not files:
            raise ValueError('Missing checkpoint: ' + str(model))
        for path in files:
            stamp = stat_fingerprint(path)
            old = cached.get(str(path))
            digest = old['sha256'] if not rehash and old and old['stat'] == stamp else sha256(path)
            if stat_fingerprint(path) != stamp:
                raise RuntimeError('Checkpoint changed while hashing: ' + str(path))
            result[str(path)] = {'sha256': digest, 'stat': stamp}
        official = read_json(bench / 'registry/gr00t-n15-checkpoint-verification.json')
        for name, row in official['verified_files'].items():
            relative = name.split('gr00t-n1.5-robocasa-human300/')[-1]
            path = str(model / relative)
            if path not in result or result[path]['sha256'] != row['official_sha256']:
                raise RuntimeError('Checkpoint differs from official pinned release: ' + relative)
        cached.update(result)
        write_json(cachefile, cached)
    return str(model), result, official['revision']
