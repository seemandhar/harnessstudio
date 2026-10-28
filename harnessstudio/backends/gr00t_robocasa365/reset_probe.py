"""P2.6(i) policy-free reset probe and P2.4 canonical-table builder (evaluator-owned, GPU/EGL), runner v2.2.

No policy is loaded, no action is executed, no outcome exists. Each reset runs in its own
fresh process on its planned (GPU, slot); a case's repeats rotate over GPU-major slots so
they land on different GPUs. Output: canonical_table.json + probe_report.json (+ per-reset
JSON). Controller (any Python with the project on sys.path):

  python probes/reset_probe.py --cases-file runs/.../subsets_frozen.json --phase development \
      --include-calibration --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 1 --repeats 3 --out DIR

HarnessStudio copy, adapted from ChunkTrust runner v2.2 probes/reset_probe.py (original sha256 prefix 6fb3de75a529);
imports repointed to harnessstudio.bench and this backend's worker/runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

PROBES = Path(__file__).resolve().parent  # this backend directory
SOURCE = PROBES
STUDIO_SRC = PROBES.parents[2]
for _p in (str(STUDIO_SRC), str(SOURCE), str(SOURCE / 'copied')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CALIBRATION_CASES = [{'id': 'CloseToasterOvenDoor--0001', 'task': 'CloseToasterOvenDoor', 'trial': 1,
                      'task_index': 2, 'horizon': 450, 'seed': 108}]


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided binomial CI by bisection on the binomial CDF."""
    if n == 0:
        return [None, None]

    def cdf(x, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))

    def root(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else root(lambda p: 1 - cdf(k - 1, p), alpha / 2, True)
    upper = 1.0 if k == n else root(lambda p: cdf(k, p), alpha / 2, False)
    return [lower, upper]


def build_table(rows, cases):
    """rows: per-reset dicts {case_id, keys{9}, scene_sha256, gpu, slot}. Modal values, divergence stats."""
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case_id'], []).append(row)
    table, renders, deviant, pairs, divergent_pairs = {}, 0, 0, 0, 0
    per_key = Counter()
    for case in cases:
        got = by_case.get(case['id'], [])
        if not got:
            table[case['id']] = {'seed': case['seed'], 'ambiguous': True, 'n_resets': 0, 'keys': {}, 'scene_sha256': ''}
            continue
        keys = sorted(got[0]['keys'])
        entry = {'seed': case['seed'], 'n_resets': len(got), 'keys': {}, 'ambiguous': False, 'divergent': []}
        for key in keys + ['__scene__']:
            values = [r['scene_sha256'] if key == '__scene__' else r['keys'][key] for r in got]
            counts = Counter(values)
            value, count = counts.most_common(1)[0]
            if count * 2 <= len(values):
                entry['ambiguous'] = True
            if len(counts) > 1:
                entry['divergent'].append(key)
                divergent_pairs += 1
                per_key[key] += 1
            pairs += 1
            renders += len(values)
            deviant += len(values) - count
            if key == '__scene__':
                entry['scene_sha256'] = value
            else:
                entry['keys'][key] = value
        entry['initial_observation_sha256'] = None if entry['ambiguous'] else _aggregate(entry['keys'])
        table[case['id']] = entry
    stats = {'case_key_pairs': pairs, 'divergent_case_key_pairs': divergent_pairs,
             'divergent_pair_rate_ci95': clopper_pearson(divergent_pairs, pairs),
             'renders': renders, 'deviant_renders': deviant,
             'deviant_render_rate_ci95': clopper_pearson(deviant, renders), 'divergent_by_key': dict(per_key),
             'ambiguous_cases': sorted(c for c, e in table.items() if e.get('ambiguous'))}
    return {'schema': 'chunktrust-canonical-table/1', 'cases': table, 'stats': stats}


def _aggregate(keys):
    import hashlib
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()


def child(args):
    from harnessstudio.bench import groot_native as gw
    import worker as ahs_worker
    import parity
    started = time.monotonic()
    determinism = {}
    env = ahs_worker.make_env(args.task, args.horizon, 'pretrain', receipt=determinism)
    try:
        initial, _ = env.reset(seed=args.seed)
        public = gw.public_copy(initial)
        hashes = gw.hashes(public)
        witness = parity.scene_witness(env)
        if args.packet:
            parity.save_packet(Path(args.packet), public)
        rerender = None
        if args.rerender_check:
            again = gw.hashes(gw.public_copy(parity.rerender_public(env)))
            rerender = {'keys_equal': {k: again[k] == hashes[k] for k in hashes if k != '__aggregate__'},
                        'scene_unchanged': parity.scene_witness(env)['initial_scene_sha256'] == witness['initial_scene_sha256']}
    finally:
        env.close()
    row = {'case_id': args.case_id, 'seed': args.seed, 'keys': {k: v for k, v in hashes.items() if k != '__aggregate__'},
           'initial_observation_sha256': hashes['__aggregate__'], 'scene_sha256': witness['initial_scene_sha256'],
           'scene_components': witness['components'], 'gpu': args.gpu, 'slot': args.slot,
           'repeat': args.repeat, 'rerender_check': rerender, 'scene_determinism': determinism, 'render_context_events': list(__import__('render_context_fix').EVENTS), 'pid': os.getpid(), 'wall_seconds': time.monotonic() - started, 'policy_loaded': False, 'actions': 0}
    Path(args.out).write_text(json.dumps(row, indent=1, sort_keys=True))
    return 0


def controller(args):
    from harnessstudio import config
    from harnessstudio.bench import resources
    from harnessstudio.bench.registry import DEFAULT_POLICYBENCH, child_environment
    from harnessstudio.bench.runner_util import topology_and_idle
    import runner as ahs_runner
    if args.cases_json:
        cases = json.loads(Path(args.cases_json).read_text())
    else:
        spec = json.loads(Path(args.cases_file).read_text())
        cases = [c for c in spec[args.phase]['cases']] + (CALIBRATION_CASES if args.include_calibration else [])
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ids = [int(x) for x in args.gpus.split(',')]
    slots = ahs_runner.plan_slots(ids, args.workers_per_gpu, 10 ** 6)
    jobs = {key: [] for key in slots}
    for i, case in enumerate(cases):
        for r in range(args.repeats):
            jobs[slots[(i + r) % len(slots)]].append((case, r))
    bench = DEFAULT_POLICYBENCH
    inventory, mapping = topology_and_idle(ids, bench)
    owner = 'chunktrust-reset-probe.' + uuid.uuid4().hex
    lease_dir = config.leases_dir()
    resources.acquire(lease_dir, owner, ids, inventory=inventory, instances_per_gpu=args.workers_per_gpu,
                      calibrated_instance_mib=args.calibrated_worker_mib)
    python = str(bench / 'envs/gr00t-n15/bin/python')
    rows, failures, lock = [], [], threading.Lock()
    started = time.time()

    def run_slot(key):
        gpu, slot = (int(v) for v in key.split('.'))
        import scene_determinism
        env = scene_determinism.client_environment(child_environment(bench, egl=mapping[str(gpu)]['egl_index']))
        for case, r in jobs[key]:
            target = out / 'resets' / f"{case['id']}--r{r}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            argv = [python, str(PROBES / 'reset_probe.py'), '--child', '--case-id', case['id'], '--task', case['task'],
                    '--seed', str(case['seed']), '--horizon', str(case['horizon']), '--gpu', str(gpu),
                    '--slot', str(slot), '--repeat', str(r), '--out', str(target)]
            if args.save_packets:
                argv += ['--packet', str(target.with_suffix('.npz'))]
            if args.rerender_check:
                argv += ['--rerender-check']
            with target.with_suffix('.log').open('w') as log:
                code = subprocess.run(argv, env=env, cwd=config.workspace(), stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
            with lock:
                if code or not target.exists():
                    failures.append({'case_id': case['id'], 'repeat': r, 'gpu': gpu, 'slot': slot, 'returncode': code})
                else:
                    rows.append(json.loads(target.read_text()))
    try:
        threads = [threading.Thread(target=run_slot, args=(key,)) for key in slots if jobs[key]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        released = resources.release(lease_dir, owner)
    table = build_table(rows, cases)
    table.update(probe={'phase': args.phase, 'gpus': ids, 'workers_per_gpu': args.workers_per_gpu,
                        'repeats': args.repeats, 'failures': failures, 'policy_free': True,
                        'wall_seconds': time.time() - started, 'lease_released': bool(released)})
    (out / 'canonical_table.json').write_text(json.dumps(table, indent=1, sort_keys=True))
    print(json.dumps({'cases': len(cases), 'resets': len(rows), 'failures': len(failures), **table['stats']}, indent=1))
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--case-id'); ap.add_argument('--task'); ap.add_argument('--seed', type=int)
    ap.add_argument('--horizon', type=int); ap.add_argument('--gpu', type=int); ap.add_argument('--slot', type=int)
    ap.add_argument('--repeat', type=int); ap.add_argument('--out'); ap.add_argument('--packet')
    ap.add_argument('--cases-file'); ap.add_argument('--phase', choices=('development', 'validation'))
    ap.add_argument('--include-calibration', action='store_true')
    ap.add_argument('--gpus'); ap.add_argument('--workers-per-gpu', type=int, default=1)
    ap.add_argument('--calibrated-worker-mib', type=int)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--save-packets', action='store_true')
    ap.add_argument('--rerender-check', action='store_true', help='diagnostic: one re-render from the unchanged state')
    ap.add_argument('--cases-json', help='explicit JSON list of cases (non-subset self-checks) instead of --cases-file')
    args = ap.parse_args(argv)
    return child(args) if args.child else controller(args)


if __name__ == '__main__':
    raise SystemExit(main())
