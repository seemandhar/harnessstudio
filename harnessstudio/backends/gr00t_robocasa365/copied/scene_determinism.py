"""Runner v2.1: make seeded RoboCasa scene construction a function of the seed alone.

Repair `repair-reset-determinism-1` (2026-10-26). Diagnosed mechanisms (CPU reproducer
`diagnostics/scene_order_repro.py`, report `reports/repair_reset_determinism_1.md`):

M1 (physical scene).  robocasa `models/fixtures/counter.py` Counter.get_reset_regions
    de-duplicates candidate counter-top geoms with ``valid_geoms = list(set(valid_geoms))``.
    The elements are xml.etree Elements, hashed by object identity (memory address), so the
    region list order depends on the process's allocation layout. ``self.rng.choice`` then maps
    the SAME seeded draw onto a DIFFERENT region -> different object/fixture placement and
    robot base. Neither PYTHONHASHSEED nor seeding np.random/random can affect it.
    Repair: inside that module only, the bare name ``set`` (exactly one Load, verified by AST
    against a pinned file hash) resolves to an order-preserving first-occurrence de-duplication.
    Membership is identical; the order is the order in which the original code enumerates the
    geoms, which is one of the orders the unpatched code already produces in some processes.

M2 (evaluator witness metadata only).  OBJ_GROUPS type-group keys are inserted by iterating a
    set of strings, so ``groups_containing_sampled_obj`` in ep_meta is ordered by the per-process
    string hash seed. Physics and public inputs are unaffected (measured), but the evaluator
    witness hashes ep_meta. Repair: the simulator CLIENT process (never the policy server) runs
    with PYTHONHASHSEED=0, and make_env refuses to build a scene otherwise.

Nothing here touches the policy, its inputs, actions, RNG, weights, horizons, success or reset
seed. np.random/random global state is deliberately NOT reseeded: it only colours invisible
group-0 counter geoms (measured) and seeding it did not remove M1 (negative control).
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import sys

HASH_SEED = '0'
# robocasa @ 4f8a2980def75a55dff96b990745b83540425f09, models/fixtures/counter.py
PINNED_COUNTER_SHA256 = '77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a'
PINNED_SET_LINES = (813,)
PATCH_NAME = 'counter.get_reset_regions:set->first_occurrence_unique'


def first_occurrence_unique(iterable=()):
    """Same members as set(iterable) (identity/equality semantics of dict keys), stable order."""
    return list(dict.fromkeys(iterable))


def bare_set_loads(source):
    return sorted(n.lineno for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.Name) and n.id == 'set' and isinstance(n.ctx, ast.Load))


def client_environment(env):
    """Environment for a simulator-client process (episode client or reset/lockstep probe)."""
    env = dict(env)
    env['PYTHONHASHSEED'] = HASH_SEED
    return env


def require_hash_seed():
    if os.environ.get('PYTHONHASHSEED') != HASH_SEED or sys.flags.hash_randomization:
        raise RuntimeError('v2.1 simulator client must run with PYTHONHASHSEED=%s (scene determinism M2)' % HASH_SEED)


def install(*, expected_sha256=PINNED_COUNTER_SHA256, require_hash=True):
    """Apply the M1 repair (idempotent) and check M2; return an evaluator-only receipt.

    Fails closed if the simulator source differs from the audited file, so a robocasa change can
    never be silently 'repaired' by a stale patch.
    """
    if require_hash:
        require_hash_seed()
    from robocasa.models.fixtures import counter
    path = Path(counter.__file__).resolve()
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f'robocasa counter.py changed ({digest}); scene-determinism patch not audited for it')
    lines = tuple(bare_set_loads(source.decode()))
    if lines != PINNED_SET_LINES:
        raise RuntimeError(f'unexpected bare set() uses in counter.py: {lines}')
    current = counter.__dict__.get('set')
    if current is not None and current is not first_occurrence_unique:
        raise RuntimeError('counter.set already shadowed by something else')
    counter.set = first_occurrence_unique
    return {'patch': PATCH_NAME, 'counter_py': str(path), 'counter_sha256': digest,
            'set_lines': list(lines), 'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
            'hash_randomization': int(sys.flags.hash_randomization)}
