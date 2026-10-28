"""P2 parity controls (evaluator-only): scene witness, canonical table, pre-action check.

Nothing here enters the policy observation, PublicObservation, selector, harness or
memory. Values are written only to evaluator records/files.

P2.1  initial_observation_sha256 stays the strict aggregate over all 9 public keys.
P2.2  initial_scene_sha256: sha256 over canonical bytes of MuJoCo state right after
      env.reset(seed) (qpos, qvel, act, mocap_pos, mocap_quat, time), a digest of compiled-model
      arrays (sizes, body/geom/joint/mesh/site) and the RoboCasa ep_meta (layout/style/objects/placements).
P2.4  pre-action check against a canonical table from a policy-free reset probe:
      class S (scene digest or any non-image key differs) -> abort before any action;
      class R (only video.* keys differ) -> re-render from the unchanged sim state
      (no physics step, no reset) at most ``max_rerenders`` times; exact match proceeds,
      otherwise abort before any action.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json

TABLE_SCHEMA = 'chunktrust-canonical-table/1'
MODEL_SIZES = ('nq', 'nv', 'nu', 'nbody', 'ngeom', 'nmesh', 'njnt', 'nsite')
MODEL_ARRAYS = ('body_pos', 'body_quat', 'body_mass', 'geom_type', 'geom_size', 'geom_pos', 'geom_quat',
                'geom_dataid', 'geom_matid', 'jnt_type', 'qpos0', 'mesh_vertnum', 'site_pos')
# geom_rgba excluded: measured process-nondeterministic (2026-10-26 self-check) while renders and
# every other compiled array matched; it is cosmetic and cannot witness layout or placement.


class ParityAbort(RuntimeError):
    def __init__(self, report):
        super().__init__(f"pre-action parity abort: class {report.get('class')}")
        self.report = report


def _canonical(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return {'__ndarray__': value.dtype.str, 'shape': list(value.shape),
                'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, float) and value != value:
        return 'NaN'
    return value if isinstance(value, (str, int, float, bool)) or value is None else repr(value)


def _array_digest(array):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(array))
    h = hashlib.sha256(json.dumps([a.dtype.str, list(a.shape)]).encode() + b'\0')
    h.update(a.tobytes())
    return h.hexdigest()


def robosuite_env(env):
    """MultiStepWrapper -> gym env (unwrapped) -> robosuite Kitchen env."""
    base = env.unwrapped
    inner = getattr(base, 'env', None)
    if inner is None or not hasattr(inner, 'sim'):
        raise ValueError('cannot reach the robosuite simulator for the scene witness')
    return base, inner


def scene_witness(env):
    """Read-only digest of the simulator state. Never steps, resets or renders."""
    _, inner = robosuite_env(env)
    data = inner.sim.data
    components = {}
    for name in ('qpos', 'qvel', 'act', 'mocap_pos', 'mocap_quat'):
        value = getattr(data, name, None)
        components[name] = None if value is None else _array_digest(value)
    components['time'] = float(data.time)
    # Compiled-model digest (the saved XML text embeds process-specific paths, measured
    # 2026-10-26 self-check: equal scenes, different XML text across processes).
    model = getattr(inner.sim.model, '_model', inner.sim.model)
    compiled = {}
    for name in MODEL_SIZES:
        value = getattr(model, name, None)
        compiled[name] = None if value is None else int(value)
    for name in MODEL_ARRAYS:
        value = getattr(model, name, None)
        compiled[name] = None if value is None else _array_digest(value)
    components['compiled_model_parts'] = compiled
    components['compiled_model_sha256'] = hashlib.sha256(
        json.dumps(compiled, sort_keys=True).encode()).hexdigest()
    meta = inner.get_ep_meta() if hasattr(inner, 'get_ep_meta') else None
    components['ep_meta_sha256'] = hashlib.sha256(
        json.dumps(_canonical(meta), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    digest = hashlib.sha256(json.dumps(components, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'initial_scene_sha256': digest, 'components': components}


def rerender_public(env):
    """Recompute observations from the unchanged sim state, mirroring MultiStepWrapper.reset."""
    base, inner = robosuite_env(env)
    raw = inner._get_observations(force_update=True)
    obs = base.get_observation(raw)
    env.obs = deque([obs] * (env.max_steps_needed + 1), maxlen=env.max_steps_needed + 1)
    return env._get_obs(env.video_delta_indices, env.state_delta_indices)


def load_table(path_or_value):
    table = path_or_value
    if not isinstance(table, dict):
        table = json.loads(open(path_or_value).read())
    if table.get('schema') != TABLE_SCHEMA or not isinstance(table.get('cases'), dict):
        raise ValueError('canonical table schema mismatch')
    for case_id, row in table['cases'].items():
        if not isinstance(row.get('keys'), dict) or not isinstance(row.get('scene_sha256'), str):
            raise ValueError('canonical table row malformed: ' + case_id)
    return table


def compare(entry, key_hashes, scene_sha256):
    keys = {k: v for k, v in key_hashes.items() if k != '__aggregate__'}
    if set(keys) != set(entry['keys']):
        return 'S', sorted(set(keys) ^ set(entry['keys']))
    differing = sorted(k for k in keys if keys[k] != entry['keys'][k])
    if scene_sha256 != entry['scene_sha256'] or any(not k.startswith('video.') for k in differing):
        return 'S', differing + (['__scene__'] if scene_sha256 != entry['scene_sha256'] else [])
    return ('R', differing) if differing else ('match', [])


def preaction_check(env, public, case_id, table, *, hash_fn, max_rerenders=2):
    """Return (public_to_use, report) or raise ParityAbort. Called before any action."""
    if table is None:
        return public, {'checked': False, 'reason': 'canonical table explicitly disabled'}
    entry = table['cases'].get(case_id)
    if entry is None or entry.get('ambiguous'):
        raise ParityAbort({'checked': True, 'class': 'S', 'reason': 'no unambiguous canonical entry',
                           'case_id': case_id, 'attempts': []})
    scene = scene_witness(env)['initial_scene_sha256']
    attempts = []
    for attempt in range(max_rerenders + 1):
        hashes = hash_fn(public)
        cls, differing = compare(entry, hashes, scene)
        attempts.append({'attempt': attempt, 'class': cls, 'differing': differing})
        if cls == 'match':
            return public, {'checked': True, 'class': 'match' if attempt == 0 else 'R_resolved',
                            'rerenders': attempt, 'attempts': attempts}
        if cls == 'S' or attempt == max_rerenders:
            raise ParityAbort({'checked': True, 'class': cls, 'rerenders': attempt, 'attempts': attempts,
                               'case_id': case_id})
        public = rerender_public(env)
        after = scene_witness(env)['initial_scene_sha256']
        if after != scene:
            raise ParityAbort({'checked': True, 'class': 'S', 'reason': 're-render changed the scene digest',
                               'rerenders': attempt + 1, 'attempts': attempts, 'case_id': case_id})
    raise AssertionError('unreachable')


def save_packet(path, public):
    """P2.3: evaluator-only initial public packet (3 RGB, 5 state, task text)."""
    import numpy as np
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in public.items()})
