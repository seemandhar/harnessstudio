"""Execution-horizon layer for the ChunkTrust-AHS Unified reference (GR00T N1.5).

Pure NumPy + stdlib; importable in the Python 3.10 GR00T env and the 3.13 controller.

Layers (outermost last):
  UnifiedHarness (identity selection, one native proposal, native action object)
    -> HorizonSelector.select(velocity, actions)  # how many prefix actions to execute
    -> VariableHorizonUnifiedHarness.commit(executed)  # Unified cadence follows actual k

The upstream numerical selector is vendored byte-for-byte in chunktrust_gr00t_ahs.py
(ChunkTrust commit a9fd5741, src/chunktrust/gr00t_ahs.py). This file only holds
episode state (Beta posterior, RNG, executed history) exactly as the upstream
GR00T N1.5 overlay EvalTraceManager does for one env (env_idx = 0).

EXTENSION SEAM: a future candidate supplies a ``selector_plugin`` ("file.py:factory")
in its own harness config. ``factory(base, config)`` receives the frozen reference
selector built from the immutable reference parameters and returns any object
implementing HorizonSelector. See EXTENSION.md.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from chunktrust_gr00t_ahs import select_exec_k_horizon

NATIVE_CHUNK = 16          # GR00T N1.5 action_horizon (config.json action_head_cfg)
MODEL_ACTION_DIM = 32      # padded model action space of the velocity trace
EFFECTIVE_ACTION_DIM = 12  # panda_omron: 4+1+3+3+1 native action fields
DENOISING_STEPS = 4        # evaluate.py GR00T server uses denoising_steps=4

# Paper/overlay-derived reference parameters. Fixed BEFORE any outcome; see
# reports/conversion_report.md section "Hyperparameter mapping".
REFERENCE_AHS_PARAMS = {
    'horizon_candidates': [4, 8, 12, 16],   # paper A.4 "RoboCasa GR1, all policies"
    'exec_mode': 'expected_round',          # paper A.4; overlay StrideEvalConfig default
    'expected_temp': 1.0,                   # overlay N1.5 default; paper gives no GR00T value
    'horizon_alpha': 4.0,                   # paper alpha_intra = 4
    'intra_cut_t': 0.25,                    # paper c_cut = 0.25
    'z_mode': 'window_rms',                 # paper "window RMS"
    'tau_window': 3,                        # paper W_tau = 3 (overlay CLI default 1: conflict, disclosed)
    'signal_mode': 'both',                  # paper lambda = 0.5 mix
    'use_speed_uniformity': True,
    'speed_window': 20,                     # paper A.4 W_h = 20 for RoboCasa GR1
    'speed_weight': 0.5,                    # paper lambda = 0.5
    'speed_beta': 4.0,                      # paper beta_inter = 4
    'speed_fallback': 'ehh_only',           # paper: intra-only without sufficient history
    'ts_epsilon': 0.05,                     # paper epsilon_exp = 0.05
    'ts_update_mode': 'kernel_forget',
    'ts_kernel_bandwidth': 10.0,            # paper sigma_K = 10
    'ts_forget_rho': 0.99,                  # paper rho_f = 0.99
    'ts_update_eta': 1.0,                   # paper eta = 1
    'ts_seed': 0,                           # overlay horizon_ts_seed default; + env_idx 0
}
SELECT_KEYS = frozenset(REFERENCE_AHS_PARAMS) - {'horizon_candidates', 'ts_seed'}


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class HorizonSelector:
    """Protocol. One instance per episode lifetime; start_episode resets all memory.

    needs_velocity: the server must return the denoising velocity trace.
    applies_horizon: False means executed K is always the native 16 (base/null arms).
    """
    needs_velocity = False
    applies_horizon = False
    identity: dict = {}

    def start_episode(self) -> None:
        raise NotImplementedError

    def select(self, *, velocity, actions_flat) -> tuple[int, dict]:
        raise NotImplementedError

    def record_executed(self, executed_flat) -> None:
        raise NotImplementedError


class FixedHorizon(HorizonSelector):
    """Paper Base arm: fixed K = H = 16 (identical to evaluate.py native GR00T)."""
    def __init__(self, k: int = NATIVE_CHUNK):
        if type(k) is not int or not 1 <= k <= NATIVE_CHUNK:
            raise ValueError('fixed K must be in [1, 16]')
        self.k = k
        self.identity = {'selector': 'fixed', 'k': k, 'version': 1}

    def start_episode(self):
        pass

    def select(self, *, velocity=None, actions_flat=None):
        return self.k, {'selector': 'fixed', 'chosen_stride': self.k}

    def record_executed(self, executed_flat):
        pass


class ChunkTrustAHS(HorizonSelector):
    """Training-free AHS with the upstream GR00T N1.5 overlay call signature.

    velocity: [T, H, D_model] float32 sampler velocities (normalized model space).
    actions_flat: [H, 12] float32 unnormalized native actions (xt_step / history space,
    as upstream uses _flatten_action_chunks of the env-space action dict).
    """
    needs_velocity = True
    applies_horizon = True

    def __init__(self, params: dict | None = None, velocity_dims: str = 'all'):
        params = dict(REFERENCE_AHS_PARAMS if params is None else params)
        if set(params) != set(REFERENCE_AHS_PARAMS):
            raise ValueError('AHS params must declare exactly the reference keys')
        cands = [int(c) for c in params['horizon_candidates']]
        if not cands or cands != sorted(set(cands)) or cands[0] < 1 or cands[-1] > NATIVE_CHUNK:
            raise ValueError('candidates must be increasing integers within [1, 16]')
        if velocity_dims not in ('all', 'effective'):
            raise ValueError('velocity_dims must be all (upstream) or effective')
        self.params = params
        self.velocity_dims = velocity_dims
        self.identity = {'selector': 'chunktrust_ahs', 'version': 1,
                         'upstream': 'hf618/ChunkTrust@a9fd57412b3a19a0728cd726b0558dd89395fb84:src/chunktrust/gr00t_ahs.py',
                         'params': params, 'velocity_dims': velocity_dims,
                         'params_sha256': canonical_sha256(params)}
        self.start_episode()

    def start_episode(self):
        # Upstream begin_episode: empty history, empty Beta state, rng(seed + env_idx).
        self.history = np.zeros((0, 0), dtype=np.float32)
        self.ts_state: dict = {}
        self.rng = np.random.default_rng(int(self.params['ts_seed']) + 0)
        self.decisions = 0

    def _velocity(self, velocity, used_h):
        v = np.asarray(velocity, dtype=np.float32)
        if v.ndim != 3 or v.shape[1] < used_h or not np.isfinite(v).all():
            raise ValueError('velocity trace must be finite [T, H, D]')
        v = v[:, :used_h, :]
        if self.velocity_dims == 'effective':
            v = v[:, :, :EFFECTIVE_ACTION_DIM]
        return v

    def select(self, *, velocity, actions_flat):
        flat = np.asarray(actions_flat, dtype=np.float32)
        if flat.ndim != 2 or not np.isfinite(flat).all():
            raise ValueError('actions must be finite [H, D]')
        used_h = min(NATIVE_CHUNK, int(flat.shape[0]))
        if self.history.shape[1] != flat.shape[1]:
            # upstream EvalTraceManager.ensure_action_history
            self.history = np.zeros((0, flat.shape[1]), dtype=np.float32)
        p = self.params
        k, info = select_exec_k_horizon(
            self._velocity(velocity, used_h), used_h,
            horizon_candidates=','.join(str(c) for c in p['horizon_candidates']),
            exec_mode=p['exec_mode'], expected_temp=p['expected_temp'],
            horizon_alpha=p['horizon_alpha'], intra_cut_t=p['intra_cut_t'],
            z_mode=p['z_mode'], tau_window=p['tau_window'],
            xt_step=np.asarray(flat[:used_h], dtype=np.float32),
            history_actions=self.history, signal_mode=p['signal_mode'],
            use_speed_uniformity=p['use_speed_uniformity'], speed_window=p['speed_window'],
            speed_weight=p['speed_weight'], speed_beta=p['speed_beta'],
            speed_fallback=p['speed_fallback'], ts_state=self.ts_state, ts_rng=self.rng,
            ts_epsilon=p['ts_epsilon'], ts_update_mode=p['ts_update_mode'],
            ts_kernel_bandwidth=p['ts_kernel_bandwidth'], ts_forget_rho=p['ts_forget_rho'],
            ts_update_eta=p['ts_update_eta'])
        self.decisions += 1
        if type(k) is not int or not 1 <= k <= used_h:
            raise ValueError('AHS returned an invalid execution horizon')
        return k, _json_safe(info)

    def record_executed(self, executed_flat):
        executed = np.asarray(executed_flat, dtype=np.float32)
        if executed.size == 0:
            return
        if self.history.size == 0 or self.history.shape[1] != executed.shape[1]:
            self.history = executed.copy()
        else:
            self.history = np.concatenate([self.history, executed], axis=0)

    def state_dict(self):
        return {'history': self.history.tolist(), 'ts_state': copy.deepcopy(self.ts_state),
                'rng': copy.deepcopy(self.rng.bit_generator.state), 'decisions': self.decisions}


class ShadowAHS(HorizonSelector):
    """Overhead/identity null: velocity readout ON, AHS computed and logged, K fixed 16."""
    needs_velocity = True
    applies_horizon = False

    def __init__(self, params=None, velocity_dims='all'):
        self.inner = ChunkTrustAHS(params, velocity_dims)
        self.identity = {'selector': 'shadow_ahs_fixed16', 'version': 1,
                         'inner': self.inner.identity}

    def start_episode(self):
        self.inner.start_episode()

    def select(self, *, velocity, actions_flat):
        k, info = self.inner.select(velocity=velocity, actions_flat=actions_flat)
        info = dict(info, shadow_ahs_k=k, chosen_stride=NATIVE_CHUNK)
        return NATIVE_CHUNK, info

    def record_executed(self, executed_flat):
        self.inner.record_executed(executed_flat)


ARMS = {
    'ahs': ('candidate', lambda c: ChunkTrustAHS(c['ahs_params'], c['velocity_dims'])),
    'base16': ('baseline', lambda c: FixedHorizon(NATIVE_CHUNK)),
    'null_trace16': ('null', lambda c: ShadowAHS(c['ahs_params'], c['velocity_dims'])),
}


def reference_config(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError('arm must be one of ' + ', '.join(ARMS))
    return {'schema': 'chunktrust-ahs-gr00t/1', 'arm': arm,
            'ahs_params': copy.deepcopy(REFERENCE_AHS_PARAMS), 'velocity_dims': 'all',
            'seeded_reset': True, 'selector_plugin': None, 'plugin_config': {}}


def validate_config(config: dict) -> dict:
    keys = {'schema', 'arm', 'ahs_params', 'velocity_dims', 'seeded_reset',
            'selector_plugin', 'plugin_config'}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError('harness config must have exactly keys ' + ', '.join(sorted(keys)))
    if config['schema'] != 'chunktrust-ahs-gr00t/1' or config['arm'] not in ARMS:
        raise ValueError('unknown schema/arm')
    if config['seeded_reset'] is not True:
        raise ValueError('paired arms require the seeded reset rule')
    if config['selector_plugin'] is not None and config['arm'] != 'ahs':
        raise ValueError('a candidate plugin composes on the ahs arm')
    ChunkTrustAHS(config['ahs_params'], config['velocity_dims'])  # validates params
    return config


def build_selector(config: dict, *, plugin_sha256: dict | None = None):
    """Return (unified_arm, selector). Plugins must be hash-bound in the frozen manifest."""
    validate_config(config)
    unified_arm, make = ARMS[config['arm']]
    selector = make(config)
    plugin = config['selector_plugin']
    if plugin:
        filename, sep, name = plugin.rpartition(':')
        path = Path(filename).resolve()
        if not sep or not name.isidentifier():
            raise ValueError('selector_plugin must be /path/file.py:factory')
        if plugin_sha256 is None or str(path) not in plugin_sha256:
            raise ValueError('selector_plugin must be frozen in the manifest source hashes')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != plugin_sha256[str(path)]:
            raise ValueError('selector_plugin changed after freeze')
        spec = importlib.util.spec_from_file_location('chunktrust_candidate_plugin', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        selector = getattr(module, name)(selector, copy.deepcopy(config['plugin_config']))
        for attr in ('start_episode', 'select', 'record_executed'):
            if not callable(getattr(selector, attr, None)):
                raise ValueError('plugin must return a HorizonSelector')
        unified_arm = 'candidate'
    return unified_arm, selector


def variable_horizon_harness_class():
    """Subclass created lazily so this module stays importable without robometaharness."""
    from robometaharness.unified import UnifiedHarness

    class VariableHorizonUnifiedHarness(UnifiedHarness):
        """Unified identity runtime whose decision cadence follows the executed prefix.

        propose() is unchanged (identity, one proposal, native object). The selected
        horizon only truncates execution of that same chunk; commit() then moves the
        next legal decision step to step + executed. Documented adapter exception:
        core Decision.execute_steps stays the native maximum min(16, remaining).
        """
        def propose(self, raw_observation, *, step):
            decision = super().propose(raw_observation, step=step)
            self._pending = (step, decision.execute_steps)
            return decision

        def commit(self, executed: int, *, selector_info=None):
            step, maximum = getattr(self, '_pending', (None, None))
            if step is None or type(executed) is not int or not 0 < executed <= maximum:
                raise ValueError('commit must follow propose with 0 < executed <= native maximum')
            self._next_step = step + executed
            self._pending = (None, None)
            if self.trace_sink:
                self.trace_sink({'event': 'horizon_commit', 'step': step, 'executed_steps': executed,
                                 'native_max_steps': maximum, 'manifest_sha256': self.manifest_sha256,
                                 'selector_chosen': None if selector_info is None
                                 else selector_info.get('chosen_stride')})

    return VariableHorizonUnifiedHarness
