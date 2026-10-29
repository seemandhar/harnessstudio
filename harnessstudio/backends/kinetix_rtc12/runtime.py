"""Kinetix RTC backend runtime (child interpreter only: imports JAX, flax and the pinned upstream code).

Repair 1 (level-batched): one HarnessStudio episode = one batched case = (protocol point (d, s), the WHOLE
ordered level universe U, block, B rollouts per level), run in lockstep through the UNCHANGED
core.pipeline.run_episode. Every per-decision jitted function is wrapped exactly as the authors wrap _eval
(real-time-chunking-kinetix@9296f31d eval_flow.py:222-248): jax.jit(shard_map(vmap over the level axis)),
mesh = all local devices, PartitionSpec('x'); per-level keys split(key(seed), L), per-level stacked pure
state dicts and level states are vmapped ARGUMENTS; inside, FlowPolicy is built per level exactly as _eval
does and the batch of B rollouts is the inner BatchEnvWrapper vmap. Arrays are [L, B, ...]; L x B is never
reshaped into one batch inside a jitted computation (the flat [H, L*B*A] action exists on the host only).

- UniverseRng holds the per-level rng [L]; every jitted call consumes and returns it in the authors' split
  order (reset split, warm-start split, then per decision one policy split and one split per env step).
- KinetixEnvPort is the arm-agnostic actuator queue: executes prev[:, :, :d] ++ new[:, :, d:s], then
  prev <- concat(new[:, :, s:], 0) (eval_flow.py:119-137); public observation {instruction, obs, prev_chunk};
  never terminates early (fixed ceil(256/s) decisions).
- KinetixClient: baseline/naive = upstream FlowPolicy.action; a guide() sampler runs the per-rollout VJP
  guided flow loop (model.py:230-265 structure); the null arm integrates the native batched velocity (the
  forward of FlowPolicy.action) and ALSO returns the guided velocity stack as a jit output (B4; its sha256
  enters the trace).
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from harnessstudio.policies import kinetix_flow as KF

_UPSTREAM = {}


def upstream(root=None):
    """Import the pinned upstream modules after verifying their sha256 (refuse on mismatch)."""
    root = Path(root or KF.kinetix_root())
    if _UPSTREAM.get("root") == root:
        return _UPSTREAM
    up = root / "upstream"
    for rel, want in KF.UPSTREAM_SHA256.items():
        got = hashlib.sha256((up / rel).read_bytes()).hexdigest()
        if got != want:
            raise RuntimeError(f"pinned upstream file changed: {rel} {got} != {want}")
    src = str(up / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import eval_flow
    import model as _model
    import train_expert
    _UPSTREAM.update(root=root, eval_flow=eval_flow, model=_model, train_expert=train_expert, upstream_dir=up)
    return _UPSTREAM


def level_name(level: str) -> str:
    return level.replace("/", "_").replace(".json", "")


def universe_tag(universe) -> str:
    return hashlib.sha256(json.dumps(list(universe)).encode()).hexdigest()[:8]


class UniverseRng:
    """Per-level rng chains [L], advanced inside the level-vmapped jitted calls (eval_flow.py:145-148, 79, 74)."""

    def __init__(self, level_keys):
        self.rng = level_keys


class KinetixUniverseCase:
    """Env, stacked level states and stacked frozen policy params for one ordered level universe."""

    def __init__(self, universe, *, rollouts: int, inference_delay: int, execute_horizon: int, seed: int = 0,
                 block: int = 0, root=None, epoch_dir=KF.CHECKPOINT_EPOCH_DIR):
        import jax
        import jax.numpy as jnp
        import kinetix.environment.env as kenv
        import kinetix.environment.env_state as kenv_state
        import kinetix.environment.wrappers as wrappers
        U = upstream(root)
        te = U["train_expert"]
        universe = tuple(universe)
        if not universe or len(set(universe)) != len(universe) or any(lv not in KF.LEVELS for lv in universe):
            raise ValueError(f"the level universe must be distinct kinetix-rtc12 levels: {universe}")
        d, s = inference_delay, execute_horizon
        if not (0 <= d <= s and 1 <= s <= KF.CHUNK_LEN):
            raise ValueError(f"need 0 <= d <= s <= {KF.CHUNK_LEN}, s >= 1")
        self.universe, self.rollouts, self.d, self.s = universe, int(rollouts), int(d), int(s)
        self.seed, self.block = int(seed), int(block)
        static_env_params = kenv_state.StaticEnvParams(**te.LARGE_ENV_PARAMS, frame_skip=te.FRAME_SKIP)
        self.env_params = kenv_state.EnvParams()
        cwd = os.getcwd()
        try:  # upstream loads levels by relative path from the checkout root (eval_flow.py:193)
            os.chdir(U["upstream_dir"])
            self.levels = te.load_levels(list(universe), static_env_params, self.env_params)
        finally:
            os.chdir(cwd)
        static_env_params = static_env_params.replace(screen_dim=te.SCREEN_DIM)
        if self.rollouts not in _WORLD_CACHE:  # one env per batch width; levels enter as arguments, as in _eval
            base = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
            _WORLD_CACHE[self.rollouts] = (base, te.BatchEnvWrapper(
                wrappers.LogWrapper(wrappers.AutoReplayWrapper(te.NoisyActionWrapper(base))), self.rollouts))
        self.base_env, self.env = _WORLD_CACHE[self.rollouts]
        self.obs_dim = jax.eval_shape(self.base_env.reset_to_level, jax.random.key(0),
                                      jax.tree.map(lambda x: x[0], self.levels), self.env_params)[0].shape[-1]
        self.action_dim = self.base_env.action_space(self.env_params).shape[0]
        if (self.obs_dim, self.action_dim) != (KF.OBS_DIM, KF.ACTION_DIM):
            raise RuntimeError(f"unexpected Kinetix shapes {(self.obs_dim, self.action_dim)}")
        keys = jax.random.split(jax.random.key(self.seed), len(universe))          # eval_flow.py:248
        self.level_keys = keys if self.block == 0 else jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, self.block)
        dicts, self.checkpoints = [], []
        for lv in universe:                                                          # eval_flow.py:201-211
            ckpt = Path(U["root"]) / "assets" / "bc" / epoch_dir / "policies" / f"{level_name(lv)}.pkl"
            self.checkpoints.append({"level": lv, "path": str(ckpt), "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()})
            with ckpt.open("rb") as f:
                dicts.append(pickle.load(f))
        self.state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *dicts))
        self.horizon_steps = math.ceil(self.env_params.max_timesteps / s) * s
        self.decisions = self.horizon_steps // s

    @property
    def L(self) -> int:
        return len(self.universe)

    @functools.cached_property
    def fns(self):
        return _functions(self.env, self.env_params, self.obs_dim, self.action_dim, self.d, self.s)


_FN_CACHE = {}
_WORLD_CACHE = {}


def level_vmapped(fn, n_args):
    """jit(shard_map(vmap(fn))) over the leading level axis, as eval_flow.py:222-228 wraps _eval."""
    import jax
    from jax.experimental import shard_map
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    return jax.jit(shard_map.shard_map(jax.vmap(fn), mesh=mesh, in_specs=(pspec,) * n_args, out_specs=pspec),
                   in_shardings=sharding, out_shardings=sharding)


def _functions(env, env_params, obs_dim, action_dim, d, s):
    key = (id(env), d, s)
    if key in _FN_CACHE:
        return _FN_CACHE[key]
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    M = upstream()["model"]
    H = KF.CHUNK_LEN
    n = KF.NUM_FLOW_STEPS

    def policy_of(level_key, state_dict):          # eval_flow.py:230-238, per level inside the vmap
        policy = M.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=M.ModelConfig(), rngs=nnx.Rngs(level_key))
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        return nnx.merge(graphdef, state)

    def reset(rng, level):                          # eval_flow.py:145-146
        rng, k = jax.random.split(rng)
        obs, env_state = env.reset_to_level(k, level, env_params)
        return rng, obs, env_state

    def naive(rng, obs, state_dict, level_key):     # eval_flow.py:147-148 (warm start) and :79-81
        rng, k = jax.random.split(rng)
        return rng, policy_of(level_key, state_dict).action(k, obs, n)

    def _guided(rng, obs, prev, state_dict, level_key, guide, effective):
        rng, k = jax.random.split(rng)
        policy = policy_of(level_key, state_dict)
        dt = 1 / n

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over the B rollouts (model.py:235)
            def per_rollout(obs, x_t, y, t):
                def denoiser(x_t):
                    v_t = policy(obs[None], x_t[None], t)[0]
                    return x_t + v_t * (1 - t), v_t

                x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                return guide(jnp, x_t, t, v_t, x_1, lambda cot: vjp_fun(cot)[0], y, d, s, H)

            v_guided = per_rollout(obs, x_t, prev, time)
            if effective:
                v_used = v_guided
            else:  # null (B4): the same batched forward as FlowPolicy.action
                v_used = policy(obs, x_t, time)
            return (x_t + dt * v_used, time + dt), v_guided

        noise = jax.random.normal(k, shape=(obs.shape[0], H, policy.action_dim))
        (x_1, _), guided = jax.lax.scan(step, (noise, 0.0), length=n)
        return rng, x_1, guided

    guided_cache = {}

    def guided(sampler, effective):
        """One compiled function per (sampler class + identity, arm effect); the bound guide is a static closure."""
        k = (type(sampler).__module__, type(sampler).__qualname__, json.dumps(sampler.identity, sort_keys=True),
             bool(effective))
        if k not in guided_cache:
            guided_cache[k] = level_vmapped(functools.partial(_guided, guide=sampler.guide, effective=bool(effective)), 5)
        return guided_cache[k]

    def execute(rng, obs, env_state, prev, new):    # eval_flow.py:71-143 (one decision)
        to_exec = jnp.concatenate([prev[:, :d], new[:, d:s]], axis=1)
        aligned = jnp.concatenate([new[:, s:], jnp.zeros((obs.shape[0], s, new.shape[-1]))], axis=1)

        def step(carry, action):
            rng, obs, env_state = carry
            rng, k = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(k, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, reward, info["returned_episode_returns"],
                                                     info["returned_episode_lengths"], info["returned_episode_solved"])

        (rng, obs, env_state), outs = jax.lax.scan(step, (rng, obs, env_state), to_exec.transpose(1, 0, 2))
        return rng, obs, env_state, aligned, to_exec, outs

    out = {"reset": level_vmapped(reset, 2), "naive": level_vmapped(naive, 4), "guided": guided,
           "execute": level_vmapped(execute, 5)}
    _FN_CACHE[key] = out
    return out


def fingerprint(x) -> np.ndarray:
    """Per-rollout 2x32-bit fingerprint of float32 rows [B, ...] (equality detection, not security)."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).reshape(np.asarray(x).shape[0], -1).view(np.uint32)
    m = a.shape[1]
    r = np.random.default_rng(20261028)
    m1 = (r.integers(0, 2 ** 31, m, dtype=np.uint64) * 2 + 1).astype(np.uint64)
    m2 = (r.integers(0, 2 ** 31, m, dtype=np.uint64) * 2 + 1).astype(np.uint64)
    a64 = a.astype(np.uint64)
    h1 = (a64 * m1).sum(axis=1, dtype=np.uint64) & 0xFFFFFFFF
    h2 = ((a64 ^ (a64 >> np.uint64(16))) * m2).sum(axis=1, dtype=np.uint64) & 0xFFFFFFFF
    return np.stack([h1, h2], axis=1).astype(np.uint32)


def level_fingerprints(x) -> np.ndarray:
    """[L, B, ...] -> [L, B, 2]."""
    x = np.asarray(x)
    return np.stack([fingerprint(x[i]) for i in range(x.shape[0])])


class KinetixEnvPort:
    """Arm-agnostic actuator queue + level-batched Kinetix env (EnvPort protocol of core.pipeline)."""

    def __init__(self, case: KinetixUniverseCase, rng: UniverseRng, *, record_steps: bool = False):
        self.case, self.rng = case, rng
        f = case.fns
        rng.rng, self.obs, self.env_state = f["reset"](rng.rng, case.levels)
        rng.rng, self.prev = f["naive"](rng.rng, self.obs, case.state_dicts, case.level_keys)   # warm start
        self.warmstart_policy_calls = 1
        self.record_steps = record_steps
        self.steps = []                                   # per decision: (dones, rewards, returns, lengths, solved) [L, s, B]
        self.executed = []                                # per decision executed actions [L, s, B, A] (record_steps)
        self.obs_log = []                                 # per decision input obs [L, B, obs] (record_steps)
        self.fingerprints = []                            # per decision: (obs fp [L, B, 2], executed-action fp [L, B, 2])

    def observation(self):
        return {"kinetix/instruction": KF.INSTRUCTION, "kinetix/obs": np.asarray(self.obs),
                "kinetix/prev_chunk": np.asarray(self.prev)}

    def execute(self, native, start, count):
        from harnessstudio.core.pipeline import StepResult
        if start != 0 or count != self.case.s:
            raise RuntimeError(f"kinetix-flow executes exactly s={self.case.s} steps per decision from 0")
        obs_before = np.asarray(self.obs)
        rng, obs, env_state, aligned, to_exec, outs = self.case.fns["execute"](
            self.rng.rng, self.obs, self.env_state, self.prev, native)
        self.rng.rng, self.obs, self.env_state, self.prev = rng, obs, env_state, aligned
        self.steps.append(tuple(np.asarray(o) for o in outs))
        ex = np.asarray(to_exec)                          # [L, B, s, A]
        if self.record_steps:
            self.executed.append(ex.transpose(0, 2, 1, 3))
            self.obs_log.append(obs_before)
        self.fingerprints.append((level_fingerprints(obs_before), level_fingerprints(ex)))
        return StepResult(self.observation(), False, count, False)

    def outcomes(self) -> dict:
        """First-episode metrics per rollout and level, as eval_flow.py:157-163. Arrays [L, B] (rewards/dones [L, T, B])."""
        dones, rewards, rets, lens, solved = (np.concatenate([s[i] for s in self.steps], axis=1) for i in range(5))
        if dones.shape[1] < self.case.env_params.max_timesteps:
            raise RuntimeError("episode shorter than max_timesteps")
        first = np.argmax(dones, axis=1)                  # [L, B]
        li = np.arange(dones.shape[0])[:, None]
        bi = np.arange(dones.shape[2])[None, :]
        return {"rollout_solved": solved[li, first, bi], "rollout_returns": rets[li, first, bi],
                "rollout_lengths": lens[li, first, bi], "rewards": rewards, "dones": dones}


class KinetixClient:
    """In-process PolicyClient over the frozen FlowPolicy (reference_rng noise; NoiseSchedule seeds recorded only)."""

    def __init__(self, case: KinetixUniverseCase, rng: UniverseRng, sampler=None):
        self.case, self.rng, self.sampler = case, rng, sampler
        self.guidance_sha256 = []
        self.calls = 0
        self.last_guided = None

    def _inputs(self, values):
        import jax.numpy as jnp
        return jnp.asarray(values["kinetix/obs"]), jnp.asarray(values["kinetix/prev_chunk"])

    @staticmethod
    def _flat(native):
        a = np.asarray(native)                            # [L, B, H, A] -> host-only [H, L*B*A]
        return a.transpose(2, 0, 1, 3).reshape(a.shape[2], -1)

    def get_action_seeded(self, observation, seed):
        obs, _prev = self._inputs(observation)
        self.rng.rng, native = self.case.fns["naive"](self.rng.rng, obs, self.case.state_dicts, self.case.level_keys)
        self.calls += 1
        return native, self._flat(native)

    def propose(self, observation, *, arm, decision_index, step_index, seeds, want):
        from harnessstudio.core.runtime import ProposeResult
        if len(seeds) != 1:
            raise RuntimeError("kinetix-flow supports one proposal per decision in this version")
        obs, prev = self._inputs(observation)
        extras = {}
        if self.sampler is None:
            self.rng.rng, native = self.case.fns["naive"](self.rng.rng, obs, self.case.state_dicts, self.case.level_keys)
        else:
            fn = self.case.fns["guided"](self.sampler, arm == "candidate")
            self.rng.rng, native, guided = fn(self.rng.rng, obs, prev, self.case.state_dicts, self.case.level_keys)
            digest = hashlib.sha256(np.ascontiguousarray(np.asarray(guided)).tobytes()).hexdigest()
            self.guidance_sha256.append(digest)
            self.last_guided = guided                        # [L, n, B, H, A] (parity diagnostics)
            extras["sampler"] = {"guidance_sha256": digest}
        self.calls += 1
        return ProposeResult([native], [self._flat(native)], extras, {"policy_calls": 1})
