"""Kinetix RTC backend runtime (child interpreter only: imports JAX, flax and the pinned upstream code).

One HarnessStudio episode = one batched case (level, block of B rollouts, protocol point (d, s)) run in
lockstep through the UNCHANGED core.pipeline.run_episode:

- KeyChain reproduces real-time-chunking-kinetix@9296f31d eval_flow.eval's single rng chain exactly
  (reset split, warm-start split, then per decision one policy split and one split per env step).
- KinetixEnvPort is the arm-agnostic actuator queue: executes prev[:, :d] ++ new[:, d:s], then
  prev <- concat(new[:, s:], 0) (eval_flow.py:119-137); returns the public observation
  {instruction, obs, prev_chunk}; never terminates early (fixed ceil(256/s) decisions).
- KinetixClient is the in-process policy port: baseline = upstream FlowPolicy.action; a guide() sampler
  runs the per-rollout VJP guided flow loop (model.py:230-265 structure); the null arm integrates the
  native batched velocity (same forward as FlowPolicy.action) and ALSO computes and returns the guided
  velocity stack as a jit output (cannot be dead-code eliminated; its sha256 enters the trace).
"""
from __future__ import annotations

import functools
import hashlib
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


class KeyChain:
    """The single upstream rng chain of one level (eval_flow.py:145-148, 79, 74)."""

    def __init__(self, level_key):
        self.rng = level_key

    def split(self):
        import jax
        self.rng, key = jax.random.split(self.rng)
        return key


class KinetixCase:
    """Env, level and frozen policy for one level; jitted functions shared across cases with equal shapes."""

    def __init__(self, level: str, *, rollouts: int, inference_delay: int, execute_horizon: int,
                 level_universe=KF.LEVELS, seed: int = 0, block: int = 0, root=None, epoch_dir=KF.CHECKPOINT_EPOCH_DIR):
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        import kinetix.environment.env as kenv
        import kinetix.environment.env_state as kenv_state
        import kinetix.environment.wrappers as wrappers
        U = upstream(root)
        te, M = U["train_expert"], U["model"]
        if level not in level_universe:
            raise ValueError(f"{level} is not in the level universe")
        d, s = inference_delay, execute_horizon
        if not (0 <= d <= s and 1 <= s <= KF.CHUNK_LEN):
            raise ValueError(f"need 0 <= d <= s <= {KF.CHUNK_LEN}, s >= 1")
        self.level, self.rollouts, self.d, self.s = level, int(rollouts), int(d), int(s)
        self.level_universe, self.seed, self.block = tuple(level_universe), int(seed), int(block)
        self.level_index = self.level_universe.index(level)
        static_env_params = kenv_state.StaticEnvParams(**te.LARGE_ENV_PARAMS, frame_skip=te.FRAME_SKIP)
        self.env_params = kenv_state.EnvParams()
        cwd = os.getcwd()
        try:  # upstream loads levels by relative path from the checkout root
            os.chdir(U["upstream_dir"])
            levels = te.load_levels([level], static_env_params, self.env_params)
        finally:
            os.chdir(cwd)
        self.level_state = jax.tree.map(lambda x: x[0], levels)
        static_env_params = static_env_params.replace(screen_dim=te.SCREEN_DIM)
        if self.rollouts not in _WORLD_CACHE:  # one env per batch width; levels enter as arguments, as in _eval
            base = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
            _WORLD_CACHE[self.rollouts] = (base, te.BatchEnvWrapper(
                wrappers.LogWrapper(wrappers.AutoReplayWrapper(te.NoisyActionWrapper(base))), self.rollouts))
        base, self.env = _WORLD_CACHE[self.rollouts]
        self.obs_dim = jax.eval_shape(base.reset_to_level, jax.random.key(0), self.level_state, self.env_params)[0].shape[-1]
        self.action_dim = base.action_space(self.env_params).shape[0]
        if (self.obs_dim, self.action_dim) != (KF.OBS_DIM, KF.ACTION_DIM):
            raise RuntimeError(f"unexpected Kinetix shapes {(self.obs_dim, self.action_dim)}")
        keys = jax.random.split(jax.random.key(self.seed), len(self.level_universe))
        level_key = keys[self.level_index]
        self.level_key = level_key if self.block == 0 else jax.random.fold_in(level_key, self.block)
        ckpt = Path(U["root"]) / "assets" / "bc" / epoch_dir / "policies" / f"{level_name(level)}.pkl"
        self.checkpoint = {"path": str(ckpt), "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
        with ckpt.open("rb") as f:
            state_dict = pickle.load(f)
        policy = M.FlowPolicy(obs_dim=self.obs_dim, action_dim=self.action_dim, config=M.ModelConfig(),
                              rngs=nnx.Rngs(self.level_key))
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        self.graphdef, self.params = graphdef, state
        self.horizon_steps = math.ceil(self.env_params.max_timesteps / s) * s
        self.decisions = self.horizon_steps // s
        self._jnp = jnp

    # ---- jitted pieces (static shapes; params passed as arguments, as in eval_flow's _eval) ----
    @functools.cached_property
    def fns(self):
        return _functions(self.env, self.env_params, self.graphdef, self.d, self.s)


_FN_CACHE = {}
_WORLD_CACHE = {}


def _functions(env, env_params, graphdef, d, s):
    key = (id(env), d, s)
    if key in _FN_CACHE:
        return _FN_CACHE[key]
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    H = KF.CHUNK_LEN
    n = KF.NUM_FLOW_STEPS

    def policy_of(params):
        return nnx.merge(graphdef, params)

    @jax.jit
    def reset(level_key, level):
        rng, key = jax.random.split(level_key)
        obs, env_state = env.reset_to_level(key, level, env_params)
        return rng, obs, env_state

    @jax.jit
    def naive(params, key, obs):
        return policy_of(params).action(key, obs, n)

    def _guided(params, key, obs, prev, guide, effective):
        policy = policy_of(params)
        dt = 1 / n

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch (model.py:235)
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

        noise = jax.random.normal(key, shape=(obs.shape[0], H, policy.action_dim))
        (x_1, _), guided = jax.lax.scan(step, (noise, 0.0), length=n)
        return x_1, guided

    guided_cache = {}

    def guided(sampler, effective):
        """One jit per (sampler class + identity, arm effect); the bound guide is a static closure."""
        import json
        k = (type(sampler).__module__, type(sampler).__qualname__, json.dumps(sampler.identity, sort_keys=True),
             bool(effective))
        if k not in guided_cache:
            guided_cache[k] = jax.jit(functools.partial(_guided, guide=sampler.guide, effective=bool(effective)))
        return guided_cache[k]

    @jax.jit
    def execute(rng, obs, env_state, prev, new):
        to_exec = jnp.concatenate([prev[:, :d], new[:, d:s]], axis=1)
        aligned = jnp.concatenate([new[:, s:], jnp.zeros((obs.shape[0], s, new.shape[-1]), new.dtype)], axis=1)

        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, reward, info["returned_episode_returns"],
                                                     info["returned_episode_lengths"], info["returned_episode_solved"])

        (rng, obs, env_state), outs = jax.lax.scan(step, (rng, obs, env_state), to_exec.transpose(1, 0, 2))
        return rng, obs, env_state, aligned, to_exec, outs

    out = {"reset": reset, "naive": naive, "guided": guided, "execute": execute}
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


class KinetixEnvPort:
    """Arm-agnostic actuator queue + batched Kinetix env (EnvPort protocol of core.pipeline)."""

    def __init__(self, case: KinetixCase, chain: KeyChain, *, record_steps: bool = False):
        self.case, self.chain = case, chain
        f = case.fns
        rng, self.obs, self.env_state = f["reset"](chain.rng, case.level_state)
        chain.rng = rng                                   # reset consumed one split
        warm_key = chain.split()                          # eval_flow.py:147-148
        self.prev = f["naive"](case.params, warm_key, self.obs)
        self.warmstart_policy_calls = 1
        self.record_steps = record_steps
        self.steps = []                                   # per decision: (dones, rewards, returns, lengths, solved)
        self.executed = []                                # per decision executed actions [s, B, A] (record_steps)
        self.obs_log = []                                 # per decision input obs [B, obs] (record_steps)
        self.fingerprints = []                            # per decision: obs fp before, executed-action fp

    def observation(self):
        return {"kinetix/instruction": KF.INSTRUCTION, "kinetix/obs": np.asarray(self.obs),
                "kinetix/prev_chunk": np.asarray(self.prev)}

    def execute(self, native, start, count):
        from harnessstudio.core.pipeline import StepResult
        if start != 0 or count != self.case.s:
            raise RuntimeError(f"kinetix-flow executes exactly s={self.case.s} steps per decision from 0")
        self._obs_before = self.obs
        obs_fp = fingerprint(np.asarray(self.obs))
        rng, obs, env_state, aligned, to_exec, outs = self.case.fns["execute"](
            self.chain.rng, self.obs, self.env_state, self.prev, native)
        self.chain.rng, self.obs, self.env_state, self.prev = rng, obs, env_state, aligned
        host = tuple(np.asarray(o) for o in outs)        # each [s, B]
        self.steps.append(host)
        ex = np.asarray(to_exec)                          # [B, s, A]
        if self.record_steps:
            self.executed.append(ex.transpose(1, 0, 2))
            self.obs_log.append(np.asarray(self._obs_before))
        self.fingerprints.append((obs_fp, fingerprint(ex)))
        return StepResult(self.observation(), False, count, False)

    def outcomes(self) -> dict:
        """First-episode metrics per rollout, as eval_flow.py:157-163."""
        dones, rewards, rets, lens, solved = (np.concatenate([s[i] for s in self.steps], axis=0) for i in range(5))
        if dones.shape[0] < self.case.env_params.max_timesteps:
            raise RuntimeError("episode shorter than max_timesteps")
        first = np.argmax(dones, axis=0)
        idx = np.arange(dones.shape[1])
        return {"rollout_solved": solved[first, idx], "rollout_returns": rets[first, idx],
                "rollout_lengths": lens[first, idx], "rewards": rewards, "dones": dones}


class KinetixClient:
    """In-process PolicyClient over the frozen FlowPolicy (reference_rng noise; NoiseSchedule seeds recorded only)."""

    def __init__(self, case: KinetixCase, chain: KeyChain, sampler=None):
        self.case, self.chain, self.sampler = case, chain, sampler
        self.guidance_sha256 = []
        self.calls = 0
        self.last_guided = None

    def _inputs(self, values):
        import jax.numpy as jnp
        return jnp.asarray(values["kinetix/obs"]), jnp.asarray(values["kinetix/prev_chunk"])

    @staticmethod
    def _flat(native):
        a = np.asarray(native)                            # [B, H, A]
        return a.transpose(1, 0, 2).reshape(a.shape[1], -1)

    def get_action_seeded(self, observation, seed):
        obs, _prev = self._inputs(observation)
        key = self.chain.split()
        native = self.case.fns["naive"](self.case.params, key, obs)
        self.calls += 1
        return native, self._flat(native)

    def propose(self, observation, *, arm, decision_index, step_index, seeds, want):
        from harnessstudio.core.runtime import ProposeResult
        if len(seeds) != 1:
            raise RuntimeError("kinetix-flow supports one proposal per decision in this version")
        obs, prev = self._inputs(observation)
        key = self.chain.split()
        extras = {}
        if self.sampler is None:
            native = self.case.fns["naive"](self.case.params, key, obs)
        else:
            fn = self.case.fns["guided"](self.sampler, arm == "candidate")
            native, guided = fn(self.case.params, key, obs, prev)
            digest = hashlib.sha256(np.ascontiguousarray(np.asarray(guided)).tobytes()).hexdigest()
            self.guidance_sha256.append(digest)
            self.last_guided = guided                        # [n, B, H, A] (parity diagnostics)
            extras["sampler"] = {"guidance_sha256": digest}
        self.calls += 1
        return ProposeResult([native], [self._flat(native)], extras, {"policy_calls": 1})
