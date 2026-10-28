"""Evaluator-only per-step log for development phases (analyst request, 2026-10-26).

A pass-through tap is inserted between the pinned MultiStepWrapper and its inner gym
env. The wrapper calls inner.step(per-step action) once per executed sub-step; the tap
forwards the call unchanged and copies (a) the executed per-step action arrays (gripper
command = action.gripper_close, plus EEF/base/control-mode commands) and (b) the
per-step proprio returned by the env (gripper qpos, EEF relative pose, base pose).
The simulator success flag is logged once per chunk by the episode loop (the value the
loop already reads). Nothing here is returned to the policy, the port, the selector,
the harness or memory; the log is written to evaluator_steps.npz after the episode.

Default OFF (--evaluator-step-log enables it); keep it off for validation unless parity
requires it. Size: ~30 float32 per step, about 0.1-0.15 MB per 1050-step episode.
"""
from __future__ import annotations

STATE_KEYS = ('state.gripper_qpos', 'state.end_effector_position_relative',
              'state.end_effector_rotation_relative', 'state.base_position', 'state.base_rotation')


class StepTap:
    def __init__(self, inner):
        object.__setattr__(self, '_inner', inner)
        object.__setattr__(self, 'rows', [])
        object.__setattr__(self, 'chunk', -1)
        object.__setattr__(self, 'chunk_success', [])

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_inner'), name)

    def __setattr__(self, name, value):
        if name in ('chunk',):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_inner'), name, value)

    def step(self, action):
        import numpy as np
        out = object.__getattribute__(self, '_inner').step(action)
        observation = out[0]
        self.rows.append({'chunk': self.chunk,
                          'action': {k: np.array(v, dtype=np.float32, copy=True) for k, v in action.items()},
                          'state': {k: np.array(observation[k], dtype=np.float32, copy=True)
                                    for k in STATE_KEYS if k in observation}})
        return out

    def mark_chunk(self, index):
        self.chunk = index

    def record_chunk_success(self, index, success):
        self.chunk_success.append((int(index), bool(success)))

    def save(self, path):
        import numpy as np
        arrays = {'chunk': np.array([r['chunk'] for r in self.rows], dtype=np.int32),
                  'chunk_success_index': np.array([c for c, _ in self.chunk_success], dtype=np.int32),
                  'chunk_success': np.array([s for _, s in self.chunk_success], dtype=bool)}
        if self.rows:
            for key in self.rows[0]['action']:
                arrays['action/' + key] = np.stack([r['action'][key] for r in self.rows])
            for key in self.rows[0]['state']:
                arrays['state/' + key] = np.stack([r['state'][key] for r in self.rows])
        np.savez_compressed(path, **arrays)
        return {'file': str(path), 'steps': len(self.rows), 'chunks_with_success_flag': len(self.chunk_success),
                'evaluator_only': True}


def install(env):
    """Insert the tap below MultiStepWrapper; returns the tap (evaluator-held only)."""
    tap = StepTap(env.env)
    env.env = tap
    return tap
