"""GR00T N1.5 / RoboCasa365 public-input helpers and task-registry probe.

Vendored from RoboMetaHarness robometaharness/benchmarking/groot_worker.py (selected
functions; see bench/PROVENANCE.json). Run with the RobotPolicyBench gr00t-n15 Python.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PUBLIC_KEYS = frozenset({
    'video.robot0_agentview_left', 'video.robot0_agentview_right', 'video.robot0_eye_in_hand',
    'state.end_effector_position_relative', 'state.end_effector_rotation_relative',
    'state.gripper_qpos', 'state.base_position', 'state.base_rotation',
    'annotation.human.task_description',
})
ACTION_SHAPES = {
    'action.end_effector_position': 3, 'action.end_effector_rotation': 3,
    'action.gripper_close': 1, 'action.base_motion': 4, 'action.control_mode': 1,
}
GROUPS = ('atomic_seen', 'composite_seen', 'composite_unseen')


def _write(path, value):
    from .common import write_json
    write_json(Path(path), value)


def _append(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    stream.flush()


def hashes(values):
    import numpy as np
    result = {}
    for key, value in sorted(values.items()):
        a = np.asarray(value)
        payload = (json.dumps(a.tolist(), ensure_ascii=False).encode()
                   if a.dtype.kind in 'OUS' else a.tobytes())
        header = json.dumps([key, a.dtype.str, list(a.shape)]).encode()
        result[key] = hashlib.sha256(header + b'\0' + payload).hexdigest()
    result['__aggregate__'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def public_copy(observation):
    import numpy as np
    if set(observation) != PUBLIC_KEYS:
        raise ValueError('Native nine-key public observation schema changed')
    return {key: np.asarray(value).copy() for key, value in observation.items()}


def instruction(public):
    import numpy as np
    text = np.asarray(public['annotation.human.task_description'])
    if text.size != 1:
        raise ValueError('Expected one original public instruction')
    value = str(text.reshape(-1)[0])
    if not value.strip():
        raise ValueError('Empty public instruction')
    return value


def flatten_native(actions):
    import numpy as np
    if not isinstance(actions, dict) or set(actions) != set(ACTION_SHAPES):
        raise ValueError('Native GR00T action dictionary schema changed')
    for key, width in ACTION_SHAPES.items():
        a = actions[key]
        if (not isinstance(a, np.ndarray) or a.dtype != np.float32
                or a.shape != (16, width) or not np.isfinite(a).all()):
            raise ValueError('Native actions must be finite float32 arrays of shape (16, width)')
    return np.concatenate([actions[key] for key in sorted(ACTION_SHAPES)], axis=-1)


def selected_native(flat, original_actions, original_flat):
    """Identity retains raw objects; changed plugin actions use the native field order."""
    import numpy as np
    # Validate the selection even on the identity path (plugins may mutate arrays).
    a = np.asarray(flat)
    if a.shape != (16, 12) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('Selected chunk must be finite float32 (16, 12)')
    if flat is original_flat and np.array_equal(a, flatten_native(original_actions)):
        return original_actions
    result = {}
    offset = 0
    for key in sorted(ACTION_SHAPES):
        width = ACTION_SHAPES[key]
        result[key] = a[:, offset:offset + width].copy()
        offset += width
    flatten_native(result)
    return result


class PublicVideo:
    """Only already returned public RGB frames; no rendering or imageio imports."""
    def __init__(self, path):
        self.path = Path(path)
        self.process = None
        self.shape = None
        self.stderr = None

    def append(self, public):
        import numpy as np
        frame = np.asarray(public['video.robot0_agentview_left'])[-1]
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Video requires public uint8 HxWx3 RGB')
        if self.process is None:
            self.shape = frame.shape
            self.stderr = self.path.with_suffix('.ffmpeg.log').open('wb')
            self.process = subprocess.Popen([
                '/usr/bin/ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
                f'{frame.shape[1]}x{frame.shape[0]}', '-framerate', '1.25', '-i', 'pipe:0',
                '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(self.path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.stderr)
        if frame.shape != self.shape:
            raise ValueError('Public video shape changed')
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        if self.process is None:
            if self.stderr:
                self.stderr.close()
            return
        error = None
        try:
            self.process.stdin.close()
        except OSError as exc:
            error = exc
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError('ffmpeg did not finish')
        finally:
            self.stderr.close()
        if error or code:
            raise RuntimeError(f'ffmpeg failed: exit={code}, error={error}')


def registry():
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    return {group: [{'name': task, 'horizon': get_task_horizon(task)}
                    for task in TASK_SET_REGISTRY[group]] for group in GROUPS}



def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('registry',), required=True)
    parser.parse_args(argv)
    print('RMH_REGISTRY:' + json.dumps(registry(), sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
