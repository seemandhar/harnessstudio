"""Read GR00T N1.5 flow-matching sampler velocities without changing actions.

The pinned Isaac-GR00T (robocasa-benchmark fork 9d7d7a9e) action head runs, for each
of T denoising steps:  pred = action_decoder(...); v = pred[:, -H:]; x += dt * v.
ChunkTrust's GR00T N1.5 overlay edits that loop to append v (``return_trace``).
Here a PyTorch forward hook on ``action_head.action_decoder`` copies the same
``pred[:, -H:]`` tensor. The hook returns None, so the module output object, every
later operation, the global torch RNG and the returned actions are unchanged.
No checkpoint, source file or preprocessing transform is modified.

``capture_states=True`` (diagnostic only, used by identity_probe.py) also records the
input sample x_tau of action_encoder so x_{tau+1} = x_tau + dt * v_tau can be checked.
"""
from __future__ import annotations

import contextlib


class VelocityTraceRecorder:
    def __init__(self, policy, *, capture_states: bool = False):
        head = policy.model.action_head
        self.head = head
        self.horizon = int(head.action_horizon)
        self.steps = int(head.num_inference_timesteps)
        self.armed = False
        self.velocities = []
        self.states = []
        self.capture_states = capture_states
        self._handles = [head.action_decoder.register_forward_hook(self._decoder_hook)]
        if capture_states:
            self._handles.append(head.action_encoder.register_forward_pre_hook(self._encoder_pre_hook))

    def _decoder_hook(self, module, inputs, output):
        if self.armed:
            # Same slice as the sampler: pred[:, -self.action_horizon:]
            self.velocities.append(output[:, -self.horizon:].detach().clone())
        return None

    def _encoder_pre_hook(self, module, inputs):
        if self.armed:
            self.states.append(inputs[0].detach().clone())
        return None

    @contextlib.contextmanager
    def recording(self):
        if self.armed:
            raise RuntimeError('nested velocity recording')
        self.velocities, self.states = [], []
        self.armed = True
        try:
            yield self
        finally:
            self.armed = False

    def take(self):
        """Return velocity [T, H, D] float32 numpy for batch element 0 (+ raw tensors)."""
        import torch
        if len(self.velocities) != self.steps:
            raise RuntimeError(f'expected {self.steps} denoising velocities, got {len(self.velocities)}')
        stacked = torch.stack(self.velocities, dim=1)  # [B, T, H, D] like overlay trace_v
        if stacked.shape[0] != 1 or stacked.shape[2] != self.horizon:
            raise RuntimeError('unexpected velocity trace shape ' + str(tuple(stacked.shape)))
        velocity = stacked[0].float().cpu().numpy()
        ds = [int((t / float(self.steps)) * self.head.num_timestep_buckets) for t in range(self.steps)]
        return velocity, ds

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
