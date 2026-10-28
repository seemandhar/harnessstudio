"""N1 common random numbers for the GR00T flow-matching initial noise.

The pinned action head draws its initial noise with ``torch.randn`` from the default
(global) generator. Under N1 the server runs each policy call inside
``torch.random.fork_rng`` after ``torch.manual_seed(seed)``, where

    seed = derive_seed(namespace, case_id, stream='primary', ordinal)

is identical in every arm for the same (case, policy-call ordinal). The forked
context restores the global generators afterwards, so the stream is dedicated to that
call. The marginal noise distribution (standard normal), the weights and every other
computation are unchanged. Candidate or null-overhead randomness must use other
``stream``/``purpose`` names so it never consumes the primary stream.
"""
from __future__ import annotations

import hashlib

PRIMARY = 'primary'
SEED_BITS = 63


def derive_seed(namespace: str, case_id: str, stream: str, ordinal: int) -> int:
    if not namespace or not case_id or not stream or type(ordinal) is not int or ordinal < 0:
        raise ValueError('CRN seed needs namespace, case, stream and a nonnegative ordinal')
    payload = '\x1f'.join(('chunktrust-crn-v1', namespace, case_id, stream, str(ordinal))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') >> (64 - SEED_BITS)


def policy_noise_seed(namespace, case_id, ordinal, stream=PRIMARY):
    return derive_seed(namespace, case_id, stream, ordinal)


def aux_generator(namespace, case_id, purpose):
    """Separate NumPy generator for candidate/null-overhead computations (never 'primary')."""
    import numpy as np
    if purpose == PRIMARY:
        raise ValueError('the primary stream is reserved for the policy noise')
    return np.random.default_rng(derive_seed(namespace, case_id, 'aux:' + purpose, 0))


def seeded_call(fn, seed):
    """Run fn() with a dedicated torch stream seeded by ``seed``; global RNG restored after."""
    import torch
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return fn()


class NoiseSchedule:
    """Client-side ordinal counter; the evaluator owns case identity, never the policy."""
    def __init__(self, namespace, case_id, enabled):
        self.namespace, self.case_id, self.enabled = namespace, case_id, bool(enabled)
        self.ordinal = 0
        self.issued = []

    def next_seed(self):
        if not self.enabled:
            return None
        seed = policy_noise_seed(self.namespace, self.case_id, self.ordinal)
        self.issued.append(seed)
        self.ordinal += 1
        return seed
