"""Budget computed from K, min_k and horizon (spec §4.6). A policy call counts one sample."""
from __future__ import annotations

import math


def max_policy_calls(*, num_samples: int, min_k: int, horizon: int) -> int:
    for name, value in (("num_samples", num_samples), ("min_k", min_k), ("horizon", horizon)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return num_samples * math.ceil(horizon / min_k)
