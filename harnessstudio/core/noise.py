"""Protocol N1 common random numbers (spec §5.3).

The tag and derivation are identical to ChunkTrust v2.2 crn.py, so sample 0 reproduces the
noise of the chunktrust-ahs-v2_2 / chunktrust-gr00t-base16-v2_2 arms for the same namespace.
The worker owns this schedule; modules never see seeds or case ids.
"""
from __future__ import annotations

import hashlib

TAG = "chunktrust-crn-v1"
SEED_BITS = 63
PRIMARY = "primary"


def derive_seed(namespace: str, case_id: str, stream: str, ordinal: int) -> int:
    if not namespace or not case_id or not stream or type(ordinal) is not int or ordinal < 0:
        raise ValueError("CRN seed needs namespace, case, stream and a nonnegative ordinal")
    payload = "\x1f".join((TAG, namespace, case_id, stream, str(ordinal))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> (64 - SEED_BITS)


class NoiseSchedule:
    def __init__(self, namespace: str, case_id: str):
        if not namespace or not case_id:
            raise ValueError("NoiseSchedule needs a namespace and a case id")
        self.namespace, self.case_id = namespace, case_id

    @staticmethod
    def stream_names(k: int) -> list:
        return [PRIMARY] + [f"alt-{i}" for i in range(1, k)]

    def seeds(self, decision_ordinal: int, k: int) -> list:
        if type(decision_ordinal) is not int or decision_ordinal < 0 or type(k) is not int or k < 1:
            raise ValueError("seeds() needs a nonnegative decision ordinal and k >= 1")
        return [derive_seed(self.namespace, self.case_id, stream, decision_ordinal)
                for stream in self.stream_names(k)]
