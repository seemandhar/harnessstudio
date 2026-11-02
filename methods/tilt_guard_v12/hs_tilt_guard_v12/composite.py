"""tilt-guard v1.2 composition: Z -> B (no G).

action_op.py and geometry.py are byte-identical copies of tilt-guard v1.1 (hs_tilt_guard_v11, frozen). v1.2 changes only
the composition: G (hs_band_guard) is removed. This subclass forces guard=False and refuses any guard parameter, so the
G branch of the shared op is unreachable and hs_band_guard is never imported (amendment A2.1: no G-based unit enters the
Campaign after wave 1 showed G outcome-inert on hard48). Every output equals TiltGuardActionOp(guard=False, ...) bitwise.
"""
from __future__ import annotations

from .action_op import TiltGuardActionOp

_REFUSED = ("guard_factory", "guard_params")


class TiltGuardV12ActionOp(TiltGuardActionOp):
    def __init__(self, guard: bool = False, **params) -> None:
        if guard:
            raise ValueError("tilt-guard v1.2 has no G component (amendment A2.1); guard must be False")
        bad = [k for k in _REFUSED if k in params]
        if bad:
            raise ValueError(f"tilt-guard v1.2 refuses guard parameters {bad}")
        super().__init__(guard=False, **params)
