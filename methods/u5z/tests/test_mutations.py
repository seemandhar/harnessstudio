"""Mutation checks (P-20261031-21): each wrong precedence variant must fail at least one synthetic precedence check,
while the real glue passes all of them (test_u5z.py). Mutants re-class a freshly built real composite."""
from __future__ import annotations

import pytest

import precedence_checks as P
from harnessstudio.core import registry
from hs_u5z.glue import U5ZActionOp


class ZWheneverNotArmed(U5ZActionOp):
    """'Z whenever U5 is not (currently) armed' -- the inadmissible variant named in C9."""
    def transform(self, chunk, obs, memory):
        u5_out = self.u5.transform(chunk, obs, memory)
        if not self.u5.armed and not self.u5.diagnostics().get("steer"):
            out = self.z.transform(chunk, obs, memory)
            self._diag = {"glue": {"executed": "z", "arm_decision": None, "z_substituted": False, "native_zero": None,
                                   "stats": dict(self.stats)}, "u5": self.u5.diagnostics(), "z": self.z.diagnostics()}
            return out
        self._diag = {"glue": {"executed": "u5", "arm_decision": self.u5.first_arm, "native_zero": True,
                               "stats": dict(self.stats)}, "u5": self.u5.diagnostics(), "z": None}
        return u5_out


class ArmingDecisionToZ(U5ZActionOp):
    """Decides by U5's arming state BEFORE this decision's U5 call (Z owns the arming decision)."""
    def transform(self, chunk, obs, memory):
        armed_before = self.u5.first_arm is not None
        u5_out = self.u5.transform(chunk, obs, memory)
        if not armed_before:
            out = self.z.transform(chunk, obs, memory)
            self._diag = {"glue": {"executed": "z", "arm_decision": None, "z_substituted": False, "native_zero": None,
                                   "stats": dict(self.stats)}, "u5": self.u5.diagnostics(), "z": self.z.diagnostics()}
            return out
        self._diag = {"glue": {"executed": "u5", "arm_decision": self.u5.first_arm, "native_zero": True,
                               "stats": dict(self.stats)}, "u5": self.u5.diagnostics(), "z": None}
        return u5_out


class ZPriority(U5ZActionOp):
    """Z executes whenever it substitutes, regardless of U5's arming."""
    def transform(self, chunk, obs, memory):
        u5_out = self.u5.transform(chunk, obs, memory)
        z_out = self.z.transform(chunk, obs, memory)
        zd = self.z.diagnostics()
        self._diag = {"glue": {"executed": "z", "arm_decision": None, "z_substituted": False, "native_zero": None,
                               "stats": dict(self.stats)}, "u5": self.u5.diagnostics(), "z": zd}
        return z_out if zd.get("z_mode") in ("tail", "hold") else u5_out


class NoZ(U5ZActionOp):
    """U5 only."""
    def transform(self, chunk, obs, memory):
        out = self.u5.transform(chunk, obs, memory)
        self._diag = {"glue": {"executed": "u5", "arm_decision": None, "stats": dict(self.stats)},
                      "u5": self.u5.diagnostics(), "z": None}
        return out


class ZFirst(U5ZActionOp):
    """Z called before U5 (order violated), otherwise the frozen rule."""
    def transform(self, chunk, obs, memory):
        if self.u5.first_arm is None:
            z_out = self.z.transform(chunk, obs, memory)
            u5_out = self.u5.transform(chunk, obs, memory)
            out = z_out if self.u5.first_arm is None else u5_out
        else:
            out = self.u5.transform(chunk, obs, memory)
        self._diag = {"glue": {"executed": "?", "arm_decision": self.u5.first_arm, "stats": dict(self.stats)},
                      "u5": self.u5.diagnostics(), "z": None}
        return out


MUTANTS = (ZWheneverNotArmed, ArmingDecisionToZ, ZPriority, NoZ, ZFirst)


def mutant_factory(cls):
    ref = registry.load("u5z-t").slots["action_op"]

    def make():
        op = ref.build()
        op.__class__ = cls
        return op
    return make


@pytest.mark.parametrize("cls", MUTANTS, ids=lambda c: c.__name__)
def test_mutant_is_caught(cls):
    failed = []
    for check in P.ALL_CHECKS:
        try:
            check(mutant_factory(cls))
        except AssertionError as exc:
            failed.append((check.__name__, str(exc)[:80]))
    assert failed, f"{cls.__name__} passed every precedence check"
    print(cls.__name__, "caught by", failed)
