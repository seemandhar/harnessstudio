"""Method/policy compatibility and protocol-parameter checks (Kinetix RTC extension, EC1/B3).

- require_method(): the single enforcement site for capabilities AND supported slots. Used by tier 1's
  'capabilities' check, `rh list` and the Kinetix runner. A policy descriptor without SUPPORTED_SLOTS
  supports every slot, so GR00T N1.5 and Pi0.5 behave exactly as before.
- require_protocol(): inference delay d / execute horizon s are protocol parameters, refused on any
  policy that does not declare Capability.ASYNC_EXECUTION.
"""
from __future__ import annotations

from .capabilities import Capability, require
from .errors import CapabilityError
from .method import SLOT_NAMES


def supported_slots(policy_descriptor) -> frozenset:
    declared = getattr(policy_descriptor, "SUPPORTED_SLOTS", None)
    return frozenset(SLOT_NAMES) if declared is None else frozenset(declared)


def require_method(method, policy_descriptor) -> None:
    require(method.required, policy_descriptor.CAPABILITIES, method=method.name, policy=policy_descriptor.NAME)
    unsupported = set(method.slots) - supported_slots(policy_descriptor)
    if unsupported:
        raise CapabilityError(f"{method.name} uses slots {sorted(unsupported)}, which {policy_descriptor.NAME} "
                              f"does not support (supported: {sorted(supported_slots(policy_descriptor))})")


def compatible(method, policy_descriptor) -> bool:
    try:
        require_method(method, policy_descriptor)
    except CapabilityError:
        return False
    return True


def require_protocol(policy_descriptor, inference_delay=None, execute_horizon=None) -> tuple:
    """Return (d, s) after validation. (None, None) means the policy's native protocol."""
    if inference_delay is None and execute_horizon is None:
        return None, None
    if Capability.ASYNC_EXECUTION not in frozenset(policy_descriptor.CAPABILITIES):
        raise CapabilityError(f"inference delay / execute horizon are protocol parameters that "
                              f"{policy_descriptor.NAME} does not support")
    d = 0 if inference_delay is None else inference_delay
    s = max(d, 1) if execute_horizon is None else execute_horizon
    chunk = policy_descriptor.CHUNK_LEN
    if type(d) is not int or type(s) is not int or not (0 <= d <= s and 1 <= s <= chunk):
        raise CapabilityError(f"need integers 0 <= d <= s and 1 <= s <= {chunk}; got d={d!r}, s={s!r}")
    return d, s
