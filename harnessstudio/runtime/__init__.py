"""HarnessStudio runtime contract: the policy-facing core HarnessStudio is built on.

Ported byte-identically from Unified Harness v1.1.0 (see PROVENANCE.json): public-observation
allowlist and forbidden keys (contracts), the bounded decision core (core.UnifiedHarness), the
native/null parity guard (parity), frozen development memory (memory), inference-only modules
(modules: identity / trajectory-medoid selectors, advisory public-stasis monitor) and the
episode-start NRP instruction operator (operators). Standard library only.
"""
from .contracts import ActionChunk, Budget, PolicySpec, PublicObservation, RuntimeContract
from .core import Decision, UnifiedHarness
from .memory import FrozenMemory, MemoryCard
from .modules import IdentitySelector, MedoidSelector, ProgressMonitor
from .parity import ParityError, ParityGuard

__all__ = ["ActionChunk", "Budget", "PolicySpec", "PublicObservation", "RuntimeContract",
           "Decision", "UnifiedHarness", "FrozenMemory", "MemoryCard", "IdentitySelector",
           "MedoidSelector", "ProgressMonitor", "ParityError", "ParityGuard"]

VERSION = "1.1.0"
