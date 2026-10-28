"""HarnessStudio exception types. Episode classification is in spec §10."""


class HarnessStudioError(Exception):
    """Base class."""


class RegistrationError(HarnessStudioError, ValueError):
    """A method config violates the registration rules (spec §4.3)."""


class CapabilityError(HarnessStudioError, ValueError):
    """A method needs a capability the policy adapter does not provide."""


class ManifestMismatch(HarnessStudioError, RuntimeError):
    """Server and worker were built from different frozen method manifests."""


class LeakageError(HarnessStudioError, RuntimeError):
    """Forbidden data reached, or would reach, a module."""


class MethodContractError(HarnessStudioError, RuntimeError):
    """A module violated its declared contract; the episode is invalid_method_defect."""
