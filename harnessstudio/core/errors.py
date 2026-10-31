"""Error types. Each one says which side broke the contract."""


class HarnessError(Exception):
    """Base class."""


class RegistrationError(HarnessError):
    """A method config or paper card breaks a registration rule."""


class ContractError(HarnessError):
    """A benchmark contract is malformed or an observation does not match it."""


class PolicyError(HarnessError):
    """The policy server returned something the protocol does not allow."""


class BudgetExceeded(HarnessError):
    """A method spent its policy, LLM or tool budget. Control ends; the episode finishes natively."""


class MethodStopped(HarnessError):
    """Raised when a method acts after calling ctx.stop()."""
