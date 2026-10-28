"""Small inference-only modules. Enabling a module is an experimental choice."""
from collections import deque
import math


class IdentitySelector:
    required_capabilities = frozenset()
    identity = {"module": "identity", "version": 1}

    def select(self, chunks, execute_steps):
        return 0


class MedoidSelector:
    """Select an existing proposal; never average incompatible action modes.

    A generic trajectory-consensus ablation, not an A3/MG-Select reproduction.
    Action dimensions require explicit fixed scales, selected before validation.
    """
    required_capabilities = frozenset({"independent_sampling", "action_geometry"})

    def __init__(self, scales):
        self.scales = tuple(scales)
        if not self.scales or any(not math.isfinite(v) or v <= 0 for v in self.scales):
            raise ValueError("Action geometry requires fixed positive scales")
        self.identity = {"module": "trajectory_medoid", "version": 1, "scales": self.scales}

    def select(self, chunks, execute_steps):
        if any(len(c.actions[0]) != len(self.scales) for c in chunks):
            raise ValueError("Action geometry dimension mismatch")
        scores = []
        for a in chunks:
            score = sum(sum(abs(float(x)-float(y))/scale
                            for ar, br in zip(a.actions[:execute_steps], b.actions[:execute_steps])
                            for x, y, scale in zip(ar, br, self.scales)) for b in chunks)
            scores.append(score)
        return min(range(len(scores)), key=scores.__getitem__)


class ProgressMonitor:
    """Advisory public-proprioception stasis signal; never a success detector.

    A stationary robot may be correctly holding an object. No automatic reset,
    recovery, stopping, queue cancellation or extra observation follows a signal.
    """
    required_capabilities = frozenset({"proprioception"})

    def __init__(self, key, window=4, tolerance=1e-4):
        if type(window) is not int or window < 2 or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Invalid monitor configuration")
        self.key, self.window, self.tolerance = key, window, tolerance
        self.identity = {"module": "public_stasis", "version": 1, "key": key,
                         "window": window, "tolerance": tolerance}
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.window)

    def observe(self, observation):
        point = tuple(float(v) for v in observation.values[self.key])
        if not point or any(not math.isfinite(v) for v in point):
            raise ValueError("Invalid public proprioception")
        if self.history and len(point) != len(self.history[0]):
            raise ValueError("Public proprioception shape changed")
        self.history.append(point)
        stalled = len(self.history) == self.window and all(
            max(abs(x-y) for x,y in zip(point, old)) <= self.tolerance for old in self.history)
        return {"signal": "public_stasis" if stalled else "none", "advisory_only": True}
