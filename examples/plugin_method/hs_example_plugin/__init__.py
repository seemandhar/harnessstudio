"""Example third-party HarnessStudio method: draw 2 seeded samples, always execute sample 0.

It exists to show the plugin route (pip install -> entry point -> rh list / rh verify) and is
behaviourally identical to the native policy. Install with:
    harnessstudio/.venv/bin/python -m pip install -e harnessstudio/examples/plugin_method
"""
from harnessstudio.core.capabilities import caps
from harnessstudio.core.method import Fidelity, MethodConfig, ModuleRef


class Pool:
    def __init__(self, num_samples=2):
        self.num_samples = num_samples
        self.identity = {"module": "example_pool", "num_samples": num_samples}


class First:
    identity = {"module": "example_first"}

    def select(self, chunks, extras, obs):
        return 0


def config() -> MethodConfig:
    return MethodConfig(name="first-sample", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        substitutions=(), required=caps("action_chunk", "seeded_sampling", "isolated_streams"),
                        slots={"proposer": ModuleRef("hs_example_plugin:Pool", {"num_samples": 2}),
                               "selector": ModuleRef("hs_example_plugin:First")},
                        description="Plugin demo: two samples, always the first (native-equivalent).")
