"""chunktrust-ahs: ChunkTrust adaptive horizon selection, adapted reproduction (spec §7)."""
from ...core.capabilities import caps
from ...core.method import Fidelity, MethodConfig, ModuleRef
from . import modules  # noqa: F401

AHS_PARAMS = {  # identical to runs/chunktrust-beat-20261026/harnesses/chunktrust_ahs/config/ahs_reference.json
    "exec_mode": "expected_round", "expected_temp": 1.0, "horizon_alpha": 4.0,
    "horizon_candidates": [4, 8, 12, 16], "intra_cut_t": 0.25, "signal_mode": "both",
    "speed_beta": 4.0, "speed_fallback": "ehh_only", "speed_weight": 0.5, "speed_window": 20,
    "tau_window": 3, "ts_epsilon": 0.05, "ts_forget_rho": 0.99, "ts_kernel_bandwidth": 10.0,
    "ts_seed": 0, "ts_update_eta": 1.0, "ts_update_mode": "kernel_forget",
    "use_speed_uniformity": True, "z_mode": "window_rms"}


def config() -> MethodConfig:
    return MethodConfig(
        name="chunktrust-ahs", version="1", paper="arXiv:2609.39754v2", fidelity=Fidelity.FAITHFUL,
        substitutions=(), required=caps("action_chunk", "seeded_sampling", "velocity_readout", "variable_execute"),
        slots={"sampler": ModuleRef("harnessstudio.methods.chunktrust_ahs.modules:VelocityReadout"),
               "executor": ModuleRef("harnessstudio.methods.chunktrust_ahs.modules:AHSExecutor",
                                     {"min_k": 4, "per_step": False, "ahs_params": AHS_PARAMS,
                                      "velocity_dims": "all"})},
        description="Adapted reproduction of ChunkTrust-AHS (training-free AHS only).")
