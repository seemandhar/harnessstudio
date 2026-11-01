"""RTCF ("Retrieve in Time, Correct in Frequency", arXiv:2608.04527v2; withdrawn by the authors in v3) as a
HarnessStudio v2 slot method: one action_op (PMA retrieval over a frozen success memory + bounded low-frequency
DCT residual) and a development stage that builds the memory. No code was released; the port follows Eq. 5-11.

Every constant the paper does not report is a frozen parameter here (see paper.yaml substitutions and the
conversion plan runs/rtcf-20261031/port/conversion-plan.md section 3). Changing one changes the method
fingerprint, so prepared memory and evaluation must use the same values.

Pure-function API for parity checks: hs_rtcf.spectral.rtcf_correct, hs_rtcf.spectral.coefficients,
hs_rtcf.pma.{local_cost, pma_init, pma_step, pma_select, frame_nn_select}, hs_rtcf.pma.ProgressiveMemoryAlignment.
"""
from __future__ import annotations

from pathlib import Path

from harnessstudio.core.card import config_from_card
from harnessstudio.core.method import ModuleRef

NORM_STATS = "assets/pi0_fast_libero_norm_stats.json"
NORM_STATS_SHA256 = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"

# Frozen a-priori values (conversion plan section 3). lambda0 = 0.1 is the paper's; the rest are not reported.
PARAMS = {
    "image_key": "agentview",
    "encoder_url": "http://127.0.0.1:8149",            # override with HS_RTCF_ENCODER_URL (not part of the method)
    "encoder_model": "google/siglip-so400m-patch14-224",
    "encoder_revision": "d04cf29fca7b6374f74d8bea1969314492266b5e",
    "pooling": "pooler_output",
    "encoder_weights_sha256": "30a00c26a045c84a0bd14adc94278d7f1cc941c8bd67e5fa50f7bed34c34e60d",   # checked via /health
    "norm_stats": NORM_STATS,
    "norm_stats_sha256": NORM_STATS_SHA256,
    "norm_mode": "zscore",      # MUST equal the policy server (pi0_fast_libero was trained with z-score; serve with
                                # use_quantile_norm=False, repair/diagnosis.md); a mismatch raises (token-grid check)
    "token_scale": 10.0,
    "lambda0": 0.1,
    "f_cut": 3,
    "delta_max": 1.0,
    "v_max": 2,
    "gamma": 0.05,
    "read_offset": 0,
    "motion": ["position", "rotation"],
    "action_limit": 1.0,
    "retrieval": "pma",
    "correction": "spectral",
    "store_png": True,
}
PREPARE = {"only_success": True, "norm_stats": NORM_STATS, "norm_stats_sha256": NORM_STATS_SHA256, "token_scale": 10.0,
           "norm_mode": "zscore"}


def _config(card: str, **override):
    return config_from_card(Path(__file__).with_name(card), slots={
        "action_op": ModuleRef("hs_rtcf.action_op:RTCFActionOp", {**PARAMS, **override}),
    }, prepare=ModuleRef("hs_rtcf.prepare:Preparer", PREPARE))


def config():
    return _config("paper.yaml")


def config_framenn():
    """Ablation 'Frame-NN' (w/o history alignment): single-frame nearest neighbour (rule inferred)."""
    return _config("paper_framenn.yaml", retrieval="frame_nn")


def config_timedomain():
    """Ablation 'Time-Domain' (w/o spectral correction): clipped residual on time-domain actions (rule inferred)."""
    return _config("paper_timedomain.yaml", correction="time_domain")
