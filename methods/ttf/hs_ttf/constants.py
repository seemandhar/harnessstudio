"""TTF-VLA constants (torch-free: the harness process imports this module for tier 0 and config building).

Source: github.com/PKU-XLab/TTF-VLA @ afbd6bf (MIT); paper arXiv:2508.19257v3.
"""
N_PATCHES = 256          # 16 x 16 patches of 14 px on a 224 px image (both featurizers)
DINO_DIM = 1024          # modeling_prismatic.py:229-231 split [1024, 1152]
SIGLIP_DIM = 1152

# The paper's OpenVLA constants: "OpenVLA employs consistent parameters: keyframe interval K=3, attention top-k=70,
# pixel threshold=0.03, and text-to-vision attention" (v3, Sec. 4.3 "Configuration Strategy"); README.md:91-100.
# Release defaults (run_libero_eval_aptube.py:97-112) for the rest. fusion_mode "hybrid" = the paper's
# "Pixel-Attention TTF" (Table 3; Eq. 4); the README's "attention_guided" is rejected by the code (F1).
PAPER = {
    "fusion_mode": "hybrid",
    "keyframe_interval": 3,
    "patch_diff_threshold": 0.03,
    "attention_top_k": 70,
    "attention_layer_id": 15,
    "attention_mode": "text",
    "use_multi_layer": False,
}
MULTI_LAYER_IDS = (10, 15, 20, 25)   # aptube_manager.py:247

# aptube_manager.py:250-255 (positions in the multimodal sequence: BOS, 256 patches, text, actions)
V_TOKEN_START, V_TOKEN_END = 1, 257
T_TOKEN_START, T_TOKEN_END = 257, 292
A_TOKEN_START = T_TOKEN_END
GRAY = (0.299, 0.587, 0.114)          # Eq. 5, aptube_manager.py:113
