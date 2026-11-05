# star: STAR (Steering Action Representations) for OpenVLA (HarnessStudio v2 method)

STAR (arXiv:2610.41009v1; code PKU-VaLuE-Lab/STAR @ 0eaa15f, pi0.5 / LIBERO-PRO only) applies a
proprioception-modulated EMA to the residual updates of the last third of a VLA's decoder layers. This package is
a re-derivation for OpenVLA-7B on SimplerEnv Google Robot, written as a K = 1 slot method whose server-side
`sampler` is the STAR residual hook. It requires the EXT openvla-residual-hook capability `residual_hooks`,
merged in d9e2e8b; see `docs/extensions/openvla-residual-hook.md`.

- **Entry points:**
  - `star`: the method.
  - `star-shadow`: the compute-matched overhead control. It computes everything and writes nothing back.
- **Fidelity:** `adapted`. The full label is `substitutions[0]` of `hs_star/paper.yaml`; choices U1-U18 are in the
  run's `audit/paper_audit.md` §6.
- **Code identity:** byte-identical to the frozen snapshot that the scored runs used (merge commit 34c17c8, snapshot
  manifest 47d097df). The card has since changed only by appended changelog strings; the fingerprints are
  unchanged.

## Results (2026-11-05; details in `docs/results/star.md`)

- **VM primary: not reproduced.** 189/756 -> 190/756 final-step, W 70 / L 69, exact McNemar p = 1.0, Newcombe
  95% CI [-2.93, +3.19] pp. The paper-implied +5.57 pp is outside the CI.
- **VA base secondary (descriptive):** 75/216 -> 71/216, W 23 / L 27, p = 0.672. Our VA baseline matches the
  paper's Native row on 5/5 tasks.
- **Mechanism active:** 740/756 and 216/216 pairs diverge. star-shadow is action-identical (M 756 / 216, X 0).
- **Cost:** 41.1 GPU-h; 0 paid calls. Run: `runs/star-port-20261104` (RoboMetaHarness-claude).
