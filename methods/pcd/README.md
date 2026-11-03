# pcd: Policy Contrastive Decoding for OpenVLA (HarnessStudio v2 method)

PCD (Wu et al., arXiv:2505.13255, ICLR 2026; code Koorye/PCD @ 31b027e) contrasts OpenVLA's action-token logits on
the real frame with those on a counterfactual frame in which the instruction's objects are removed. This package
is a `null: shadow` controller on the EXT openvla-contrast `logits` readout. That extension is merged into harnessstudio
`main` in 4f93662 (built on branch `ext2/openvla-contrast` @ adc271a), and EXT simpler-seeded in ecc9b68; see
`docs/extensions/openvla-contrast.md`.

## Results (2026-11-03; details in `docs/results/pcd.md`)

**Setting A** is the authors' seeded SIMPLER protocol with OpenVLA-7B base. Scope: **subset only: 2 of 9 tasks, chosen
for the largest reported gains; 64 of 2700 paired trials**.
- **Result:** baseline **22/64** -> PCD **36/64**, W 17 / L 3, exact two-sided McNemar p = **0.00258**, Newcombe hybrid
  (uncorrected phi) 95% CI **[+0.090, +0.335]**. The paper's pooled +0.266 lies inside the CI.
- **Label:** **"gain reproduced in direction; magnitude compatible"**.
- **Null:** M 64/0/0. Mechanism inactive in 0/64 episodes.
- **Fidelity label:** **adapted** (float32 action wire [B7 violated]; HarnessStudio transport; env identity; SAM2 _C
  unresolved).
- **Point estimate:** +21.9 pp, below the paper's +26.6. The interval is wide, and the tasks were selected for their
  large reported gains.

**Descriptive runs:**
- **Move Near**, the "predicted no-gain task": 19/32 -> 21/32, W 4 / L 2, p = 0.6875.
- **Setting B** (libero_goal transfer screen, never "reproduced"): 23/30 -> 23/30, W 3 / L 3, p = 1.0. Tasks 6 and 9
  each have one target never masked.

**Cost:** 17.58 GPU-h in total, 0 LLM calls.

## Per step
1. **Counterfactual frame.** `/step(frame, instruction)` on the perception service (`service/`, its own venv and
   owner) returns the counterfactual frame ô and mask evidence. `/reset(instruction)` runs once per episode.
2. **Stream A.** `ctx.policy.sample(1, readouts=["logits"])` is the native greedy decode: the native action plus the
   raw float32 logits of all 7 positions.
3. **Stream B.** The same request, with the first public image replaced by ô. A and B are separate requests,
   never batched.
4. **Combine.** `hs_pcd.combine.pcd_code_combine` re-derives `openvla_contrast.py:40-58`:
   - positions 0-5 become `(1 + 0.8) * s - 0.8 * c` when B has 7 rows;
   - the gripper comes from A;
   - if A has fewer than 7 rows, A is replaced by zero rows;
   - if B has fewer than 7 rows, the contrast is skipped;
   - the argmax is over the full logit width.
5. **Execute.** The combined tokens are executed through the server's bin table, which is bitwise the server's own
   de-tokenization.

**Null arm.** Steps 1-4 also run in the null arm, on the shadow session and on the method's own perception session.
The framework then executes the native action. Decodes per step:

| arm | decodes per step |
|---|---|
| baseline | 1 |
| candidate | 2 |
| null | 3 |

## Deployment
| item | how |
|---|---|
| perception service | start it per `service/README.md` (evaluator, leased GPU). Set `HS_PCD_PERCEPTION_URL=http://127.0.0.1:PORT`. |
| tiles | set `HS_PCD_TILE_DIR=<run artifact dir>` (refused under `harnessstudio/`). Real \| cf PNGs every `tile_every` steps and at first detection, in `<dir>/<session>/`. |
| profile | not hard-coded. Setting B (LIBERO): `openvla-libero-logits` (from the extension). Setting A (SIMPLER): the separate simpler-seeded extension's profile (server mode `simpler_pcd_authors`). |
| parser / prompt | the perception service owns the frozen instruction parser and alias table (`parser: service`). The policy prompt is the server mode's. |
| tier 1 | uses `hs_pcd.perception.StubPerception` only on HarnessStudio's synthetic `fake` benchmark (contract `control_mode: fake`). Everywhere else a missing or failing service raises, and the episode is invalid. |

## Checks
Run these from the worktree, in the isolated test venv `runs/pcd-port-20261103/venvs/hs2-test`. That venv has
pinned third-party packages plus a `.pth` pointing to vla-eval `src`, the worktree and this directory, and an
entry-point shim. Nothing is built or installed into `harnessstudio/.venv`.

```bash
V=/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/pcd-port-20261103/venvs/hs2-test
cd /home/sejain/research/cvpr27/RoboMetaHarness-claude/hs2-ext-openvla-contrast
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $V/bin/python -B -m harnessstudio.cli verify pcd --tier 0
chmod a-w ../harnessstudio/methods/pcd/hs_pcd   # the clean-import check runs `python -I` (bytecode env ignored)
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $V/bin/python -B -m harnessstudio.cli verify pcd
chmod u+w,g+w ../harnessstudio/methods/pcd/hs_pcd
cd ../harnessstudio/methods/pcd && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $V/bin/python -B -m pytest -q -p no:cacheprovider tests
# combine parity vs the authors' code (torch 2.3.1 = the perception venv):
CUDA_VISIBLE_DEVICES= .../venvs/pcd-perception/bin/python -I -B scripts/check_parity.py --out .../runs/pcd-port-20261103/method/parity
```

The card is `hs_pcd/paper.yaml`. It covers the settings, labels and substitutions, the reported rows with
denominators, the budgets and the changelog.
