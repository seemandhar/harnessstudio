# Framework extensions (gated)

One file per extension: `EXT_ID.md` with the paper mechanism, why existing slots/ctx/profiles cannot express it,
the smallest change that would, and the compatibility claim (native/identity bitwise; every installed method's
tier 0/1 unchanged; parity scripts still pass). Process (porting skill EVOLUTION.md, Loop 2): independent review of
the proposal -> worktree `git worktree add ../hs2-ext-EXT_ID -b ext2/EXT_ID main` -> tests -> `python3
scripts/regress.py --source ../hs2-ext-EXT_ID --out ../hs2-ext-EXT_ID/regress.json` (`"ok": true`) -> GPU check by
the evaluator -> independent diff review -> merge into `main` only when `scripts/regress.py --inflight` is empty.

## Open backlog
- (done 2026-10-31) Claude through the Anthropic API: `anthropic-operator.md`. Optional later: a `claude_cli`
  operator for the subscription login.
- (done 2026-10-31) Astra confirm-before-use in `OperatorPin` / the LLM tool: `astra-confirm.md`.
- (built 2026-11-01, review pending) explicit openpi normalization mode on the openpi policy server: `openpi-norm.md`.
- (done 2026-11-03, merged from `ext2/openvla-contrast` @ adc271a; GPU checks G1-G5, G7, G8 and the library gate PASS; G6 and G9 measured; run pcd-port-20261103) native-decode
  `logits` readout on a new OpenVLA server subclass for policy contrastive decoding (PCD): `openvla-contrast.md`.
- (done 2026-11-03, merged from `ext2/simpler-seeded` @ 9b2b714, stacked on `ext2/openvla-contrast`; GPU checks
  B1-B7 and S6-S10 done, run pcd-port-20261103) the PCD authors' seeded SIMPLER protocol (benchmark subclass + contract) and the OpenVLA
  server mode `simpler_pcd_authors`: `simpler-seeded.md`.
- (built 2026-11-04 on `ext2/openvla-residual-hook` from main f7dcd0f, review pending; run star-port-20261104) decoder-layer
  residual hooks on the simpler_pcd_authors OpenVLA server (capability `residual_hooks`) and the SimplerEnv official
  VM / VA-base grids with `states`: `openvla-residual-hook.md`. Not merged: main stays untouched until the PCD
  discovery run's RUN CLOSED.
- Items in `docs/core-todo.md`.
