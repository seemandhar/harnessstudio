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
- Items in `docs/core-todo.md`.
