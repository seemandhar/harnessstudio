# EXT astra-confirm (2026-10-31)

- **Need:** the user revised the workspace rule from "never use Astra" (2026-10-20) to "unban astra - but ask
  before use confirm" (2026-10-31). The core refused every Astra model; it must instead refuse unconfirmed ones.
- **Change:** `OperatorPin.user_confirmation` (record of the user's explicit yes: who, when, scope). An Astra pin
  without it raises `RegistrationError`; the LLM tool refuses unconfirmed Astra models at construction and when a
  provider answers with an Astra model the pin did not confirm. Empty confirmations are omitted from
  `MethodConfig.to_json`, so every existing method keeps its fingerprint.
- **Compatibility:** fingerprints of all 15 installed methods identical before/after (checked); no method uses
  Astra; `scripts/regress.py` on this worktree: ok (46 unit tests, tier 0 + tier 1 of 15 methods). The gate was
  also checked to fail on a sabotaged copy (pytest rc 1, 15/15 methods failing).
- **Also fixed here:** `scripts/regress.py` interpreter resolution for worktrees (HS_PYTHON / own .venv /
  canonical v2 .venv) and PASS detection over the full tier output (a 6-line tail missed the PASS line).
- **Review:** authorized directly by the user's instruction; merged by the main session after the RoboHarness
  evaluation that imported the core finished (`regress.py --inflight` empty).
