# EXT anthropic-operator (2026-10-31)

- **Need:** the user added an Anthropic API key so harness plugins can use Claude. The LLM tool spoke only the
  OpenAI chat format to NRP and OpenAI.
- **Change:** preset `anthropic` (`https://api.anthropic.com/v1`) and the `anthropic-version: 2023-06-01` header for
  that host (its `/models` returns 400 without it). Chat goes through Anthropic's OpenAI-compatible endpoint, so
  the operator, budgets, traces and parsers are unchanged. Registry: provider `anthropic` with 14 listed models,
  `temperature: null` for Claude 5.x, key variable `HS_ANTHROPIC_API_KEY`; `endpoints check` sends the header too.
- **Verified live** through `harnessstudio.tools.llm`: GET /models, text, system+JSON, a 256x256 image
  (claude-sonnet-5-5, claude-haiku-5-5).
- **Compatibility:** no other host gets the header (unit test); all method fingerprints unchanged (no card uses
  Claude); `scripts/regress.py` on this worktree: see regress.json.
