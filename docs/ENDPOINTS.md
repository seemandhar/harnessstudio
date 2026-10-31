# Endpoints available to HarnessStudio methods

Every model API and local service a method package can use, how to pin it, and what is allowed. The
machine-readable source is `harnessstudio/endpoints.yaml`; this page explains it. Last live check: 2026-10-31.

```bash
.venv/bin/python -m harnessstudio.endpoints list                          # what exists
.venv/bin/python -m harnessstudio.endpoints check                         # live: /models + local /health
.venv/bin/python -m harnessstudio.endpoints pin nrp qwen3-vl --name planner --original-model "Qwen3-VL-32B"
```
`pin` prints a `paper.yaml` `operators:` entry (OperatorPin fields) with the provider's recommended settings.
From Python: `harnessstudio.endpoints.operator_entry(provider, model, name, ...)`.

## Rules (binding)
- **NRP is the default** for any LLM/VLM inside a harness. OpenAI and Claude are explicit opt-ins: pin the model,
  record call/token budgets in `paper.yaml`, disclose the substitution. No silent fallback between models.
- **Astra (`gpt-6-astra`)**: confirm-before-use (user rule 2026-10-31, was banned 2026-10-20). Ask the user and get
  an explicit yes for that specific use (model, purpose, run scope, expected calls/cost); pass the record as
  `user_confirmation` when pinning. The registry refuses an Astra pin without it.
- **Credentials** only through environment variables named in `api_key_env` (loaded from the private `.env` by
  the launching shell: `set -a; . <private .env>; set +a`). Never write, print or log a key.
- A paper's original model that we cannot or may not use is replaced by the closest registered model
  (`closest_for` in the YAML) and recorded as `original_model`; the method is then `fidelity: adapted`.

## Model providers

| Provider | Base URL / preset | Key variable | Status |
|---|---|---|---|
| NRP (open-weight) | `https://ellm.nrp-nautilus.io/v1`, preset `nrp` | `NRP_API_KEY` | default; 17 models listed |
| OpenAI | `https://api.openai.com/v1`, preset `openai` | `OPENAI_API_KEY` | opt-in; gpt-6.x, gpt-5.x, gpt-4.x, o3/o4, embeddings |
| Claude (Anthropic) | `claude -p` CLI (Claude Code 2.1.316) | CLI login; no `ANTHROPIC_API_KEY` set | opt-in; **not wired yet** (see below) |
| LiteLLM gateways :4141 / :4142 | Anthropic-format proxies for Claude Code itself | - | not harness providers; down |

**NRP models** (all OpenAI-compatible `/chat/completions`):
- `qwen3-vl` (image + text; served as Qwen/Qwen3.8-Flash-Next-FP8). Thinks by default: pin
  `extra_body: {chat_template_kwargs: {enable_thinking: false}}` (the registry adds it). Verified for image calls,
  JSON plans and 0-1000 boxes. 5-40 s per call, sometimes 100-280 s; 8-12 concurrent streams worked.
  Used by harness-vla, roboharness, llm-subtask-planner. Closest to Qwen3-VL / Qwen2.5-VL planners.
- `gemma-vl` (vision by name, probe first); text: `qwen3`, `qwen3-small`, `gpt-oss`, `glm-5`, `kimi`, `minimax`,
  `minimax-m2`, `deepseek-v4-flash`, `gemma`, `gemma-small`, `gemma-small-e4b`, `gemma4-small`, `gemma4-12b`;
  embeddings `qwen3-embedding` (`/embeddings`, probe first).

**OpenAI models**: `gpt-6.1-sol` (image + text, reasoning; verified; used by harness-vla-sol*, fast-harness),
`gpt-6.1-terra`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-4.1`, `gpt-4o`,
`gpt-4o-mini`, `o3`, `o4-mini`, embeddings `text-embedding-3-large/small`. Reasoning models take only the default
temperature (`temperature: null`, the registry sets it) and `max_completion_tokens` (the LLM tool switches
automatically). `gpt-6-astra` is listed under `confirm_required`.

**Claude**: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`, `claude-fable-5-1` via the headless CLI.
HarnessStudio's LLM tool speaks only the OpenAI chat format today, so a method cannot call Claude through
`ctx.tools["llm"]` yet. Adding a `claude_cli` operator (pinned `--model`, no tools, timeout, usage parsing,
forbidden/confirm checks, then images) is a framework extension: follow the gated route in the porting skill's
EVOLUTION.md. Never change Claude Code's own authentication.

## Local services

| Service | Address | GPU | Start | Used by |
|---|---|---|---|---|
| pi0.5 LIBERO (seeded T1), profile `pi05-libero-hs` | `ws://127.0.0.1:8106` | 6 | see `endpoints.yaml` `start` | most LIBERO methods |
| pi0.5 LIBERO stock, profile `pi05-libero-stock` | `ws://127.0.0.1:8101` | 4 | `hs2-gpu/start_server.sh` | parity checks |
| GR00T N1.5 RoboCasa365, profile `gr00t-n15-robocasa365` | `ws://127.0.0.1:8105` | 5 | see YAML | RoboCasa365 methods |
| OpenVLA LIBERO, profile `openvla-libero-hs` | `ws://127.0.0.1:8115` | 5 | `hs2-gpu/start_openvla.sh` | scale |
| SAM3 (RPent server, best mask) | `http://127.0.0.1:8117` `POST /call` | 7 | see YAML | harness-vla |
| RoboHarness tools (SAM3 all masks, overlay, inpaint, CLIP+MiniLM encoder) | `http://127.0.0.1:8131` | 4 | `hs2-gpu/roboharness/start_tool_server.sh` | roboharness |

Benchmarks: Docker images `ghcr.io/allenai/vla-evaluation-harness/libero:latest` and `.../libero-pro:latest`
(LIBERO-Pro data in `hs2-gpu/libero_pro/hf_libero_pro`); RoboCasa365 in-process (`--no-docker`).
Model assets (SAM3 checkpoint, CLIP ViT-L/14 and MiniLM snapshots, openpi cache) are listed under `assets`; the
HF cache has no `refs/main`, so load by snapshot directory. GPUs 0-3 belong to other users: use 4-7.

## Adding an endpoint
Probe it (`endpoints check` shows models a provider lists but the registry does not), test the actual role on one
real input, then add it to `endpoints.yaml` with `modalities`, `verified` dates and notes, and run
`.venv/bin/python -m pytest -q tests/test_endpoints.py`. Local services need `url`, `health` or protocol, `start`.
