# Cross-Provider Evaluation — Quick Start

This directory contains scripts for the **9-model cross-provider evaluation matrix** added to address VLDB 2026 reviewer concerns R1 O1 / R3 O1 (evaluation limited to two Claude models).

The matrix is **3 providers × 3 tiers**:

| Tier | Anthropic | OpenAI | Google |
|---|---|---|---|
| Flagship | Opus 4.7 | GPT-5.2 | Gemini 2.5 Pro |
| Mid | Sonnet 4.6 | GPT-5 | Gemini 2.5 Flash |
| Small | Haiku 4.5 | GPT-5 mini | Gemini 2.5 Flash-Lite |

## Default transport: public CLI (no API keys)

The framework invokes inference via each provider's **public CLI** under your personal subscription:

| Provider | Binary | Auth |
|---|---|---|
| Anthropic | `claude` (Claude Code) | `claude login` (Pro/Max subscription or API key, handled by CLI) |
| OpenAI | `codex` (Codex CLI) | `codex login` (ChatGPT Plus/Pro account or API key, handled by CLI) |
| Google | `gemini` (Gemini CLI) | `gemini auth login` (personal Google account or API key, handled by CLI) |

The eval scripts never see an API key. Auth is entirely the CLI's responsibility.

## 1. Install the public CLIs

On your target machine:

```bash
# Anthropic Claude Code
npm install -g @anthropic-ai/claude-code
# OR: curl -fsSL https://claude.ai/install.sh | bash
claude login

# OpenAI Codex CLI
brew install --cask codex
# OR: npm install -g @openai/codex
codex login

# Google Gemini CLI
npm install -g @google/gemini-cli
# OR: brew install gemini-cli
gemini auth login
```

Each CLI handles its own auth. If you have Claude Pro/Max, ChatGPT Plus/Pro, and a personal Google account, you can run the full matrix at zero marginal cost.

## 2. Diagnose CLI setup

```bash
python scripts/doctor_cli.py
```

This probes each binary and reports which path resolved, what `--version` says, and (with `--probe-call`) whether a trivial round-trip succeeds.

## 3. Smoke-test all 9 models

```bash
python scripts/smoke_test_cross_provider.py
```

Calls each model with a trivial `SELECT 1` prompt and prints a pass/fail table. Models whose CLI is unavailable on the host are skipped (not failed). Output: `results/smoke/smoke_{timestamp}.jsonl`.

Filter to a tier or specific models:

```bash
python scripts/smoke_test_cross_provider.py --tier flagship
python scripts/smoke_test_cross_provider.py --models anthropic-opus-4-7,openai-gpt-5-2
```

## 4. Run the full cross-provider matrix

Prerequisites:
- ClickHouse running locally (`localhost:9000`) with `analytics`, `clickbench`, `ssb` databases loaded

Then:

```bash
python scripts/run_cross_provider_evaluation.py
```

The runner uses the **optimal config from the VLDB 2026 paper** (Table 8):
- `format = MARKDOWN`
- `scope = RELEVANT_SUBSET`
- `metadata = DESCRIPTIONS`
- `examples = DYNAMIC_FEW_SHOT`
- `prompt_version = FULL`

It runs every (model × dataset) pair. Output JSONL: `results/cross_provider/{model_key}/{dataset}.jsonl`. Aggregate matrix: `results/cross_provider/aggregate_matrix.csv`.

### Common flags

```bash
# Smoke run (5 queries per dataset, all 9 models)
python scripts/run_cross_provider_evaluation.py --max-queries 5

# One tier only
python scripts/run_cross_provider_evaluation.py --tier flagship

# One model
python scripts/run_cross_provider_evaluation.py --models anthropic-opus-4-7

# One dataset
python scripts/run_cross_provider_evaluation.py --datasets custom_analytics

# Higher concurrency (per-model — watch CLI subscription rate limits)
python scripts/run_cross_provider_evaluation.py --concurrency 4

# Skip ClickHouse execution (just generate SQL)
python scripts/run_cross_provider_evaluation.py --no-execute

# Compose the prompt pipeline without calling any model
python scripts/run_cross_provider_evaluation.py --dry-run

# Re-run even queries already in checkpoint JSONL
python scripts/run_cross_provider_evaluation.py --force
```

### Resumability

The runner is resumable. If you interrupt it, re-running with the same args will skip queries already present in the per-model JSONL files. Use `--force` to override.

### Subscription rate limits

CLI mode uses your interactive-tier subscription quotas:
- Claude Pro / Max — message-based hourly limits
- ChatGPT Plus / Pro — message-based hourly limits
- Gemini personal — request-per-day caps on AI Studio

For a 2,000-call evaluation matrix, expect to spread the run over multiple sessions. Lower `--concurrency` to 2-4 to avoid throttling.

If you have API keys and prefer batch throughput, switch any model to `transport: sdk` in `config/cross_provider_models.yaml` — the framework will fall through to the SDK adapter using the appropriate `*_API_KEY` env var.

## 5. Falling back to SDK transport

Set `transport: sdk` on a model entry to use the Python SDK + API key path. Required env vars:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AIza...
```

Or set `defaults.transport: sdk` to switch all models at once.

## 6. Adding a model

Edit `config/cross_provider_models.yaml`. Each entry needs:
- `provider`: `anthropic` | `openai` | `google`
- `model_id`: the exact id the provider's CLI accepts (`claude --models`, `codex models`, `gemini models list`)
- `display_name`, `tier`, context/output limits, costs

The factory in `framework/llm/factory.py` auto-routes new entries.

## 7. Adding a provider

1. Create `framework/llm/<provider>_cli_caller.py` subclassing `CLICallerBase` from `cli_caller.py`. Set `BINARY`, `ALLOWED_PATHS`, `PROVIDER`. Implement `_build_invocation` and `_parse_output`.
2. Register in `framework/llm/factory.py` under `_DISPATCH`.
3. Add models for that provider in the yaml.

## File map

| File | Purpose |
|---|---|
| `config/cross_provider_models.yaml` | 9-model matrix definition |
| `framework/llm/__init__.py` | Package exports |
| `framework/llm/base.py` | `LLMResponse`, `LLMCallerBase`, `extract_sql` |
| `framework/llm/cli_caller.py` | CLI base + Claude/Codex/Gemini wrappers (default transport) |
| `framework/llm/anthropic_caller.py` | SDK fallback for Anthropic |
| `framework/llm/openai_caller.py` | SDK fallback for OpenAI |
| `framework/llm/google_caller.py` | SDK fallback for Google |
| `framework/llm/factory.py` | Config loading + transport dispatch |
| `framework/llm_caller.py` | Original Anthropic-only caller (kept for backwards compat) |
| `scripts/doctor_cli.py` | CLI installation diagnostic |
| `scripts/setup_cross_provider.sh` | One-shot venv + Python dep install |
| `scripts/smoke_test_cross_provider.py` | Per-model connectivity test |
| `scripts/run_cross_provider_evaluation.py` | Full cross-provider matrix runner |

## Reviewer mapping

| Reviewer concern | Addressed by |
|---|---|
| R1 O1 — single model family | All 9 models, 3 providers |
| R3 O1 — limited to two Claude models | All 9 models across families |
| (R1 O1 cross-OLAP) | Phase 2 — DuckDB / Snowflake / BigQuery (not yet built) |
| (R3 O3 missing baselines) | Phase 4 — DIN-SQL / MAC-SQL / CHESS (not yet built) |

See `.claude/local/research/PLAN.md` for the full roadmap.
