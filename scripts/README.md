# Cross-Provider Evaluation — Quick Start

This directory contains scripts for the **9-model cross-provider evaluation matrix** added to address VLDB 2026 reviewer concerns R1 O1 / R3 O1 (single-model-family limitation).

The matrix is **3 providers × 3 tiers**:

| Tier | Anthropic | OpenAI | Google |
|---|---|---|---|
| Flagship | Opus 4.7 | GPT-5.2 | Gemini 2.5 Pro |
| Mid | Sonnet 4.6 | GPT-5 | Gemini 2.5 Flash |
| Small | Haiku 4.5 | GPT-5 mini | Gemini 2.5 Flash-Lite |

## 1. Install dependencies

```bash
bash scripts/setup_cross_provider.sh
```

This:
- Creates / activates `.venv` (or reuses parent DataPup `.venv`)
- Installs `anthropic`, `openai`, `google-genai`, `pyyaml`, `clickhouse-driver`
- Verifies API key environment variables are set
- Refuses to run if Meta-CLI environment variables (`ANTHROPIC_BASE_URL`, etc.) are detected

## 2. Set personal API keys

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # console.anthropic.com/settings/keys
export OPENAI_API_KEY=sk-...               # platform.openai.com/api-keys
export GOOGLE_API_KEY=AIza...              # aistudio.google.com/apikey
```

**Use personal keys, NOT Meta-issued keys.** The paper bylines us as Independent Researchers; using employer-funded inference creates IP and EB-1A risk (see `.claude/local/research/A3-meta-cli-policy.md`).

## 3. Smoke-test all 9 models

```bash
.venv/bin/python scripts/smoke_test_cross_provider.py
```

Calls each model with a trivial `SELECT 1` prompt and prints a pass/fail table. Models with missing API keys are skipped (not failed). Output: `results/smoke/smoke_{timestamp}.jsonl`.

Filter to a tier or specific models:

```bash
.venv/bin/python scripts/smoke_test_cross_provider.py --tier flagship
.venv/bin/python scripts/smoke_test_cross_provider.py --models anthropic-opus-4-7,openai-gpt-5-2
```

## 4. Run the full cross-provider matrix

Prerequisites:
- ClickHouse running locally (`localhost:9000`) with `analytics`, `clickbench`, `ssb` databases loaded
  - Re-load via `bash load_clickbench.sh` and `bash load_ssb.sh` if needed (these scripts live in the parent DataPup repo)

Then:

```bash
.venv/bin/python scripts/run_cross_provider_evaluation.py
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
.venv/bin/python scripts/run_cross_provider_evaluation.py --max-queries 5

# One tier only
.venv/bin/python scripts/run_cross_provider_evaluation.py --tier flagship

# One model
.venv/bin/python scripts/run_cross_provider_evaluation.py --models anthropic-opus-4-7

# One dataset
.venv/bin/python scripts/run_cross_provider_evaluation.py --datasets custom_analytics

# Higher concurrency (per-model, watch rate limits)
.venv/bin/python scripts/run_cross_provider_evaluation.py --concurrency 8

# Skip ClickHouse execution (just generate SQL — useful when CH is unavailable)
.venv/bin/python scripts/run_cross_provider_evaluation.py --no-execute

# Compose the prompt pipeline without calling any model
.venv/bin/python scripts/run_cross_provider_evaluation.py --dry-run

# Re-run even queries already in checkpoint JSONL
.venv/bin/python scripts/run_cross_provider_evaluation.py --force
```

### Resumability

The runner is resumable. If you interrupt it, re-running with the same args will skip queries already present in the per-model JSONL files. Use `--force` to override.

### Estimated cost

Full 9-model matrix on all 206 queries: **~$224** at frontier-tier pricing (Opus 4.7 is the dominant cost contributor). With Tier 1+2 baselines layered on (DIN-SQL + MAC-SQL + CHESS), ~$675. Per `feedback_datapup_no_budget.md`, no budget cap applies.

## 5. Adding a model

Edit `config/cross_provider_models.yaml`. Each entry needs:
- `provider`: `anthropic` | `openai` | `google`
- `model_id`: provider's canonical model id (pinned)
- `display_name`, `tier`, context/output limits, costs

The factory in `framework/llm/factory.py` auto-routes new entries to the right provider class.

## 6. Adding a provider

1. Create `framework/llm/<provider>_caller.py` subclassing `LLMCallerBase`. Implement `call()` with retry + token capture.
2. Set `PROVIDER = "<name>"` on the class.
3. Register in `framework/llm/factory.py` under `_PROVIDER_CLASSES`.
4. Add models for that provider in the yaml.

## File map

| File | Purpose |
|---|---|
| `config/cross_provider_models.yaml` | 9-model matrix definition |
| `framework/llm/__init__.py` | Package exports |
| `framework/llm/base.py` | `LLMResponse`, `LLMCallerBase`, `extract_sql` |
| `framework/llm/anthropic_caller.py` | Anthropic SDK wrapper |
| `framework/llm/openai_caller.py` | OpenAI SDK wrapper |
| `framework/llm/google_caller.py` | google-genai SDK wrapper |
| `framework/llm/factory.py` | Config loading + provider dispatch |
| `framework/llm_caller.py` | Original Anthropic-only caller (kept for backwards compat) |
| `scripts/setup_cross_provider.sh` | One-shot setup + dep install + key check |
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
