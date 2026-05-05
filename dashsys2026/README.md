# DASHSys 2026 Artifact Bundle

This directory is the self-contained artifact bundle for the DASHSys 2026 submission:

> DataPup: Schema-Aware and Execution-Aware Text-to-SQL for Human-in-the-Loop Analytical Database Clients

## Contents

| Path | Purpose |
|---|---|
| `paper.pdf` | Final compiled submission PDF. |
| `paper.tex` | Submission LaTeX source. |
| `evaluation/benchmark/` | Custom analytics, ClickBench, and SSB query/schema artifacts. |
| `evaluation/framework/` | Prompt builder, schema linker, executor, metrics, and comparator. |
| `evaluation/run_cli_experiments.py` | CLI generation runner for Claude, Codex, and Gemini. |
| `evaluation/score_existing_cli_results.py` | Execution scoring for generated SQL. |
| `evaluation/repair_existing_cli_results.py` | One-attempt execution repair over failed SQL. |
| `evaluation/setup_duckdb.py` | Local DuckDB dataset generator. |
| `evaluation/run_duckdb_cli_experiments.py` | Focused DuckDB CLI validation runner. |
| `evaluation/analyze_strong_accept_evidence.py` | Produces the consolidated evidence tables. |
| `evaluation/steps/` | Shell entry points used for the submitted experiments. |
| `evaluation/results/` | Saved per-query outputs and aggregate summaries. |

## Reproduce DuckDB Validation

The DuckDB validation is the fastest reproducible experiment and does not require a ClickHouse server.

```bash
python3 -m venv evaluation/.venv_cli
evaluation/.venv_cli/bin/python -m pip install -r requirements.txt
evaluation/.venv_cli/bin/python evaluation/setup_duckdb.py --scale 0.1 --overwrite
bash evaluation/steps/08_duckdb_claude_validation_parallel.sh
bash evaluation/steps/09_strong_accept_evidence.sh
```

Expected summaries:

- `evaluation/results/duckdb_cli_validation/claude_cli/baseline/summary.json`
- `evaluation/results/duckdb_cli_validation/claude_cli/best/summary.json`
- `evaluation/results/strong_accept_evidence.md`

The paper reports 130 portable custom-analytics queries. In the saved run, the full-schema JSON baseline reaches 97.7% execution success and 60.8% result correctness; the revised prompt reaches 100.0% execution success and 60.8% result correctness.

## Reproduce Full CLI Validation

Prerequisites:

- Python 3.10+
- Claude CLI authenticated with `claude login`
- Codex CLI authenticated with `codex login`
- Gemini CLI authenticated with `gemini auth login`
- A local ClickHouse binary, exposed through `DATAPUP_CLICKHOUSE_BIN`

Run:

```bash
export DATAPUP_CLICKHOUSE_BIN=/path/to/clickhouse
bash evaluation/steps/00_preflight.sh
bash evaluation/steps/01_prepare_clickhouse.sh
nohup bash evaluation/steps/04_full_generation_parallel.sh \
  > evaluation/results/cli_runs_full_generation.nohup.log 2>&1 &
```

After generation finishes:

```bash
bash evaluation/steps/05_full_execute_parallel.sh
bash evaluation/steps/07_repair_existing_failed_sql.sh
bash evaluation/steps/09_strong_accept_evidence.sh
```

The full-generation and repair outputs used in the paper are already saved under:

- `evaluation/results/cli_runs_full_generation/`
- `evaluation/results/cli_runs_full_generation_validation/`
- `evaluation/results/cli_runs_repair_existing/`

## Evidence Files

| File | What it supports |
|---|---|
| `evaluation/results/headline_contrasts.md` | Main paired prompt-design comparisons. |
| `evaluation/results/fewshot_leakage_audit.json` | Few-shot leakage audit. |
| `evaluation/results/strong_accept_evidence.md` | Current DataPup vs revised prompt, semantic audit, and DuckDB summary. |
| `evaluation/results/cli_runs_repair_existing/summary.json` | Post-repair CLI validation table. |
| `evaluation/results/duckdb_cli_validation/claude_cli/*/summary.json` | DuckDB second-engine validation. |

## Compile Paper

```bash
latexmk -pdf -interaction=nonstopmode paper.tex
```

The checked-in `paper.pdf` was compiled from `paper.tex`.
