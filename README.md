# DataPup Research: Schema-Aware Prompt Engineering for Text-to-SQL

Supplementary material for the VLDB 2026 Industrial Track paper:

> **DataPup: Schema-Aware Prompt Engineering for Text-to-SQL in an AI-Assisted Analytical Database Client [Application & Experience]**
>
> Sahith Vibudhi and Krishna Chaitanya Balusu

## Overview

This repository contains the benchmark datasets, evaluation framework, experimental results, and analysis code for our systematic ablation study of schema-aware prompt engineering for Text-to-SQL generation targeting analytical databases (ClickHouse).

**DataPup** is an open-source, AI-assisted database client: [https://github.com/DataPupOrg/DataPup](https://github.com/DataPupOrg/DataPup)

## Repository Structure

```
.
├── benchmark/                    # Benchmark datasets
│   ├── queries/                  # Query sets (JSON)
│   │   ├── aggregation.json      # 30 aggregation queries
│   │   ├── simple_select.json    # 25 simple SELECT queries
│   │   ├── window_functions.json # 25 window function queries
│   │   ├── time_series.json      # 30 time-series queries
│   │   ├── complex_joins.json    # 20 complex JOIN queries
│   │   ├── clickhouse_specific.json  # 20 ClickHouse-specific queries
│   │   ├── clickbench.json       # 43 ClickBench queries
│   │   └── ssb.json              # 13 Star Schema Benchmark queries
│   ├── schemas/                  # Database schemas
│   │   ├── custom_analytics/     # 4-table analytics schema
│   │   ├── clickbench/           # 1-table wide schema (105 columns)
│   │   └── ssb/                  # 5-table star schema
│   └── examples/
│       └── examples.json         # Few-shot example pool
│
├── framework/                    # Evaluation framework
│   ├── experiment_runner.py      # Main experiment orchestrator
│   ├── prompt_builder.py         # Prompt assembly (format, scope, metadata, examples)
│   ├── llm_caller.py             # LLM API interface
│   ├── sql_executor.py           # ClickHouse query execution
│   ├── result_comparator.py      # Semantic result comparison
│   ├── metrics.py                # EX, RC, SL, TE metric computation
│   ├── schema_linker.py          # Schema linking evaluation
│   ├── chain_of_thought.py       # CoT prompting variant
│   ├── self_consistency.py       # Self-consistency voting
│   └── self_corrector.py         # SQL self-correction
│
├── analysis/                     # Statistical analysis
│   ├── statistical_tests.py      # McNemar's test, Cochran's Q, bootstrap CIs
│   ├── visualizations.py         # Figure generation
│   ├── latex_tables.py           # LaTeX table generation
│   └── run_statistical_analysis.py  # Full analysis driver
│
├── results/                      # All experimental results
│   ├── phase1/                   # RQ1: Schema format comparison (JSONL + JSON summaries)
│   ├── phase2/                   # RQ2-RQ4: Scope, metadata, examples (JSONL + JSON summaries)
│   ├── ablation/                 # RQ5: System prompt ablation
│   ├── cross_model/              # RQ6: Claude 3.5 Sonnet vs Sonnet 4
│   ├── cross_dataset/            # RQ7: ClickBench and SSB generalization
│   ├── dail_sql/                 # RQ8: DAIL-SQL baseline comparison
│   ├── repeated_trials/          # Reproducibility trials
│   └── figures/                  # All figures (PDF + PNG)
│
├── config/                       # Experiment configuration
│   ├── experiment_config.yaml    # Ablation dimension definitions
│   └── model_config.yaml         # Model and inference settings
│
├── run_phase1.py                 # Phase 1 experiment runner
├── run_phase2.py                 # Phase 2 experiment runner
├── run_all_experiments.py        # Full experiment suite
├── run_single_config.py          # Single configuration runner
├── reevaluate.py                 # Re-evaluation with updated pipeline
└── generate_publication_outputs.py  # Generate figures and tables
```

## Benchmarks

| Dataset | Queries | Tables | Description |
|---------|---------|--------|-------------|
| Custom Analytics | 150 | 4 | Primary benchmark: events, users, sessions, products |
| ClickBench | 43 | 1 (105 cols) | Wide single-table web analytics |
| Star Schema Benchmark | 13 | 5 | Classical star schema (lineorder + dimensions) |

## Results Format

Per-query results are stored as JSONL files. Each line contains:

```json
{
  "query_id": "AG-001",
  "category": "Aggregation",
  "natural_language": "How many total events are in the events table?",
  "gold_sql": "SELECT count() FROM analytics.events",
  "predicted_sql": "SELECT count() AS total_events FROM analytics.events",
  "pred_executed": true,
  "result_match": true,
  "table_f1": 1.0,
  "column_f1": 1.0,
  "input_tokens": 1393,
  "latency_ms": 2394.68
}
```

## Key Results

| Configuration | RC | 95% CI |
|--------------|-----|--------|
| Baseline (DDL, Full, None, Zero-shot) | 29.3% | (22%–37%) |
| + Markdown format | 30.7% | (23%–38%) |
| + Relevant-subset scope | 59.3% | (51%–67%) |
| + Column descriptions | 60.7% | (53%–69%) |
| + Dynamic few-shot examples | 66.0% | (59%–74%) |
| + System prompt tuning (optimal) | **68.7%** | — |

## Reproducing Results

### Prerequisites

- Python 3.10+
- ClickHouse server with benchmark data loaded
- Anthropic API key (for Claude models)

### Setup

```bash
pip install -r requirements.txt
```

### Load benchmark data

```bash
./load_clickbench.sh   # Load ClickBench dataset
./load_ssb.sh          # Load Star Schema Benchmark
```

### Run experiments

```bash
# Full ablation study
python run_all_experiments.py

# Single configuration
python run_single_config.py --format markdown --scope relevant_subset \
    --metadata descriptions --examples dynamic_few_shot

# Statistical analysis
python analysis/run_statistical_analysis.py
```

## Citation

```bibtex
@article{vibudhi2026datapup,
  title={DataPup: Schema-Aware Prompt Engineering for Text-to-SQL
         in an AI-Assisted Analytical Database Client},
  author={Vibudhi, Sahith and Balusu, Krishna Chaitanya},
  journal={Proceedings of the VLDB Endowment},
  volume={19},
  year={2026}
}
```

## License

This research material is released under the MIT License.
