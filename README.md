# DataPup Research Artifacts

This repository contains the public research artifacts for DataPup, an open-source AI-assisted analytical database client.

The current submission artifact is in:

```bash
dashsys2026/
```

## DASHSys 2026 Paper

> DataPup: Schema-Aware and Execution-Aware Text-to-SQL for Human-in-the-Loop Analytical Database Clients
>
> Sahith Vibudhi and Krishna Chaitanya Balusu

DataPup project repository: <https://github.com/DataPupOrg/DataPup>

## Artifact Map

| Path | Contents |
|---|---|
| `dashsys2026/paper.pdf` | Compiled DASHSys 2026 submission PDF. |
| `dashsys2026/paper.tex` | LaTeX source for the submission. |
| `dashsys2026/evaluation/benchmark/` | Query sets, schemas, and few-shot examples. |
| `dashsys2026/evaluation/framework/` | Prompt construction, SQL execution, metrics, and result comparison code. |
| `dashsys2026/evaluation/steps/` | Fire-and-forget scripts used for the CLI and DuckDB experiments. |
| `dashsys2026/evaluation/results/headline_contrasts.*` | Planned paired contrasts reported in the paper. |
| `dashsys2026/evaluation/results/fewshot_leakage_audit.json` | Few-shot leakage audit. |
| `dashsys2026/evaluation/results/cli_runs_full_generation/` | Raw CLI generation outputs for Claude, Codex, and Gemini. |
| `dashsys2026/evaluation/results/cli_runs_full_generation_validation/` | Execution-scored CLI outputs before repair. |
| `dashsys2026/evaluation/results/cli_runs_repair_existing/` | Scored outputs after one execution-repair attempt. |
| `dashsys2026/evaluation/results/duckdb_cli_validation/` | Focused DuckDB second-engine validation. |
| `dashsys2026/evaluation/results/strong_accept_evidence.*` | Consolidated evidence used by the DASHSys revision. |

The root-level `benchmark/`, `framework/`, `results/`, and `scripts/` directories are retained for the earlier VLDB 2026 artifact and cross-provider scaffolding. The DASHSys paper should be evaluated against `dashsys2026/`.

## Key DASHSys Results

| Result | Evidence |
|---|---|
| Current DataPup full-schema JSON zero-shot: 17.3% RC. | `dashsys2026/evaluation/results/strong_accept_evidence.md` |
| Revised prompt: 66.0% RC on the 150-query custom analytics benchmark. | `dashsys2026/evaluation/results/headline_contrasts.md` |
| Current-to-revised paired comparison: McNemar exact p=2.7e-19. | `dashsys2026/evaluation/results/strong_accept_evidence.json` |
| Best configuration remains strongest across Claude, Codex, and Gemini CLI runs after one repair attempt. | `dashsys2026/evaluation/results/cli_runs_repair_existing/summary.json` |
| Focused DuckDB validation: revised prompt reaches 60.8% RC with 100.0% execution success on 130 portable queries. | `dashsys2026/evaluation/results/duckdb_cli_validation/` |

## Quick Reproduction

```bash
cd dashsys2026
python3 -m venv evaluation/.venv_cli
evaluation/.venv_cli/bin/python -m pip install -r requirements.txt
evaluation/.venv_cli/bin/python evaluation/setup_duckdb.py --scale 0.1 --overwrite
bash evaluation/steps/08_duckdb_claude_validation_parallel.sh
bash evaluation/steps/09_strong_accept_evidence.sh
```

The ClickHouse plus multi-CLI experiment can be rerun from `dashsys2026/` after installing/authenticating the Claude, Codex, and Gemini CLIs and providing a local ClickHouse binary:

```bash
export DATAPUP_CLICKHOUSE_BIN=/path/to/clickhouse
bash evaluation/steps/00_preflight.sh
bash evaluation/steps/01_prepare_clickhouse.sh
nohup bash evaluation/steps/04_full_generation_parallel.sh \
  > evaluation/results/cli_runs_full_generation.nohup.log 2>&1 &
```

Then score and repair:

```bash
bash evaluation/steps/05_full_execute_parallel.sh
bash evaluation/steps/07_repair_existing_failed_sql.sh
bash evaluation/steps/09_strong_accept_evidence.sh
```

## Notes

- The CLI experiments intentionally use each provider CLI in noninteractive permissive mode, as encoded in `dashsys2026/evaluation/steps/04_full_generation_parallel.sh`.
- No API keys are stored in this repository.
- Local runtime directories, virtual environments, generated DuckDB databases, ClickHouse data directories, and nohup logs are ignored.

## License

This research material is released under the MIT License.
