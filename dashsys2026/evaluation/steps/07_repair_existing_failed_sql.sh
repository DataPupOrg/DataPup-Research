#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "== Repair existing generated SQL: failed-execution rows only =="

if ! evaluation/bin/clickhouse-client --host localhost --port 19000 --query "SELECT 1" >/dev/null 2>&1; then
  echo "Dedicated ClickHouse is not ready on localhost:19000."
  echo "Run: bash evaluation/steps/01_prepare_clickhouse.sh"
  exit 1
fi

evaluation/repair_existing_cli_results.py \
  evaluation/results/cli_runs_full_generation/claude_cli/baseline/results.jsonl \
  evaluation/results/cli_runs_full_generation/claude_cli/best/results.jsonl \
  evaluation/results/cli_runs_full_generation/codex_cli/baseline/results.jsonl \
  evaluation/results/cli_runs_full_generation/codex_cli/best/results.jsonl \
  evaluation/results/cli_runs_full_generation/gemini_cli/baseline/results.jsonl \
  evaluation/results/cli_runs_full_generation/gemini_cli/best/results.jsonl \
  --output-dir evaluation/results/cli_runs_repair_existing \
  --max-repairs 1 \
  --model-timeout-sec 300 \
  --sql-timeout-sec 60 \
  --max-semantic-rows 5000 \
  --progress 25

echo
echo "REPAIR_EXISTING_DONE"
