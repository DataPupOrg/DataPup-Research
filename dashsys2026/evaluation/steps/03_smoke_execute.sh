#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "== Smoke execute: all enabled models, baseline+best, 3 queries, SQL execution + repair =="

if ! evaluation/bin/clickhouse-client --host localhost --port 19000 --query "SELECT 1" >/dev/null 2>&1; then
  echo "Dedicated ClickHouse is not ready on localhost:19000."
  echo "Run: bash evaluation/steps/01_prepare_clickhouse.sh"
  exit 1
fi

RUN_CLAUDE="${RUN_CLAUDE:-1}" \
RUN_CODEX="${RUN_CODEX:-1}" \
RUN_GEMINI="${RUN_GEMINI:-1}" \
CLAUDE_MODEL_CMD='/usr/local/bin/claude --dangerously-disable-osx-sandbox --dangerously-skip-permissions -p' \
CODEX_MODEL_CMD='/usr/local/bin/codex --dangerously-disable-osx-sandbox exec --dangerously-bypass-approvals-and-sandbox -' \
GEMINI_MODEL_CMD='/usr/local/bin/gemini --dangerously-disable-osx-sandbox --approval-mode=yolo -p ""' \
CONFIGS='baseline best' \
LIMIT=3 \
EXECUTE=1 \
REPAIR_ON_ERROR=1 \
MAX_REPAIRS=1 \
PARALLEL=0 \
OVERWRITE=1 \
OUTPUT_DIR="$ROOT_DIR/evaluation/results/cli_runs_smoke_execute" \
CLI_ENV_FILE=/dev/null bash evaluation/run_cli_matrix.sh

echo
echo "SMOKE_EXECUTE_DONE"
