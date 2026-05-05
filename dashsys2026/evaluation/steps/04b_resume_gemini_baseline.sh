#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "== Resume generation: Gemini baseline only, append missing custom_analytics queries =="

RUN_CLAUDE=0 \
RUN_CODEX=0 \
RUN_GEMINI=1 \
CLAUDE_MODEL_CMD='/usr/local/bin/claude --dangerously-disable-osx-sandbox --dangerously-skip-permissions -p' \
CODEX_MODEL_CMD='/usr/local/bin/codex --dangerously-disable-osx-sandbox exec --dangerously-bypass-approvals-and-sandbox -' \
GEMINI_MODEL_CMD='/usr/local/bin/gemini --dangerously-disable-osx-sandbox --approval-mode=yolo -p ""' \
CONFIGS='baseline' \
LIMIT= \
EXECUTE=0 \
REPAIR_ON_ERROR=0 \
PARALLEL=0 \
OVERWRITE=0 \
RESUME=1 \
OUTPUT_DIR="$ROOT_DIR/evaluation/results/cli_runs_full_generation" \
CLI_ENV_FILE=/dev/null bash evaluation/run_cli_matrix.sh

echo
echo "GEMINI_BASELINE_RESUME_DONE"
