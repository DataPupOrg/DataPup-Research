#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV="$ROOT_DIR/evaluation/.venv_cli"
DUCKDB_PATH="$ROOT_DIR/evaluation/duckdb/datapup.duckdb"
OUT_DIR="$ROOT_DIR/evaluation/results/duckdb_cli_validation"
MODEL_CMD='/usr/local/bin/claude --dangerously-disable-osx-sandbox --dangerously-skip-permissions -p'

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing evaluation virtualenv: $VENV" >&2
  exit 1
fi

if [[ ! -f "$DUCKDB_PATH" ]]; then
  "$VENV/bin/python" evaluation/setup_duckdb.py --scale "${DATAPUP_DUCKDB_SCALE:-0.1}" --overwrite
fi

mkdir -p "$OUT_DIR/logs"

"$VENV/bin/python" evaluation/run_duckdb_cli_experiments.py \
  --model-name claude_cli \
  --model-cmd "$MODEL_CMD" \
  --model-cwd /tmp/datapup_duckdb_claude_baseline \
  --config baseline \
  --output-dir "$OUT_DIR" \
  --overwrite \
  > "$OUT_DIR/logs/claude_cli_baseline.log" 2>&1 &
pid_baseline=$!

"$VENV/bin/python" evaluation/run_duckdb_cli_experiments.py \
  --model-name claude_cli \
  --model-cmd "$MODEL_CMD" \
  --model-cwd /tmp/datapup_duckdb_claude_best \
  --config best \
  --output-dir "$OUT_DIR" \
  --overwrite \
  > "$OUT_DIR/logs/claude_cli_best.log" 2>&1 &
pid_best=$!

wait "$pid_baseline"
wait "$pid_best"

cat "$OUT_DIR/claude_cli/baseline/summary.json"
cat "$OUT_DIR/claude_cli/best/summary.json"
