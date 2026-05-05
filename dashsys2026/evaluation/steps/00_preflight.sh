#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CLICKHOUSE_BIN="${DATAPUP_CLICKHOUSE_BIN:-$(cd "$ROOT_DIR/.." && pwd)/clickhouse}"

echo "== DataPup DASHSys CLI Experiment Preflight =="
echo "Repo: $ROOT_DIR"
echo

ok=1

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "[ok] $name: $(command -v "$name")"
  else
    echo "[missing] $name"
    ok=0
  fi
}

check_cmd python3
check_cmd claude
check_cmd codex
check_cmd gemini

if [[ -x "$CLICKHOUSE_BIN" ]]; then
  echo "[ok] clickhouse binary: $CLICKHOUSE_BIN"
  "$CLICKHOUSE_BIN" local --query "SELECT 1" >/dev/null
  echo "[ok] clickhouse local query works"
else
  echo "[missing] clickhouse binary: $CLICKHOUSE_BIN"
  ok=0
fi

if [[ -x evaluation/bin/clickhouse-client ]]; then
  echo "[ok] ClickHouse client wrapper: evaluation/bin/clickhouse-client"
else
  echo "[missing] ClickHouse client wrapper: evaluation/bin/clickhouse-client"
  ok=0
fi

python3 -m py_compile evaluation/run_cli_experiments.py evaluation/summarize_cli_runs.py
echo "[ok] Python experiment scripts compile"

if evaluation/bin/clickhouse-client --host localhost --port 19000 --query "SELECT 1" >/dev/null 2>&1; then
  echo "[ok] dedicated ClickHouse already running on localhost:19000"
else
  echo "[info] dedicated ClickHouse is not running yet on localhost:19000"
fi

echo
if [[ "$ok" == "1" ]]; then
  echo "PREFLIGHT_OK"
else
  echo "PREFLIGHT_FAILED"
  exit 1
fi
