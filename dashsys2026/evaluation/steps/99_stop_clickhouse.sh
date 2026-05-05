#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if evaluation/bin/clickhouse-client --host localhost --port 19000 --query "SELECT 1" >/dev/null 2>&1; then
  evaluation/bin/clickhouse-client --host localhost --port 19000 --query "SYSTEM SHUTDOWN" || true
  echo "STOP_REQUESTED"
else
  echo "ClickHouse is not running on localhost:19000"
fi
