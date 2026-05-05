#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CLICKHOUSE_BIN="${DATAPUP_CLICKHOUSE_BIN:-$(cd "$ROOT_DIR/.." && pwd)/clickhouse}"
RUNTIME_DIR="${DATAPUP_CH_RUNTIME_DIR:-$ROOT_DIR/evaluation/clickhouse_runtime}"
TCP_PORT="${DATAPUP_CH_TCP_PORT:-19000}"
HTTP_PORT="${DATAPUP_CH_HTTP_PORT:-18123}"
INTERSERVER_PORT="${DATAPUP_CH_INTERSERVER_PORT:-19009}"
SCALE="${DATAPUP_SCALE:-0.1}"

if [[ ! -x "$CLICKHOUSE_BIN" ]]; then
  echo "ClickHouse binary not found: $CLICKHOUSE_BIN" >&2
  exit 1
fi

mkdir -p \
  "$RUNTIME_DIR/data" \
  "$RUNTIME_DIR/tmp" \
  "$RUNTIME_DIR/user_files" \
  "$RUNTIME_DIR/format_schemas" \
  "$RUNTIME_DIR/access" \
  "$RUNTIME_DIR/logs" \
  "$RUNTIME_DIR/preprocessed_configs"

CONFIG="$RUNTIME_DIR/config.xml"
USERS="$RUNTIME_DIR/users.xml"
PID_FILE="$RUNTIME_DIR/clickhouse.pid"

cat > "$CONFIG" <<EOF
<clickhouse>
    <logger>
        <level>information</level>
        <log>$RUNTIME_DIR/logs/clickhouse.log</log>
        <errorlog>$RUNTIME_DIR/logs/clickhouse.err.log</errorlog>
        <console>0</console>
    </logger>
    <path>$RUNTIME_DIR/data/</path>
    <tmp_path>$RUNTIME_DIR/tmp/</tmp_path>
    <user_files_path>$RUNTIME_DIR/user_files/</user_files_path>
    <format_schema_path>$RUNTIME_DIR/format_schemas/</format_schema_path>
    <access_control_path>$RUNTIME_DIR/access/</access_control_path>
    <status_file>$RUNTIME_DIR/status</status_file>
    <preprocessed_configs>$RUNTIME_DIR/preprocessed_configs/</preprocessed_configs>
    <listen_host>127.0.0.1</listen_host>
    <http_port>$HTTP_PORT</http_port>
    <tcp_port>$TCP_PORT</tcp_port>
    <interserver_http_port>$INTERSERVER_PORT</interserver_http_port>
    <max_connections>4096</max_connections>
    <keep_alive_timeout>3</keep_alive_timeout>
    <max_concurrent_queries>100</max_concurrent_queries>
    <uncompressed_cache_size>0</uncompressed_cache_size>
    <mark_cache_size>536870912</mark_cache_size>
    <users_config>$USERS</users_config>
    <default_profile>default</default_profile>
    <default_database>default</default_database>
    <timezone>UTC</timezone>
    <mlock_executable>false</mlock_executable>
</clickhouse>
EOF

cat > "$USERS" <<'EOF'
<clickhouse>
    <profiles>
        <default>
            <max_memory_usage>10000000000</max_memory_usage>
            <load_balancing>random</load_balancing>
        </default>
    </profiles>
    <users>
        <default>
            <password></password>
            <networks>
                <ip>::/0</ip>
            </networks>
            <profile>default</profile>
            <quota>default</quota>
            <access_management>1</access_management>
        </default>
    </users>
    <quotas>
        <default>
            <interval>
                <duration>3600</duration>
                <queries>0</queries>
                <errors>0</errors>
                <result_rows>0</result_rows>
                <read_rows>0</read_rows>
                <execution_time>0</execution_time>
            </interval>
        </default>
    </quotas>
</clickhouse>
EOF

echo "== Starting dedicated ClickHouse =="
echo "Binary: $CLICKHOUSE_BIN"
echo "Runtime: $RUNTIME_DIR"
echo "Native port: $TCP_PORT"
echo "HTTP port: $HTTP_PORT"

if evaluation/bin/clickhouse-client --host localhost --port "$TCP_PORT" --query "SELECT 1" >/dev/null 2>&1; then
  echo "[ok] ClickHouse already running on localhost:$TCP_PORT"
else
  "$CLICKHOUSE_BIN" server \
    --daemon \
    --config-file="$CONFIG" \
    --pid-file="$PID_FILE"

  ready=0
  for _ in $(seq 1 60); do
    if evaluation/bin/clickhouse-client --host localhost --port "$TCP_PORT" --query "SELECT 1" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done

  if [[ "$ready" != "1" ]]; then
    echo "ClickHouse did not become ready. Recent error log:" >&2
    tail -n 160 "$RUNTIME_DIR/logs/clickhouse.err.log" >&2 || true
    exit 1
  fi
  echo "[ok] ClickHouse started"
fi

echo
echo "== Preparing Python loader venv =="
VENV="$ROOT_DIR/evaluation/.venv_cli"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install faker tqdm clickhouse-connect >/dev/null
echo "[ok] loader dependencies installed in $VENV"

echo
echo "== Loading custom_analytics benchmark data =="
echo "Scale: $SCALE"
evaluation/bin/clickhouse-client --host localhost --port "$TCP_PORT" --multiquery --query "
DROP DATABASE IF EXISTS analytics SYNC;
"
evaluation/bin/clickhouse-client --host localhost --port "$TCP_PORT" --multiquery < evaluation/benchmark/schemas/custom_analytics/ddl.sql

"$VENV/bin/python" evaluation/benchmark/schemas/custom_analytics/generate_data.py \
  --host localhost \
  --port "$HTTP_PORT" \
  --scale "$SCALE"

echo
echo "== Verifying row counts =="
evaluation/bin/clickhouse-client --host localhost --port "$TCP_PORT" --query "
SELECT 'analytics.events' AS table, count() AS rows FROM analytics.events
UNION ALL SELECT 'analytics.users', count() FROM analytics.users
UNION ALL SELECT 'analytics.sessions', count() FROM analytics.sessions
UNION ALL SELECT 'analytics.products', count() FROM analytics.products
FORMAT PrettyCompact
"

echo
echo "CLICKHOUSE_READY"
