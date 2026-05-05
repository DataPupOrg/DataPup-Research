#!/usr/bin/env bash
set -euo pipefail

# Fire-and-forget matrix runner for DataPup CLI model experiments.
#
# Configure model commands via environment variables or an env file, then run:
#
#   nohup bash evaluation/run_cli_matrix.sh > evaluation/results/cli_runs/matrix.nohup.log 2>&1 &
#
# Default model commands run all three CLIs in permissive/noninteractive mode:
#   Claude: --dangerously-skip-permissions
#   Codex:  --dangerously-bypass-approvals-and-sandbox
#   Gemini: --approval-mode=yolo
#
# The Meta launcher-level --dangerously-disable-osx-sandbox flag is also used
# because these wrappers otherwise try to create a sandbox that can fail in
# long-running shell contexts.
#
# Override any command with:
#   CLAUDE_MODEL_CMD='...'
#   CODEX_MODEL_CMD='...'
#   GEMINI_MODEL_CMD='...'
#
# Disable a model with:
#   RUN_CLAUDE=0
#   RUN_CODEX=0
#   RUN_GEMINI=0
#
# Commands are passed to run_cli_experiments.py. If the command contains the
# literal token {prompt_file} or {prompt}, the runner substitutes it. Otherwise,
# it sends the prompt on stdin.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${CLI_ENV_FILE:-}" ]]; then
  if [[ -r "$CLI_ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$CLI_ENV_FILE"
  fi
elif [[ -f "evaluation/cli_matrix.env" ]]; then
  # shellcheck source=/dev/null
  source "evaluation/cli_matrix.env"
fi

: "${OUTPUT_DIR:=$ROOT_DIR/evaluation/results/cli_runs}"
: "${DATASET:=custom_analytics}"
: "${CONFIGS:=baseline best}"
: "${RELEVANT_SOURCE:=heuristic}"
: "${LIMIT:=}"
: "${QUERY_IDS:=}"
: "${CATEGORIES:=}"
: "${EXECUTE:=0}"
: "${REPAIR_ON_ERROR:=0}"
: "${MAX_REPAIRS:=1}"
: "${MODEL_TIMEOUT_SEC:=300}"
: "${SQL_TIMEOUT_SEC:=60}"
: "${CLICKHOUSE_CLIENT:=$ROOT_DIR/evaluation/bin/clickhouse-client}"
: "${CLICKHOUSE_HOST:=localhost}"
: "${CLICKHOUSE_PORT:=19000}"
: "${CLICKHOUSE_USER:=default}"
: "${CLICKHOUSE_PASSWORD:=${CLICKHOUSE_PASSWORD:-}}"
: "${OVERWRITE:=1}"
: "${RESUME:=0}"
: "${PARALLEL:=0}"
: "${RUN_CLAUDE:=1}"
: "${RUN_CODEX:=1}"
: "${RUN_GEMINI:=1}"
if [[ -z "${CLAUDE_MODEL_CMD:-}" ]]; then
  CLAUDE_MODEL_CMD='/usr/local/bin/claude --dangerously-disable-osx-sandbox --dangerously-skip-permissions -p'
fi
if [[ -z "${CODEX_MODEL_CMD:-}" ]]; then
  CODEX_MODEL_CMD='/usr/local/bin/codex --dangerously-disable-osx-sandbox exec --dangerously-bypass-approvals-and-sandbox -'
fi
if [[ -z "${GEMINI_MODEL_CMD:-}" ]]; then
  GEMINI_MODEL_CMD='/usr/local/bin/gemini --dangerously-disable-osx-sandbox --approval-mode=yolo -p ""'
fi

mkdir -p "$OUTPUT_DIR/logs"

declare -a MODEL_NAMES=()
declare -a MODEL_CMDS=()

add_model() {
  local name="$1"
  local cmd="$2"
  if [[ -n "$cmd" ]]; then
    MODEL_NAMES+=("$name")
    MODEL_CMDS+=("$cmd")
  fi
}

if [[ "$RUN_CLAUDE" == "1" ]]; then
  add_model "claude_cli" "$CLAUDE_MODEL_CMD"
fi
if [[ "$RUN_CODEX" == "1" ]]; then
  add_model "codex_cli" "$CODEX_MODEL_CMD"
fi
if [[ "$RUN_GEMINI" == "1" ]]; then
  add_model "gemini_cli" "$GEMINI_MODEL_CMD"
fi

if [[ "${#MODEL_NAMES[@]}" -eq 0 ]]; then
  cat >&2 <<'EOF'
No models enabled.

Enable at least one model, for example:

  export RUN_CLAUDE=1
  export RUN_CODEX=1
  export RUN_GEMINI=1

Then run:

  bash evaluation/run_cli_matrix.sh

If your CLI needs a prompt file instead of stdin, use {prompt_file}, e.g.:

  export SOME_MODEL_CMD='some-cli --prompt-file {prompt_file}'
EOF
  exit 2
fi

build_base_args() {
  local model_name="$1"
  local model_cmd="$2"
  local config="$3"
  local model_cwd="/tmp/datapup_cli_${model_name}"
  mkdir -p "$model_cwd"

  local args=(
    "python3" "evaluation/run_cli_experiments.py"
    "--model-name" "$model_name"
    "--model-cmd" "$model_cmd"
    "--model-cwd" "$model_cwd"
    "--model-timeout-sec" "$MODEL_TIMEOUT_SEC"
    "--dataset" "$DATASET"
    "--config" "$config"
    "--relevant-source" "$RELEVANT_SOURCE"
    "--output-dir" "$OUTPUT_DIR"
    "--sql-timeout-sec" "$SQL_TIMEOUT_SEC"
  )

  if [[ -n "$LIMIT" ]]; then
    args+=("--limit" "$LIMIT")
  fi

  if [[ -n "$QUERY_IDS" ]]; then
    IFS=',' read -r -a ids <<< "$QUERY_IDS"
    for id in "${ids[@]}"; do
      [[ -n "$id" ]] && args+=("--query-id" "$id")
    done
  fi

  if [[ -n "$CATEGORIES" ]]; then
    IFS=',' read -r -a cats <<< "$CATEGORIES"
    for cat in "${cats[@]}"; do
      [[ -n "$cat" ]] && args+=("--category" "$cat")
    done
  fi

  if [[ "$EXECUTE" == "1" ]]; then
    args+=(
      "--execute"
      "--clickhouse-client" "$CLICKHOUSE_CLIENT"
      "--clickhouse-host" "$CLICKHOUSE_HOST"
      "--clickhouse-port" "$CLICKHOUSE_PORT"
      "--clickhouse-user" "$CLICKHOUSE_USER"
    )
    if [[ -n "$CLICKHOUSE_PASSWORD" ]]; then
      args+=("--clickhouse-password" "$CLICKHOUSE_PASSWORD")
    fi
  fi

  if [[ "$REPAIR_ON_ERROR" == "1" ]]; then
    args+=("--repair-on-error" "--max-repairs" "$MAX_REPAIRS")
  fi

  if [[ "$OVERWRITE" == "1" ]]; then
    args+=("--overwrite")
  fi

  if [[ "$RESUME" == "1" ]]; then
    args+=("--resume")
  fi

  printf '%q ' "${args[@]}"
}

run_one() {
  local model_name="$1"
  local model_cmd="$2"
  local config="$3"
  local log_file="$OUTPUT_DIR/logs/${model_name}_${config}.log"

  echo "================================================================"
  echo "Running model=${model_name} config=${config}"
  echo "Log: ${log_file}"
  echo "================================================================"

  local quoted_cmd
  quoted_cmd="$(build_base_args "$model_name" "$model_cmd" "$config")"
  echo "$quoted_cmd" > "$OUTPUT_DIR/logs/${model_name}_${config}.cmd"
  bash -lc "$quoted_cmd" > "$log_file" 2>&1
}

declare -a PIDS=()
IFS=' ' read -r -a CONFIG_ARRAY <<< "$CONFIGS"

for i in "${!MODEL_NAMES[@]}"; do
  model_name="${MODEL_NAMES[$i]}"
  model_cmd="${MODEL_CMDS[$i]}"
  for config in "${CONFIG_ARRAY[@]}"; do
    if [[ "$PARALLEL" == "1" ]]; then
      run_one "$model_name" "$model_cmd" "$config" &
      PIDS+=("$!")
    else
      run_one "$model_name" "$model_cmd" "$config"
    fi
  done
done

if [[ "$PARALLEL" == "1" ]]; then
  status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "At least one matrix job failed. Check $OUTPUT_DIR/logs." >&2
    exit "$status"
  fi
fi

python3 evaluation/summarize_cli_runs.py --input-dir "$OUTPUT_DIR"

echo
echo "CLI matrix complete."
echo "Summaries:"
echo "  $OUTPUT_DIR/cli_runs_summary.csv"
echo "  $OUTPUT_DIR/cli_runs_summary.md"
