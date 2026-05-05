#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

found=0
for dir in evaluation/results/cli_runs*; do
  if [[ -d "$dir" ]]; then
    if find "$dir" -name summary.json -print -quit | grep -q .; then
      echo "== Summarizing $dir =="
      python3 evaluation/summarize_cli_runs.py --input-dir "$dir"
      found=1
      echo
    fi
  fi
done

if [[ "$found" != "1" ]]; then
  echo "No CLI run summaries found under evaluation/results/cli_runs*"
  exit 1
fi

echo
echo "SUMMARY_DONE"
echo "Read:"
echo "  evaluation/results/cli_runs*/cli_runs_summary.md"
echo "  evaluation/results/cli_runs*/cli_runs_failures.md"
