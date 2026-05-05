#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV="$ROOT_DIR/evaluation/.venv_cli"
"$VENV/bin/python" evaluation/analyze_strong_accept_evidence.py

echo
echo "STRONG_ACCEPT_EVIDENCE_DONE"
echo "Read:"
echo "  evaluation/results/strong_accept_evidence.md"
echo "  evaluation/results/strong_accept_evidence.json"
