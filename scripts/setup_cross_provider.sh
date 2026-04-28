#!/usr/bin/env bash
# =============================================================================
# scripts/setup_cross_provider.sh
#
# One-shot setup for the 9-model cross-provider evaluation matrix.
#
# What it does:
#   1. Creates / activates .venv if missing
#   2. Installs the three provider SDKs at pinned-major versions
#      (anthropic, openai, google-genai) — only needed if any model is
#      configured with transport: sdk in cross_provider_models.yaml
#   3. Verifies that ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
#      are set (warn-only — CLI transport doesn't need them)
#   4. Optionally runs the smoke test (--smoke)
#
# For CLI transport (the default), install the public CLIs separately:
#   npm install -g @anthropic-ai/claude-code @google/gemini-cli
#   brew install --cask codex
# Then `claude login`, `codex login`, `gemini auth login`.
# Then run scripts/doctor_cli.py to verify.
#
# Usage:
#   bash scripts/setup_cross_provider.sh           # setup only
#   bash scripts/setup_cross_provider.sh --smoke   # setup + smoke test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${REPO_ROOT}"

echo "================================================================"
echo "DataPup-Research — cross-provider setup"
echo "Repo root: ${REPO_ROOT}"
echo "================================================================"

# ---------------------------------------------------------------------------
# Create / activate venv
# ---------------------------------------------------------------------------
echo
echo "[1/4] Setting up Python venv..."
PARENT_VENV="${REPO_ROOT}/../DataPup/.venv"
LOCAL_VENV="${REPO_ROOT}/.venv"

if [[ -d "${LOCAL_VENV}" ]]; then
  VENV_PATH="${LOCAL_VENV}"
  echo "  Using existing local venv: ${VENV_PATH}"
elif [[ -d "${PARENT_VENV}" ]]; then
  VENV_PATH="${PARENT_VENV}"
  echo "  Using parent repo venv: ${VENV_PATH}"
else
  VENV_PATH="${LOCAL_VENV}"
  echo "  Creating new local venv: ${VENV_PATH}"
  python3 -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
PY="${VENV_PATH}/bin/python"
PIP="${VENV_PATH}/bin/pip"

echo "  Python: $(${PY} --version)"
echo "  Pip:    $(${PIP} --version)"

# ---------------------------------------------------------------------------
# Install Python deps
# ---------------------------------------------------------------------------
echo
echo "[2/4] Installing Python dependencies..."
"${PIP}" install --upgrade --quiet \
  "anthropic>=0.40.0" \
  "openai>=1.60.0" \
  "google-genai>=1.0.0" \
  "pyyaml>=6.0" \
  "clickhouse-driver>=0.2.9"
echo "  Installed: anthropic, openai, google-genai, pyyaml, clickhouse-driver"

# ---------------------------------------------------------------------------
# Report on auth options
# ---------------------------------------------------------------------------
echo
echo "[3/4] Auth environment summary..."
echo "  CLI transport (default):"
echo "    Run 'python scripts/doctor_cli.py' to check claude/codex/gemini installation"
echo "  SDK transport (per-model override):"
[[ "${ANTHROPIC_API_KEY:-}" == "" ]] && echo "    ANTHROPIC_API_KEY: not set" || echo "    ANTHROPIC_API_KEY: set"
[[ "${OPENAI_API_KEY:-}" == "" ]] && echo "    OPENAI_API_KEY:    not set" || echo "    OPENAI_API_KEY:    set"
if [[ "${GOOGLE_API_KEY:-}" == "" && "${GEMINI_API_KEY:-}" == "" ]]; then
  echo "    GOOGLE_API_KEY:    not set"
else
  echo "    GOOGLE_API_KEY:    set"
fi

# ---------------------------------------------------------------------------
# Optional: smoke test
# ---------------------------------------------------------------------------
echo
echo "[4/4] Smoke test..."
if [[ "${1:-}" == "--smoke" ]]; then
  echo "  Running scripts/smoke_test_cross_provider.py..."
  "${PY}" "${REPO_ROOT}/scripts/smoke_test_cross_provider.py"
else
  echo "  Skipped (pass --smoke to run scripts/smoke_test_cross_provider.py)"
fi

echo
echo "================================================================"
echo "Setup complete."
echo
echo "Next steps:"
echo "  1. (CLI transport) Install public CLIs and log in:"
echo "       npm install -g @anthropic-ai/claude-code @google/gemini-cli"
echo "       brew install --cask codex"
echo "       claude login && codex login && gemini auth login"
echo "  2. Verify CLIs:"
echo "       python scripts/doctor_cli.py"
echo "  3. Smoke test all 9 models:"
echo "       python scripts/smoke_test_cross_provider.py"
echo "  4. Run full cross-provider evaluation:"
echo "       python scripts/run_cross_provider_evaluation.py --help"
echo "================================================================"
