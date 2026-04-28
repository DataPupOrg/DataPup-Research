#!/usr/bin/env bash
# =============================================================================
# scripts/setup_cross_provider.sh
#
# One-shot setup for the 9-model cross-provider evaluation matrix.
#
# What it does:
#   1. Creates / activates .venv if missing
#   2. Installs the three provider SDKs at pinned-major versions
#      (anthropic, openai, google-genai)
#   3. Verifies that ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
#      are set in the environment
#   4. Refuses to run if any Meta-CLI environment variables are detected
#      (per memory feedback_no_meta_cli_for_datapup.md)
#   5. Optionally runs the smoke test (--smoke)
#
# Usage:
#   bash scripts/setup_cross_provider.sh           # setup only
#   bash scripts/setup_cross_provider.sh --smoke   # setup + smoke test
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate repo root (this script lives in scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${REPO_ROOT}"

echo "================================================================"
echo "DataPup-Research — cross-provider setup"
echo "Repo root: ${REPO_ROOT}"
echo "================================================================"

# ---------------------------------------------------------------------------
# Refuse Meta-CLI environment leakage
# ---------------------------------------------------------------------------
echo
echo "[1/5] Checking for Meta-CLI environment leakage..."
LEAKS=()
[[ "${ANTHROPIC_BASE_URL:-}" != "" ]] && LEAKS+=("ANTHROPIC_BASE_URL")
[[ "${ANTHROPIC_CUSTOM_HEADERS:-}" != "" ]] && LEAKS+=("ANTHROPIC_CUSTOM_HEADERS")
[[ "${OPENAI_BASE_URL:-}" != "" ]] && LEAKS+=("OPENAI_BASE_URL")
[[ "${GOOGLE_GENAI_USE_VERTEXAI:-}" != "" ]] && LEAKS+=("GOOGLE_GENAI_USE_VERTEXAI")
if [[ ${#LEAKS[@]} -gt 0 ]]; then
  echo "  ERROR: the following env vars are set and may route inference through"
  echo "         Meta-internal infrastructure or proxies:"
  for v in "${LEAKS[@]}"; do echo "    - $v"; done
  echo
  echo "  The DataPup paper requires direct calls under personal credentials."
  echo "  Unset these and re-run:"
  for v in "${LEAKS[@]}"; do echo "    unset $v"; done
  exit 1
fi
echo "  OK — no Meta-CLI leakage detected."

# ---------------------------------------------------------------------------
# Create / activate venv
# ---------------------------------------------------------------------------
echo
echo "[2/5] Setting up Python venv..."
# Prefer parent DataPup repo's .venv if present (existing analysis env);
# fall back to a fresh local .venv.
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
# Install / upgrade SDKs
# ---------------------------------------------------------------------------
echo
echo "[3/5] Installing provider SDKs..."
# Pinned to major versions known to support GPT-5.x, Claude 4.x, Gemini 2.5
"${PIP}" install --upgrade --quiet \
  "anthropic>=0.40.0" \
  "openai>=1.60.0" \
  "google-genai>=1.0.0" \
  "pyyaml>=6.0" \
  "clickhouse-driver>=0.2.9"
echo "  Installed: anthropic, openai, google-genai, pyyaml, clickhouse-driver"

# ---------------------------------------------------------------------------
# Verify API keys
# ---------------------------------------------------------------------------
echo
echo "[4/5] Verifying API key environment variables..."
MISSING=()
[[ "${ANTHROPIC_API_KEY:-}" == "" ]] && MISSING+=("ANTHROPIC_API_KEY")
[[ "${OPENAI_API_KEY:-}" == "" ]] && MISSING+=("OPENAI_API_KEY")
if [[ "${GOOGLE_API_KEY:-}" == "" && "${GEMINI_API_KEY:-}" == "" ]]; then
  MISSING+=("GOOGLE_API_KEY (or GEMINI_API_KEY)")
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "  WARNING: missing keys (smoke test will skip those providers):"
  for k in "${MISSING[@]}"; do echo "    - $k"; done
  echo
  echo "  To set them (use personal API keys, NOT Meta-issued):"
  echo "    export ANTHROPIC_API_KEY=sk-ant-..."
  echo "    export OPENAI_API_KEY=sk-..."
  echo "    export GOOGLE_API_KEY=AIza..."
  echo
  echo "  Get keys at:"
  echo "    https://console.anthropic.com/settings/keys"
  echo "    https://platform.openai.com/api-keys"
  echo "    https://aistudio.google.com/apikey"
else
  echo "  OK — all 3 provider keys are set."
fi

# ---------------------------------------------------------------------------
# Optional: smoke test
# ---------------------------------------------------------------------------
echo
echo "[5/5] Smoke test..."
if [[ "${1:-}" == "--smoke" ]]; then
  if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "  Skipping smoke test — set the missing keys first."
  else
    echo "  Running scripts/smoke_test_cross_provider.py..."
    "${PY}" "${REPO_ROOT}/scripts/smoke_test_cross_provider.py"
  fi
else
  echo "  Skipped (pass --smoke to run scripts/smoke_test_cross_provider.py)"
fi

echo
echo "================================================================"
echo "Setup complete."
echo
echo "Next steps:"
echo "  1. Set any missing API keys (see [4/5] above)"
echo "  2. Run smoke test:"
echo "       python scripts/smoke_test_cross_provider.py"
echo "  3. Run full cross-provider evaluation:"
echo "       python scripts/run_cross_provider_evaluation.py --help"
echo "================================================================"
