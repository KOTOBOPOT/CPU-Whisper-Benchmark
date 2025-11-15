#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Please create it (e.g. 'cp .env.example .env')." >&2
  exit 1
fi

# Load variables from .env into the current shell
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
WHISPER_MODEL_WEIGHTS_PATH=$LOCAL_WHISPER_WEIGHTS_DIR

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Neither '${PYTHON_BIN}' nor 'python' found in PATH. Please install Python 3.11+." >&2
    exit 1
  fi
fi

# Ensure project root (which contains utils/) is available for imports
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${PROJECT_ROOT}"
fi

HOST="${WHISPER_BACKEND_HOST:-0.0.0.0}"
PORT="${WHISPER_BACKEND_PORT:-8001}"
APP_MODULE="app.main:app"

if ! "${PYTHON_BIN}" -c "import uvicorn" >/dev/null 2>&1; then
  echo "The '${PYTHON_BIN}' environment is missing the 'uvicorn' package. Install dependencies via:" >&2
  echo "  ${PYTHON_BIN} -m pip install -r backends/distillation/backend/requirements.txt" >&2
  exit 1
fi

if [[ -n "${WHISPER_MODEL_WEIGHTS_PATH:-}" ]] && [[ ! -e "${WHISPER_MODEL_WEIGHTS_PATH}" ]]; then
  echo "Warning: WHISPER_MODEL_CHECKPOINT_PATH='${WHISPER_MODEL_WEIGHTS_PATH}' does not exist on the host filesystem." >&2
fi

cd "${SCRIPT_DIR}"
echo "Starting Whisper distillation backend with uvicorn (host: ${HOST}, port: ${PORT})..."
exec "${PYTHON_BIN}" -m uvicorn "${APP_MODULE}" --host "${HOST}" --port "${PORT}"
