#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${PROJECT_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Please create it (e.g. 'cp .env.example .env')." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Neither 'docker compose' nor 'docker-compose' is available in PATH." >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
echo "Building and starting Whisper backend template container..."
"${COMPOSE_CMD[@]}" up --build --remove-orphans -d

CONTAINER_NAME="whisper-backend-template"
PORT="${WHISPER_BACKEND_PORT:-8000}"

echo "${CONTAINER_NAME} is running on port ${PORT}."
