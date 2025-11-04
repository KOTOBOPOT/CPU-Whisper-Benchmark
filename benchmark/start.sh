#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  echo "ENV_FILE: ${ENV_FILE}"
fi

: "${BENCH_NAME:?BENCH_NAME must be set (check .env).}"
: "${WHISPER_BACKEND_PORT:?WHISPER_BACKEND_PORT must be set (check .env).}"

BACKEND_NAME="${WHISPER_BACKEND_NAME:-unknown}"
BACKEND_HOST="${WHISPER_BACKEND_HOST:-http://127.0.0.1}"
RESULTS_ROOT="${RESULTS_DIR:-${REPO_ROOT}/results}"
RUN_STEM="${BACKEND_NAME}_${BENCH_NAME}"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
OUTPUT_DIR="${RESULTS_ROOT}/${RUN_STEM}/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
  "--bench-name" "${BENCH_NAME}"
  "--backend-port" "${WHISPER_BACKEND_PORT}"
  "--backend-host" "${BACKEND_HOST}"
  "--backend-name" "${BACKEND_NAME}"
  "--output-dir" "${OUTPUT_DIR}"
  "--data-root" "${BENCH_DATA_ROOT:-${REPO_ROOT}/benchmark/data}"
)

if [[ -n "${BENCH_ENDPOINT:-}" ]]; then
  ARGS+=("--endpoint" "${BENCH_ENDPOINT}")
fi

if [[ -n "${BENCH_MAX_SAMPLES:-}" ]]; then
  ARGS+=("--max-samples" "${BENCH_MAX_SAMPLES}")
fi

if [[ -n "${BENCH_REQUEST_TIMEOUT:-}" ]]; then
  ARGS+=("--request-timeout" "${BENCH_REQUEST_TIMEOUT}")
fi

if [[ -n "${BENCH_SLEEP:-}" ]]; then
  ARGS+=("--sleep" "${BENCH_SLEEP}")
fi

if [[ -n "${BENCH_PAYLOAD_KEY:-}" ]]; then
  ARGS+=("--payload-key" "${BENCH_PAYLOAD_KEY}")
fi

if [[ -n "${BENCH_TEXT_KEY:-}" ]]; then
  ARGS+=("--text-key" "${BENCH_TEXT_KEY}")
fi

if [[ -n "${BENCH_ALLOW_BLANK_TEXT:-}" ]]; then
  case "${BENCH_ALLOW_BLANK_TEXT,,}" in
    1|true|yes|on)
      ARGS+=("--allow-blank-text")
      ;;
  esac
fi

if [[ -n "${BENCH_EXTRA_HEADERS:-}" ]]; then
  mapfile -t HEADER_ITEMS < <(printf '%s' "${BENCH_EXTRA_HEADERS}" | tr ';' $'\n' | tr -d $'\r')
  for raw_header in "${HEADER_ITEMS[@]}"; do
    trimmed="${raw_header#${raw_header%%[![:space:]]*}}"
    trimmed="${trimmed%${trimmed##*[![:space:]]}}"
    if [[ -n "${trimmed}" ]]; then
      ARGS+=("--header" "${trimmed}")
    fi
  done
fi

printf 'Running benchmark %s against backend %s...\n' "${BENCH_NAME}" "${BACKEND_NAME}"
python3 "${REPO_ROOT}/benchmark/run_benchmark.py" "${ARGS[@]}"

printf 'Results saved to %s\n' "${OUTPUT_DIR}"
