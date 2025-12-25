#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_MODULE="Consulting.test_case"
BENCHMARK_NAME="Consulting"
TASK_NAME="ConsultingInterview"
CASEBOOK_JSON="Consulting/casebook_all_cases.json"

MODEL="deepseek-v3"
BACKEND="openai"
API_BASE="http://35.220.164.252:3888/v1/"
MAX_TURNS=10
RETRIEVE_K=8
TEMPERATURE=0.5

SAVE_PATH="results/${BENCHMARK_NAME}_run"
LOG_NAME="${BENCHMARK_NAME}_run_${TASK_NAME}.log"

echo "Starting ${TASK_NAME} run. Logs: ${LOG_NAME}"

nohup python -m "${BENCHMARK_MODULE}" \
  --casebook_json "${CASEBOOK_JSON}" \
  --output_dir "${SAVE_PATH}" \
  --model "${MODEL}" \
  --backend "${BACKEND}" \
  --api_base "${API_BASE}" \
  --max_turns "${MAX_TURNS}" \
  --retrieve_k "${RETRIEVE_K}" \
  --temperature "${TEMPERATURE}" \
  "$@" \
  > "${LOG_NAME}" 2>&1 &
