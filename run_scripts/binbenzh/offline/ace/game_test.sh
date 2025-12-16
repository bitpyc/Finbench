#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_MODULE="SeriousGame.agent_gym_anytrading"
BENCHMARK_NAME="SeriousGame"
TASK_NAME="SeriousGameTest"

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
  "$@" \
  > "${LOG_NAME}" 2>&1 &
