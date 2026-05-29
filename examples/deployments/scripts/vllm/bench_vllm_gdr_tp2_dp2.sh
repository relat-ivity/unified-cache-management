#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/models/DeepSeek-V2-Lite}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-/v1/completions}"

DATASET_NAME="${DATASET_NAME:-random}"
NUM_PROMPTS="${NUM_PROMPTS:-12}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-16000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-2}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-123456}"
RUNS="${RUNS:-2}"

PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}"
METRIC_PERCENTILES="${METRIC_PERCENTILES:-90,99}"
IGNORE_EOS="${IGNORE_EOS:-1}"
CHECK_READY="${CHECK_READY:-1}"

cmd=(
  vllm bench serve
  --backend vllm
  --model "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --host "${HOST}"
  --port "${PORT}"
  --endpoint "${ENDPOINT}"
  --dataset-name "${DATASET_NAME}"
  --num-prompts "${NUM_PROMPTS}"
  --random-input-len "${RANDOM_INPUT_LEN}"
  --random-output-len "${RANDOM_OUTPUT_LEN}"
  --request-rate "${REQUEST_RATE}"
  --seed "${SEED}"
  --percentile-metrics "${PERCENTILE_METRICS}"
  --metric-percentiles "${METRIC_PERCENTILES}"
)

if [[ "${IGNORE_EOS}" != "0" ]]; then
  cmd+=(--ignore-eos)
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

if [[ "${CHECK_READY}" != "0" ]] && command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null; then
    echo "Warning: vLLM server is not responding at http://${HOST}:${PORT}/v1/models" >&2
  fi
fi

for run in $(seq 1 "${RUNS}"); do
  echo "vLLM bench run ${run}/${RUNS}: ${MODEL} -> ${HOST}:${PORT}${ENDPOINT}"
  "${cmd[@]}"
done
