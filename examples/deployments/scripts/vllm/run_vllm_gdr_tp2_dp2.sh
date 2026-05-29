#!/usr/bin/env bash
set -euo pipefail

# DEVICE_GDR_NICS follows CUDA logical device order after CUDA_VISIBLE_DEVICES.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/path/to/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20000}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DEVICE_GDR_NICS="${DEVICE_GDR_NICS:-mlx5_0,mlx5_1,mlx5_2,mlx5_3}"
UCM_CONFIG_FILE="${UCM_CONFIG_FILE:-${REPO_ROOT}/examples/ucm_config_example.yaml}"

ENABLE_UCM_PATCH="${ENABLE_UCM_PATCH:-1}"
VLLM_HASH_ATTENTION="${VLLM_HASH_ATTENTION:-0}"
VLLM_CPU_AFFINITY="${VLLM_CPU_AFFINITY:-0}"
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"

count_csv_items() {
  local value="$1"
  local count=0
  local item
  local -a items=()

  IFS=',' read -ra items <<< "${value}"
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ -n "${item}" ]]; then
      count=$((count + 1))
    fi
  done

  echo "${count}"
}

required_gpu_count=$((TP_SIZE * DP_SIZE * PP_SIZE))
visible_gpu_count="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"
gdr_nic_count="$(count_csv_items "${DEVICE_GDR_NICS}")"

if [[ "${visible_gpu_count}" -ne "${required_gpu_count}" ]]; then
  echo "CUDA_VISIBLE_DEVICES has ${visible_gpu_count} devices, but TP*DP*PP requires ${required_gpu_count}." >&2
  exit 1
fi

if [[ "${gdr_nic_count}" -ne "${visible_gpu_count}" ]]; then
  echo "DEVICE_GDR_NICS has ${gdr_nic_count} NICs, but CUDA_VISIBLE_DEVICES has ${visible_gpu_count} devices." >&2
  exit 1
fi

if [[ "${MODEL}" == "/path/to/model" ]]; then
  echo "Please set MODEL=/path/to/model before running this script." >&2
  exit 1
fi

if [[ ! -f "${UCM_CONFIG_FILE}" ]]; then
  echo "UCM_CONFIG_FILE does not exist: ${UCM_CONFIG_FILE}" >&2
  exit 1
fi

if ! grep -Eiq '^[[:space:]]*use_gdr:[[:space:]]*true([[:space:]]*#.*)?$' "${UCM_CONFIG_FILE}"; then
  echo "Warning: ${UCM_CONFIG_FILE} does not contain 'use_gdr: true'; GDR may not be enabled." >&2
fi

export CUDA_VISIBLE_DEVICES
export DEVICE_GDR_NICS
export ENABLE_UCM_PATCH
export VLLM_HASH_ATTENTION
export VLLM_CPU_AFFINITY
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

KV_TRANSFER_CONFIG=$(
  cat <<EOF
{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"${UCM_CONFIG_FILE}"}}
EOF
)

cmd=(
  vllm serve "${MODEL}"
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --pipeline-parallel-size "${PP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --block-size "${BLOCK_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"
)

if [[ -n "${SERVED_MODEL_NAME}" ]]; then
  cmd+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

if [[ "${ENFORCE_EAGER:-0}" != "0" ]]; then
  cmd+=(--enforce-eager)
fi

if [[ "${ENABLE_PREFIX_CACHING:-0}" == "0" ]]; then
  cmd+=(--no-enable-prefix-caching)
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  cmd+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  cmd+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DEVICE_GDR_NICS=${DEVICE_GDR_NICS}"
echo "UCM_CONFIG_FILE=${UCM_CONFIG_FILE}"
echo "Starting vLLM with TP=${TP_SIZE}, DP=${DP_SIZE}, PP=${PP_SIZE}"

exec "${cmd[@]}"
