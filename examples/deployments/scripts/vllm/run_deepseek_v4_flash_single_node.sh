#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # ,4,5,6,7
export DEVICE_GDR_NICS=mlx5_0,mlx5_2,mlx5_4,mlx5_6 # ,mlx5_8,mlx5_10,mlx5_12,mlx5_14

LOG_FILE="./vllm_deepseek_v4_flash_$(date +%Y%m%d_%H%M%S).log"

vllm serve /home/models/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --gpu-memory-utilization 0.8 \
  --no-disable-hybrid-kv-cache-manager \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-transfer-config \
  '{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "./examples/ucm_deepseek_v4_flash_cache_posix_local.yaml"}
  }' \
2>&1 | tee "./vllm_deepseek_v4_flash_$(date +%Y%m%d_%H%M%S).log"
