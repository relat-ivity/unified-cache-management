#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve /models/MiniMax-M2.7 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --max-num-batched-tokens 32768 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-transfer-config \
  '{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "./examples/ucm_deepseek_cuda.yaml"}
  }' \
2>&1 | tee "./deepseek_vllm.log"