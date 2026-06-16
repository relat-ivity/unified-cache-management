#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Native vLLM prefix caching stores reusable KV blocks in the normal GPU HBM KV cache.
# DeepGEMM accelerates FP8 MoE expert computation; it is independent of KV storage.
vllm serve /models/MiniMax-M2.7 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --no-disable-hybrid-kv-cache-manager \
  --kernel-config '{"moe_backend":"deep_gemm"}' \
  --host 0.0.0.0 \
  --port 8000 \
2>&1 | tee "./deepseek_vllm.log"
#   --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}'
