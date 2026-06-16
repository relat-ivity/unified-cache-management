#!/usr/bin/env bash
set -euo pipefail

# 100%

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 50 \
  --num-warmup 1 \
  --prefix-repetition-prefix-len 128000 \
  --prefix-repetition-suffix-len 0 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --send 100 \
  2>&1 | tee -a "./result/100.log"

# 95%

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 1 \
  --prefix-repetition-prefix-len 121600 \
  --prefix-repetition-suffix-len 0 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --seed 95

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 50 \
  --prefix-repetition-prefix-len 121600 \
  --prefix-repetition-suffix-len 6400 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --seed 95 \
  2>&1 | tee -a "./result/95.log"

# 90%

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 1 \
  --prefix-repetition-prefix-len 115200 \
  --prefix-repetition-suffix-len 0 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --seed 90

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 50 \
  --prefix-repetition-prefix-len 115200 \
  --prefix-repetition-suffix-len 12800 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --seed 90 \
  2>&1 | tee -a "./result/90.log"

# 80%

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 1 \
  --prefix-repetition-prefix-len 102400 \
  --prefix-repetition-suffix-len 0 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --seed 80

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 50 \
  --prefix-repetition-prefix-len 102400 \
  --prefix-repetition-suffix-len 25600 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --seed 80 \
  2>&1 | tee -a "./result/80.log"

# 50%

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 1 \
  --prefix-repetition-prefix-len 64000 \
  --prefix-repetition-suffix-len 0 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --seed 50

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name prefix_repetition \
  --num-prompts 50 \
  --prefix-repetition-prefix-len 64000 \
  --prefix-repetition-suffix-len 64000 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 2 \
  --max-concurrency 1 \
  --seed 50 \
  2>&1 | tee -a "./result/50.log"

# 0%
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model /models/MiniMax-M2.7 \
  --dataset-name random \
  --num-prompts 50 \
  --random-input-len 128000 \
  --random-output-len 2 \
  --random-prefix-len 0 \
  --max-concurrency 1 \
  --seed 0 \
  2>&1 | tee -a "./result/0.log"
