export ASCEND_RT_VISIBLE_DEVICES=4
export TRITON_ALL_BLOCKS_PARALLEL=1
export ENABLE_UCM_PATCH=1

vllm serve /models/Qwen3.5-0.8B \
  --gdn-prefill-backend triton \
  --no-enable-prefix-caching \
  --mamba-cache-mode all \
  --no-disable-hybrid-kv-cache-manager \
  --enforce-eager \
  --max-num-batched-tokens 16384 \
  --block-size 256 \
  --kv-transfer-config '{
    "kv_connector": "UCMKvBridgeHybridConnector",
    "kv_connector_module_path": "ucm.integration.vllm.kv_bridge_hybrid_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "UCM_CONFIG_FILE": "/root/unified-cache-management/examples/ucm_config_example.yaml"
    }
  }'