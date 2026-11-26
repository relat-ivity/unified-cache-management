"""Adapter for collecting END tensors on older builds without the audit hook."""

import os

from ucm.integration.vllm.kv_bridge_hybrid_connector import (
    UCMKvBridgeHybridConnector as HybridConnector,
)
from ucm.integration.vllm.kvb_end_audit import dump_end_tensors


class UCMKvBridgeHybridConnector(HybridConnector):
    """Add the same audit to the older connector without changing its forward."""

    def _transfer_end_prefix_pages(self, request, point, attempt, states=None):
        directory = os.environ.get("UCM_KVB_END_AUDIT_DIR")
        selected = directory and os.environ.get("UCM_KVB_END_AUDIT_TOKENS") == str(point.tokens)
        if selected and states is not None:
            dump_end_tensors(self, request, point, attempt, states, directory)
        transferred = super()._transfer_end_prefix_pages(request, point, attempt, states)
        if selected and states is None:
            dump_end_tensors(self, request, point, attempt, None, directory)
        return transferred
