"""Export selected KVB END states before storage and after restoration."""

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import torch

from ucm.logger import init_logger

logger = init_logger(__name__)


def _cpu(tensor):
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _fingerprint(tensor):
    raw = tensor.view(torch.uint8).numpy()
    return (str(tensor.dtype), tuple(tensor.shape), hashlib.sha256(raw).hexdigest())


def dump_end_tensors(connector, request, point, attempt, states, directory):
    """Save per-rank tensors using logical token positions, independent of physical slots."""
    from ucm.integration.vllm.kv_bridge_hybrid_connector import (
        _is_full_attention_group,
        block_size_from_kv_cache_spec,
    )

    phase = "save" if states is not None else "load"
    rank = connector.tp_rank % connector.tp_size
    try:
        ssm, fa = {}, {}
        fa_tensors = {} if os.environ.get("UCM_KVB_END_AUDIT_FA_TENSORS") == "1" else None
        for gid, group in enumerate(connector._kv_cache_config.kv_cache_groups):
            size = block_size_from_kv_cache_spec(group.kv_cache_spec)
            tail = (point.tokens - 1) // size
            table = request.group_vllm_block_ids[gid]
            for name in group.layer_names:
                cache = connector.kv_caches[name]
                if _is_full_attention_group(group.kv_cache_spec):
                    pages = fa[name] = []
                    if fa_tensors is not None:
                        fa_tensors[name] = []
                    for block in range(tail + 1):
                        length = min(size, point.tokens - block * size)
                        kv = tuple(_cpu(t[table[block], :length]) for t in cache[:2])
                        pages.append(tuple(_fingerprint(t) for t in kv))
                        if fa_tensors is not None:
                            fa_tensors[name].append(kv)
                else:
                    components = states[name] if states is not None else (t[table[tail]] for t in cache[:2])
                    ssm[name] = tuple(_cpu(t) for t in components)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{phase}-{attempt.hex()}-tp{rank}-{uuid4().hex[:8]}.pt"
        payload = {
            "phase": phase,
            "tokens": point.tokens,
            "key": point.key.hex(),
            "attempt": attempt.hex(),
            "tp_rank": rank,
            "tp_size": connector.tp_size,
            "ssm": ssm,
            "fa": fa,
        }
        if fa_tensors is not None:
            payload["fa_tensors"] = fa_tensors
        torch.save(payload, path)
        logger.info("KVB END audit phase=%s rank=%d tokens=%d file=%s", phase, rank, point.tokens, path)
    except Exception:
        # An audit failure must not fail a serving request.
        logger.exception("KVB END audit failed phase=%s tokens=%d rank=%d", phase, point.tokens, rank)
