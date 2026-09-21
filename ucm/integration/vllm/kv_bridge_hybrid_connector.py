import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from ucm.integration.vllm.hla_connector import (
    HLARequestMeta,
    UCMHybridLinearAttentionConnector,
    block_size_from_kv_cache_spec,
    layer_name_to_kv_cache_spec,
)
from ucm.integration.vllm.ucm_connector import {
    RequestDispatchMeta,
    RequestHasher, 
    _record_counter,
}
from ucm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def _concrete_specs(spec: KVCacheSpec) -> list[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return list(spec.kv_cache_specs.values())
    return [spec]


def _is_full_attention_group(spec: KVCacheSpec) -> bool:
    specs = _concrete_specs(spec)
    return bool(specs) and all(isinstance(s, FullAttentionSpec) for s in specs)


def _is_mamba_all_group(spec: KVCacheSpec) -> bool:
    specs = _concrete_specs(spec)
    return bool(specs) and all(
        isinstance(s, MambaSpec) and s.mamba_cache_mode == "all" for s in specs
    )


@dataclass
class MambaAllGroupInfo:
    group_id: int
    block_size: int
    layer_names: tuple[str, ...]
    seed: bytes
    is_mamba_all: bool

    @property
    def is_full_attention(self) -> bool:
        return not self.is_mamba_all


class MambaAllGroupManager:
    """Hash and lookup manager for full-attention + Mamba all-mode groups.

    In all mode every completed Mamba block owns a persistent state, so Mamba
    groups participate in the same prefix lookup contract as attention groups.
    There are no align-mode null blocks and no reverse lookup for a single
    checkpoint state.
    """

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        request_hasher: RequestHasher,
        base_seed: bytes,
    ) -> None:
        self.request_hasher = request_hasher
        self.groups_by_id: list[MambaAllGroupInfo] = []
        self.full_attn_groups: list[MambaAllGroupInfo] = []
        self.state_groups: list[MambaAllGroupInfo] = []

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            is_mamba_all = _is_mamba_all_group(spec)
            is_full_attention = _is_full_attention_group(spec)

            if not is_mamba_all and not is_full_attention:
                raise ValueError(
                    "UCMKvBridgeHybridConnector only supports FullAttentionSpec and "
                    f"MambaSpec(mamba_cache_mode='all'); group={group_id}, spec={spec}"
                )

            info = MambaAllGroupInfo(
                group_id=group_id,
                block_size=block_size_from_kv_cache_spec(spec),
                layer_names=tuple(group.layer_names),
                seed=request_hasher((b"UCM_GROUP_SEED", base_seed, group_id)),
                is_mamba_all=is_mamba_all,
            )
            self.groups_by_id.append(info)
            if is_mamba_all:
                self.state_groups.append(info)
            else:
                self.full_attn_groups.append(info)

        if not self.full_attn_groups:
            raise ValueError("UCMKvBridgeHybridConnector requires at least one full-attention group")
        if not self.state_groups:
            raise ValueError("UCMKvBridgeHybridConnector requires at least one Mamba all-mode group")

        block_sizes = [g.block_size for g in self.groups_by_id]
        self.lcm_block_size = math.lcm(*block_sizes)

        logger.info(
            "MambaAllGroupManager initialized: lcm_block_size=%s, full_attn=%s, mamba_all=%s",
            self.lcm_block_size,
            [(g.group_id, g.block_size) for g in self.full_attn_groups],
            [(g.group_id, g.block_size) for g in self.state_groups],
        )

    @property
    def num_groups(self) -> int:
        return len(self.groups_by_id)

    def compute_block_hashes(
        self, group: MambaAllGroupInfo, token_ids: list[int]
    ) -> list[bytes]:
        result: list[bytes] = []
        parent = group.seed
        block_size = group.block_size

        for start in range(0, len(token_ids), block_size):
            block_tokens = token_ids[start : start + block_size]
            if len(block_tokens) != block_size:
                break
            parent = self.request_hasher((parent, tuple(block_tokens)))
            result.append(parent)
        return result

    def compute_all_group_block_ids(self, token_ids: list[int]) -> list[list[bytes]]:
        return [self.compute_block_hashes(g, token_ids) for g in self.groups_by_id]

    def lookup_external_hit_tokens(
        self,
        num_computed_tokens: int,
        group_block_ids: list[list[bytes]],
        lookup_on_prefix: Callable[[list[bytes]], int],
        lookup_on_reverse: Callable[[list[bytes]], int],
    ) -> tuple[int, int, list[bytes]]:
        del lookup_on_reverse  # all mode never needs align-mode reverse checkpoint lookup

        if num_computed_tokens % self.lcm_block_size != 0:
            raise ValueError(
                f"num_computed_tokens={num_computed_tokens} is not aligned to "
                f"lcm_block_size={self.lcm_block_size}"
            )
        if len(group_block_ids) != self.num_groups:
            raise ValueError(
                f"group_block_ids={len(group_block_ids)} does not match "
                f"num_groups={self.num_groups}"
            )

        candidates: list[int] = []
        for group in self.groups_by_id:
            hashes = group_block_ids[group.group_id]
            hbm_blocks = num_computed_tokens // group.block_size
            external_hashes = hashes[hbm_blocks:]
            if not external_hashes:
                candidates.append(0)
                continue

            try:
                hit_blocks = lookup_on_prefix(external_hashes) + 1
            except Exception as exc:
                logger.error(
                    "all-mode prefix lookup failed for group=%s: %s: %s",
                    group.group_id,
                    type(exc).__name__,
                    exc,
                )
                _record_counter("connector_lookup_errors_total")
                candidates.append(0)
                continue

            candidates.append(max(hit_blocks, 0) * group.block_size)

        if not candidates:
            return 0, 0, []

        external_hit_tokens = min(candidates)
        external_hit_tokens = (
            external_hit_tokens // self.lcm_block_size
        ) * self.lcm_block_size
        if external_hit_tokens <= 0:
            return 0, 0, []

        total_hit_tokens = num_computed_tokens + external_hit_tokens
        mamba_prefetch_hashes: list[bytes] = []
        for group in self.state_groups:
            end_block = total_hit_tokens // group.block_size
            mamba_prefetch_hashes.extend(
                group_block_ids[group.group_id][:end_block]
            )

        return (
            external_hit_tokens,
            external_hit_tokens // self.lcm_block_size,
            mamba_prefetch_hashes,
        )


class UCMKvBridgeHybridConnector(UCMHybridLinearAttentionConnector):
    """UCM connector for hybrid full-attention + Mamba/GDN all mode.

    The worker-side transfer implementation and hybrid physical layout are
    inherited from UCMHybridLinearAttentionConnector. Scheduler-side hashing,
    lookup and dispatch are replaced with block-granular all-mode semantics.
    """

    @classmethod
    def supports_kv_cache_layout(cls, kv_cache_config) -> bool:
        if kv_cache_config is None:
            return False
        if (
            current_platform.device_type != "npu"
            and not current_platform.is_cuda_alike()
        ):
            return False

        layer_to_specs = layer_name_to_kv_cache_spec(kv_cache_config)
        for raw_tensor in kv_cache_config.kv_cache_tensors:
            shared_specs = [
                spec
                for layer_name in raw_tensor.shared_by
                for spec in layer_to_specs.get(layer_name, [])
            ]
            has_attention = any(
                isinstance(spec, FullAttentionSpec) for spec in shared_specs
            )
            has_mamba_all = any(
                isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "all"
                for spec in shared_specs
            )
            if has_attention and has_mamba_all:
                return True
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        # Reuse HLA worker/store/layout initialization, then replace the
        # scheduler-side group manager with all-mode semantics.
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        # block id 0 is a real cache block in all mode. It is NOT the null
        # placeholder used by mamba-align block tables.
        self._skip_null_vllm_blocks = False

        if role == KVConnectorRole.SCHEDULER:
            self.group_manager = MambaAllGroupManager(
                kv_cache_config=kv_cache_config,
                request_hasher=self.request_hasher,
                base_seed=self._seed,
            )
            self.block_size = self.group_manager.lcm_block_size
            self.hash_block_size = self.group_manager.lcm_block_size

        logger.info("%s initialized for mamba_cache_mode=all", type(self).__name__)

    @staticmethod
    def _append_block_range(
        dst_ucm_ids: list[bytes],
        dst_vllm_ids: list[int],
        src_ucm_ids: list[bytes],
        src_vllm_ids: list[int],
        start_block: int,
        end_block: int,
        *,
        request_id: str,
        group_id: int,
        reason: str,
    ) -> None:
        if start_block >= end_block:
            return

        ucm_slice = src_ucm_ids[start_block:end_block]
        vllm_slice = src_vllm_ids[start_block:end_block]
        if len(ucm_slice) != len(vllm_slice):
            raise RuntimeError(
                "Mamba all-mode dispatch range mismatch: "
                f"request_id={request_id}, group_id={group_id}, reason={reason}, "
                f"range=[{start_block}, {end_block}), "
                f"ucm_blocks={len(ucm_slice)}, vllm_blocks={len(vllm_slice)}"
            )

        # Do not filter vLLM block id 0 here: it is a valid physical block.
        dst_ucm_ids.extend(ucm_slice)
        dst_vllm_ids.extend(vllm_slice)

    def _generate_hla_dispatch_meta(
        self,
        req_meta: HLARequestMeta,
        new_tokens: int,
        new_vllm_block_ids_per_group: tuple[list[int], ...],
        need_load: bool = True,
        request_id: str = "",
        incoming_block_ids_are_full: bool = False,
    ) -> RequestDispatchMeta:
        manager: Optional[MambaAllGroupManager] = self.group_manager  # type: ignore[assignment]
        assert manager is not None

        if len(new_vllm_block_ids_per_group) != manager.num_groups:
            raise ValueError(
                f"new block group count={len(new_vllm_block_ids_per_group)} "
                f"does not match connector groups={manager.num_groups}"
            )

        # Keep the complete worker block table in request metadata. Depending on
        # the scheduler output, the incoming IDs may be either a full table or
        # only newly allocated suffix blocks.
        for gid in range(manager.num_groups):
            incoming = list(new_vllm_block_ids_per_group[gid])
            existing = req_meta.group_vllm_block_ids[gid]
            if incoming_block_ids_are_full:
                req_meta.group_vllm_block_ids[gid] = incoming
            elif not existing:
                req_meta.group_vllm_block_ids[gid] = incoming
            elif incoming:
                suffix_len = len(incoming)
                if existing[-suffix_len:] != incoming:
                    existing.extend(incoming)

        load_ucm_ids: list[bytes] = []
        load_vllm_ids: list[int] = []
        dump_ucm_ids: list[bytes] = []
        dump_vllm_ids: list[int] = []

        lcm = manager.lcm_block_size
        external_hit_lcm_blocks = (
            req_meta.total_hit_block_num - req_meta.hbm_hit_block_num
        )
        hbm_hit_tokens = req_meta.hbm_hit_block_num * lcm
        total_hit_tokens = req_meta.total_hit_block_num * lcm

        if need_load and external_hit_lcm_blocks > 0:
            # Keep full-attention blocks first because the inherited HLA worker
            # uses this boundary for MLA rank scoping.
            for group in manager.full_attn_groups:
                self._append_block_range(
                    load_ucm_ids,
                    load_vllm_ids,
                    req_meta.group_ucm_block_ids[group.group_id],
                    req_meta.group_vllm_block_ids[group.group_id],
                    hbm_hit_tokens // group.block_size,
                    total_hit_tokens // group.block_size,
                    request_id=request_id,
                    group_id=group.group_id,
                    reason="load-attention",
                )
            load_full_attn_count = len(load_ucm_ids) if self.is_mla else 0

            # In all mode every matched Mamba block is materialized. Loading all
            # of them keeps the local prefix cache valid for later HBM reuse.
            for group in manager.state_groups:
                self._append_block_range(
                    load_ucm_ids,
                    load_vllm_ids,
                    req_meta.group_ucm_block_ids[group.group_id],
                    req_meta.group_vllm_block_ids[group.group_id],
                    hbm_hit_tokens // group.block_size,
                    total_hit_tokens // group.block_size,
                    request_id=request_id,
                    group_id=group.group_id,
                    reason="load-mamba-all",
                )
        else:
            load_full_attn_count = 0

        if req_meta.token_processed < req_meta.num_token_ids:
            dump_start = req_meta.token_processed
            dump_end = min(
                req_meta.token_processed + new_tokens,
                req_meta.num_token_ids,
            )

            for group in manager.full_attn_groups:
                self._append_block_range(
                    dump_ucm_ids,
                    dump_vllm_ids,
                    req_meta.group_ucm_block_ids[group.group_id],
                    req_meta.group_vllm_block_ids[group.group_id],
                    dump_start // group.block_size,
                    dump_end // group.block_size,
                    request_id=request_id,
                    group_id=group.group_id,
                    reason="dump-attention",
                )
            dump_full_attn_count = len(dump_ucm_ids) if self.is_mla else 0

            # The all-mode kernels are responsible for writing every completed
            # block-boundary state into the corresponding vLLM state block.
            for group in manager.state_groups:
                self._append_block_range(
                    dump_ucm_ids,
                    dump_vllm_ids,
                    req_meta.group_ucm_block_ids[group.group_id],
                    req_meta.group_vllm_block_ids[group.group_id],
                    dump_start // group.block_size,
                    dump_end // group.block_size,
                    request_id=request_id,
                    group_id=group.group_id,
                    reason="dump-mamba-all",
                )
        else:
            dump_full_attn_count = 0

        req_meta.token_processed += new_tokens

        return RequestDispatchMeta(
            (load_ucm_ids, load_vllm_ids),
            (dump_ucm_ids, dump_vllm_ids)
        )

