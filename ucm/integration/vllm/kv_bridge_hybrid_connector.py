import math
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.forward_context import ForwardContext
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from ucm.integration.vllm.hla_connector import (
    HLARequestDispatchMeta,
    HLARequestMeta,
    UCMHybridLinearAttentionConnector,
    block_size_from_kv_cache_spec,
    layer_name_to_kv_cache_spec,
)
from ucm.integration.vllm.ucm_connector import (
    RequestDispatchMeta,
    RequestHasher,
    UCMDirectConnector,
    _record_counter,
)
from ucm.logger import init_logger

logger = init_logger(__name__)

MIN_CHUNK_LEN = 512
MAX_CLUSTER_NUM = 100000


class RotaryEmbedding3D:
    """Relative half-split RoPE from the original Full Attention KVB code.

    ``head_dim`` is the rotary dimension, which may be smaller than the key
    head dimension. YaRN changes frequencies only: correction never applies
    mscale a second time to an already-rotated key.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
        device="cpu",
        *,
        use_yarn: bool = False,
        scaling_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        extrapolation_factor: float = 1.0,
    ):
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE rotary dimension must be positive and even")
        if max_position <= 0 or base <= 1 or scaling_factor <= 0:
            raise ValueError("invalid RoPE position, base or scaling factor")
        if beta_fast <= 0 or beta_slow <= 0:
            raise ValueError("YaRN beta values must be positive")
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_yarn = use_yarn
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.extrapolation_factor = extrapolation_factor
        self._build_cache()

    def _yarn_linear_ramp_mask(self, min_v: float, max_v: float, d: int):
        if min_v == max_v:
            max_v += 0.001
        linear_func = (
            torch.arange(d, dtype=torch.float32, device=self.device) - min_v
        ) / (max_v - min_v)
        return torch.clamp(linear_func, 0, 1)

    def _compute_inv_freq(self):
        pos_freqs = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
            / self.head_dim
        )
        inv_freq_standard = 1.0 / pos_freqs
        if not self.use_yarn:
            return inv_freq_standard
        inv_freq_extrapolation = inv_freq_standard
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)
        low = math.floor(
            self.head_dim
            * math.log(self.max_position / (self.beta_fast * 2 * math.pi))
            / (2 * math.log(self.base))
        )
        high = math.ceil(
            self.head_dim
            * math.log(self.max_position / (self.beta_slow * 2 * math.pi))
            / (2 * math.log(self.base))
        )
        low, high = max(low, 0), max(high, 0)
        inv_freq_mask = (
            1 - self._yarn_linear_ramp_mask(low, high, self.head_dim // 2)
        ) * self.extrapolation_factor
        return (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

    def _build_cache(self, cache_len=None):
        inv_freq = self._compute_inv_freq()
        if cache_len is None:
            cache_len = (
                int(self.max_position * self.scaling_factor) + 256
                if self.use_yarn
                else self.max_position + 1
            )
        t = torch.arange(cache_len, dtype=torch.float32, device=self.device)
        freqs = torch.outer(t, inv_freq)
        # Original KVB correction uses pure phase, with no mscale.
        self.cos_sin_cached = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(
            self.dtype
        )

    def _get_cos_sin(self, delta_positions):
        if self.cos_sin_cached.device != delta_positions.device:
            self.device = delta_positions.device
            self.cos_sin_cached = self.cos_sin_cached.to(self.device)
        abs_positions = delta_positions.abs()
        if abs_positions.numel():
            required = int(abs_positions.max()) + 1
            if required > len(self.cos_sin_cached):
                # Extend the lookup table without changing YaRN's original
                # context length or its frequency interpolation parameters.
                self._build_cache(required)
        cos_sin = self.cos_sin_cached[abs_positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        sign = (delta_positions >= 0).to(cos.dtype) * 2 - 1
        sin = sin * sign.unsqueeze(-1)
        return cos.unsqueeze(-2), sin.unsqueeze(-2)

    def shift(self, key, delta_positions):
        """key [..., heads, dim], delta_positions [...], in token positions."""
        if self.head_dim > key.shape[-1]:
            raise ValueError("rotary dimension exceeds key head dimension")
        if delta_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("RoPE relative positions must be integer tensors")
        delta_positions = delta_positions.to(device=key.device, dtype=torch.long)
        orig_dtype = key.dtype
        key_f32 = key[..., : self.head_dim].float()
        cos, sin = self._get_cos_sin(delta_positions)
        cos, sin = cos.float(), sin.float()
        x1, x2 = key_f32.chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        output = torch.cat((o1, o2), dim=-1).to(orig_dtype)
        # The model may rotate only a prefix of each head.
        return torch.cat((output, key[..., self.head_dim :]), dim=-1)


class SingleVllmConfig:
    """Cache the original KVB padding settings per VllmConfig instance."""

    _instances: ClassVar[dict[int, "SingleVllmConfig"]] = {}

    def __new__(cls, vllm_config):
        key = id(vllm_config)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, vllm_config):
        if getattr(self, "_initialized", False):
            return
        from ucm.utils import Config

        self.vllm_config = vllm_config
        launch_config = Config(vllm_config.kv_transfer_config).get_config()
        self.padding_config = launch_config.get("padding_config")
        padding = self.padding_config or {}
        self.chunk_end_token_id = padding.get("chunk_end_token_id", -1)
        self.chunk_pad_token_id = padding.get("chunk_pad_token_id")
        self._initialized = True


def is_token_in_chunk_end(token_id: int, chunk_end_token_id) -> bool:
    if isinstance(chunk_end_token_id, (list, tuple, set)):
        return token_id in chunk_end_token_id
    return chunk_end_token_id is not None and token_id == chunk_end_token_id


def pad_rag_chunks(token_ids: list[int], block_size: int, pad_id: int | None):
    """Pad a completed chunk immediately before its final END token."""
    if not token_ids or pad_id is None:
        return token_ids
    if block_size <= 0:
        raise ValueError("KVB padding block_size must be positive")
    remainder = len(token_ids) % block_size
    if remainder == 0:
        return token_ids
    pad_len = block_size - remainder
    # Keep the last token as a list (the original snippet added an int).
    return token_ids[:-1] + [pad_id] * pad_len + token_ids[-1:]


def kvb_replace_padding(prompt_token_ids: list[int], vllm_config: "VllmConfig"):
    """Align END-terminated chunks; leave an unfinished prompt tail intact.

    Invoke before constructing the scheduler Request and its block hashes.
    Repeated preprocessing is idempotent, including already padded input.
    """
    padding_config = SingleVllmConfig(vllm_config)
    if not padding_config.padding_config or padding_config.chunk_pad_token_id is None:
        return prompt_token_ids
    block_size = vllm_config.cache_config.block_size
    start = 0
    chunks = []
    for end, token in enumerate(prompt_token_ids):
        if is_token_in_chunk_end(token, padding_config.chunk_end_token_id):
            chunks.append(
                pad_rag_chunks(
                    prompt_token_ids[start : end + 1],
                    block_size,
                    padding_config.chunk_pad_token_id,
                )
            )
            start = end + 1
    if start < len(prompt_token_ids):
        chunks.append(prompt_token_ids[start:])
    return [token for chunk in chunks for token in chunk]


class KVCacheLayoutKVB:
    """Store-compatible byte segments with views derived from registered tensors."""

    def __init__(self, layout, num_blocks, device):
        self.layout = layout
        self.num_blocks = num_blocks
        self.buffers = [
            torch.empty((num_blocks, int(n)), dtype=torch.uint8, device=device)
            for n in layout.tensor_size_lists
        ]

    def extract_block_addrs(self, block_ids):
        ids = np.asarray(block_ids, dtype=np.uint64)
        base = np.asarray([b.data_ptr() for b in self.buffers], dtype=np.uint64)
        sizes = np.asarray([b.shape[1] for b in self.buffers], dtype=np.uint64)
        return ids[:, None] * sizes[None, :] + base[None, :]

    def tensor_view(self, layer_name, tensor):
        """tensor must expose its physical block axis as dimension zero."""
        row = self.layout.row_slices[self.layout.layer_name_to_row[layer_name]]
        for index in range(row.start, row.stop):
            offset = tensor.data_ptr() - int(self.layout.base_ptrs[index])
            size = self.buffers[index].shape[1]
            item = tensor.element_size()
            extent = (
                1
                + sum(
                    (n - 1) * s for n, s in zip(tensor.shape[1:], tensor.stride()[1:])
                )
            ) * item
            if offset < 0 or offset + extent > size:
                continue
            if size % item or offset % item:
                continue
            if tensor.shape[0] > 1 and tensor.stride(0) * item != int(
                self.layout.block_stride_lists[index]
            ):
                continue
            raw = self.buffers[index].view(tensor.dtype)
            return raw.as_strided(
                (self.num_blocks, *tensor.shape[1:]),
                (size // item, *tensor.stride()[1:]),
                offset // item,
            )
        raise ValueError(f"{layer_name}: tensor is not described by the store layout")


class KVCluster:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kvcluster_info: "KVClusterInfo",
        req_id="-1",
    ):
        self.kvcluster_info = kvcluster_info
        self.request_hasher = RequestHasher(vllm_config, 0)
        self._seed = self.request_hasher("KV_BRIDGE_HASH_SEED")
        self.cluster_id = self._get_cluster_hash(kvcluster_info.chunk_hash_ids)
        self.request_id = req_id

    def _get_cluster_hash(self, chunk_hash_ids: list[bytes]):
        parent_block_hash_value = self._seed
        for chunk_hash_id in chunk_hash_ids:
            hash_value = self.request_hasher((parent_block_hash_value, chunk_hash_id))
            parent_block_hash_value = hash_value
        return parent_block_hash_value


class KVClusterMap:
    def __init__(self, capacity: int = MAX_CLUSTER_NUM):
        if capacity < 1:
            raise ValueError("cluster capacity must be positive")
        self.capacity = capacity
        self.candidates: OrderedDict[bytes, KVCluster] = OrderedDict()
        self.inverted: dict[bytes, dict[bytes, None]] = defaultdict(dict)

    def insert_cluster(self, cluster: KVCluster | None) -> None:
        if cluster is None or cluster.cluster_id in self.candidates:
            return
        if len(self.candidates) >= self.capacity:
            key, old = self.candidates.popitem(last=False)
            for chunk in set(old.kvcluster_info.chunk_hash_ids):
                self.inverted[chunk].pop(key, None)
                if not self.inverted[chunk]:
                    del self.inverted[chunk]
        self.candidates[cluster.cluster_id] = cluster
        for chunk in set(cluster.kvcluster_info.chunk_hash_ids):
            self.inverted[chunk][cluster.cluster_id] = None

    def search_top_one(self, chunks, lengths, exclude_request_id=""):
        scores: dict[bytes, int] = defaultdict(int)
        for chunk in dict.fromkeys(chunks):
            for key in self.inverted.get(chunk, ()):
                if self.candidates[key].request_id != exclude_request_id:
                    scores[key] += lengths[chunk]
        return max(scores, key=scores.get) if scores else None


@dataclass
class KVClusterInfo:
    chunk_hash_ids: list[bytes] = field(default_factory=list)
    chunk_start_positions: dict[bytes, int] = field(default_factory=dict)
    chunk_lens: dict[bytes, int] = field(default_factory=dict)
    chunk_num: int = 0
    chunk_start_block_ids: dict[bytes, list[bytes]] = field(default_factory=dict)
    chunk_previous_block_ids: dict[bytes, list[bytes | None]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class GroupChunk:
    group_id: int
    block_size: int
    block_ids: tuple[bytes, ...]
    source_offset: int
    # Included as the first loaded block, before the chunk's actual blocks.
    has_previous_state: bool = False


@dataclass(frozen=True)
class LoadChunkMeta:
    chunk_hash: bytes
    source_start: int
    destination_start: int
    length: int
    groups: tuple[GroupChunk, ...]
    # A dispatch may need earlier g values to denoise a later prefill slice.
    output_offset: int = 0
    output_length: int = 0

    @property
    def shift(self):
        return self.destination_start - self.source_start


@dataclass
class KVBRequestMeta(HLARequestMeta):
    new_kv_cluster: KVCluster | None = None
    load_chunk_meta: list[LoadChunkMeta] = field(default_factory=list)


@dataclass
class KVBRequestDispatchMeta(HLARequestDispatchMeta):
    load_chunk_meta: list[LoadChunkMeta] = field(default_factory=list)
    group_vllm_block_ids: list[list[int]] = field(default_factory=list)


def dispatch_chunks(chunks, start, end, alignment):
    result = []
    for chunk in chunks:
        first = max(start, chunk.destination_start)
        last = min(end, chunk.destination_start + chunk.length)
        first = (first + alignment - 1) // alignment * alignment
        last = last // alignment * alignment
        if first >= last:
            continue
        # Retain B's origin and all g up to this step's last endpoint.
        length = last - chunk.destination_start
        groups = tuple(
            replace(
                g,
                block_ids=g.block_ids[
                    : (g.source_offset + length + g.block_size - 1) // g.block_size
                    + int(g.has_previous_state)
                ],
            )
            for g in chunk.groups
        )
        result.append(
            replace(
                chunk,
                length=length,
                groups=groups,
                output_offset=first - chunk.destination_start,
                output_length=last - first,
            )
        )
    return result


def denoise_gdn(states, g, initial_state=None):
    """states [blocks, heads, ...], g [blocks, tokens, heads].

    Implements the all-mode scalar-per-head propagation contract supplied by
    the GDN writer, not a general matrix-transition delta-rule inverse.
    """
    if states.ndim < 3 or g.ndim != 3:
        raise ValueError(
            "expected states [blocks, heads, ...], g [blocks, tokens, heads]"
        )
    if states.shape[:2] != (g.shape[0], g.shape[2]):
        raise ValueError("GDN state and g block/head dimensions disagree")
    if initial_state is None:
        initial_state = torch.zeros_like(states[0], dtype=torch.float32)
    if initial_state.shape != states.shape[1:]:
        raise ValueError("initial GDN state shape disagrees with checkpoint shape")
    decay = g.float().sum(dim=1).cumsum(dim=0).exp()
    decay = decay.reshape(*decay.shape, *([1] * (states.ndim - 2)))
    return states.float() - decay * initial_state.float().unsqueeze(0)


def attention_components(cache, spec, num_blocks):
    if isinstance(cache, (tuple, list)):
        if len(cache) != 2:
            raise ValueError("Full Attention requires separate K and V components")
        parts = cache
    elif cache.ndim == 5 and cache.shape[0] == 2 and cache.shape[1] == num_blocks:
        parts = (cache[0], cache[1])
    elif cache.ndim == 5 and cache.shape[0] == num_blocks and cache.shape[1] == 2:
        parts = (cache[:, 0], cache[:, 1])
    else:
        raise ValueError("unsupported Full Attention cache axes")
    result = []
    for tensor in parts:
        if tensor.ndim != 4 or tensor.shape[0] != num_blocks:
            raise ValueError("expected four-dimensional paged K/V")
        if tensor.shape[1:3] == (spec.block_size, spec.num_kv_heads):
            result.append(tensor)
        elif tensor.shape[1:3] == (spec.num_kv_heads, spec.block_size):
            result.append(tensor.transpose(1, 2))
        else:
            raise ValueError("K/V token/head axes disagree with FullAttentionSpec")
    return tuple(result)


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
        kv_cache_config: KVCacheConfig,
        connector: "UCMKvBridgeHybridConnector",
    ) -> None:
        self.connector = connector
        request_hasher = RequestHasher(connector._vllm_config, 0)
        base_seed = request_hasher.seed
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
            raise ValueError(
                "UCMKvBridgeHybridConnector requires at least one full-attention group"
            )
        if not self.state_groups:
            raise ValueError(
                "UCMKvBridgeHybridConnector requires at least one Mamba all-mode group"
            )

        block_sizes = [g.block_size for g in self.groups_by_id]
        self.lcm_block_size = math.lcm(*block_sizes)
        if any(g.block_size != self.lcm_block_size for g in self.state_groups):
            raise ValueError("KVB requires Mamba block_size == connector block_size")

        logger.info(
            "MambaAllGroupManager initialized: lcm_block_size=%s, full_attn=%s, mamba_all=%s",
            self.lcm_block_size,
            [(g.group_id, g.block_size) for g in self.full_attn_groups],
            [(g.group_id, g.block_size) for g in self.state_groups],
        )

    @property
    def num_groups(self) -> int:
        return len(self.groups_by_id)

    def compute_all_group_block_ids(self, request) -> list[list[bytes]]:
        return [
            self.connector.compute_block_hashes(g, request) for g in self.groups_by_id
        ]

    def lookup_external_hit_tokens(
        self,
        num_computed_tokens: int,
        group_block_ids: list[list[bytes]],
        lookup_on_prefix: Callable[[list[bytes]], int],
        lookup_on_reverse: Callable[[list[bytes]], int],
    ) -> tuple[int, int, list[bytes]]:
        # All mode never needs align-mode reverse checkpoint lookup.
        del lookup_on_reverse

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
            except Exception as exc:  # noqa: BLE001 - store implementations have distinct exception types
                logger.error(
                    "all-mode prefix lookup failed for group=%s: %s: %s",
                    group.group_id,
                    type(exc).__name__,
                    exc,
                )
                _record_counter("connector_lookup_errors_total")
                candidates.append(0)
                continue

            candidates.append(
                max(0, min(hit_blocks, len(external_hashes))) * group.block_size
            )

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
            mamba_prefetch_hashes.extend(group_block_ids[group.group_id][:end_block])

        return (
            external_hit_tokens,
            external_hit_tokens // self.lcm_block_size,
            mamba_prefetch_hashes,
        )


class UCMKvBridgeHybridConnector(UCMHybridLinearAttentionConnector):
    """UCM connector for hybrid full-attention + Mamba/GDN all mode.

    Prefix transfer uses the HLA physical layout. Non-prefix chunks load into
    separate staging pages: FA keys are shifted to their new RoPE positions,
    while GDN checkpoints are denoised without writing destination states.
    Chunk hits never count as scheduler-computed tokens in this phase.
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
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Initialize the shared transfer implementation directly: HLA's ctor
        # constructs an align-mode manager, which misclassifies all-mode states.
        self.group_manager = None
        UCMDirectConnector.__init__(
            self,
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        # block id 0 is a real cache block in all mode. It is NOT the null
        # placeholder used by mamba-align block tables.
        self._skip_null_vllm_blocks = False
        padding = self.launch_config.get("padding_config") or {}
        end = padding.get("chunk_end_token_id", -1)
        self.chunk_end_token_ids = set(end if isinstance(end, (list, tuple)) else [end])
        self.chunk_end_token_ids.discard(None)
        self.chunk_end_token_ids.discard(-1)
        self.chunk_pad_token_id = padding.get("chunk_pad_token_id")
        self.kv_cluster_map = KVClusterMap(MAX_CLUSTER_NUM)
        self._kvb_staging = None
        self.kvb_denoised_states = {}
        self._kvb_rope = {}
        self._kvb_processed_metadata = None

        if role == KVConnectorRole.SCHEDULER:
            self.group_manager = MambaAllGroupManager(
                kv_cache_config=kv_cache_config,
                connector=self,
            )
            self.block_size = self.group_manager.lcm_block_size
            self.hash_block_size = self.group_manager.lcm_block_size
            self._bind_request_block_hasher()

        logger.info("%s initialized for mamba_cache_mode=all", type(self).__name__)

    def compute_block_hashes(self, group: MambaAllGroupInfo, request) -> list[bytes]:
        return self.generate_hash(group.block_size, request.all_token_ids, group.seed)

    def generate_chunk_hash(self, token_ids):
        return self.request_hasher((self._seed, tuple(token_ids)))

    def _process_req(self, all_token_ids, group_block_ids, prefix_len, request_id=""):
        info = KVClusterInfo()
        start = prefix_len
        for end in range(prefix_len, len(all_token_ids)):
            if not is_token_in_chunk_end(all_token_ids[end], self.chunk_end_token_ids):
                continue
            if end + 1 - start >= MIN_CHUNK_LEN:
                key = self.generate_chunk_hash(all_token_ids[start : end + 1])
                info.chunk_hash_ids.append(key)
                # Preserve the original hash-keyed last-occurrence contract.
                info.chunk_start_positions[key] = start
                info.chunk_lens[key] = end + 1 - start
                info.chunk_num += 1
                info.chunk_start_block_ids[key] = [
                    group_block_ids[g.group_id][start // g.block_size]
                    for g in self.group_manager.groups_by_id
                ]
                info.chunk_previous_block_ids[key] = [
                    group_block_ids[g.group_id][start // g.block_size - 1]
                    if g.is_mamba_all and start // g.block_size > 0
                    else None
                    for g in self.group_manager.groups_by_id
                ]
            start = end + 1
        if not info.chunk_hash_ids:
            return None, None
        best_id = self.kv_cluster_map.search_top_one(
            info.chunk_hash_ids, info.chunk_lens, request_id
        )
        return info, best_id

    def get_kvbridge_matched_tokens(self, request, group_block_ids, prefix_hit_len):
        info, best_id = self._process_req(
            request.all_token_ids, group_block_ids, prefix_hit_len, request.request_id
        )
        if info is None:
            return None, []
        cluster = KVCluster(self._vllm_config, info, request.request_id)
        hits = []
        if best_id is not None:
            hits = self.get_hit_meta(
                request, cluster, self.kv_cluster_map.candidates[best_id]
            )
        return cluster, hits

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        matched, async_load = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        meta = self.requests_meta.get(request.request_id)
        if meta is None:
            return matched, async_load
        meta = KVBRequestMeta(
            **{
                name: getattr(meta, name)
                for name in HLARequestMeta.__dataclass_fields__
            }
        )
        self.requests_meta[request.request_id] = meta
        if not self.chunk_end_token_ids:
            return matched, async_load
        meta.new_kv_cluster, meta.load_chunk_meta = self.get_kvbridge_matched_tokens(
            request,
            meta.group_ucm_block_ids,
            num_computed_tokens + matched,
        )
        if meta.load_chunk_meta:
            logger.info(
                "KVB request=%s matched_chunks=%s matched_tokens=%s (normal compute enabled)",
                request.request_id,
                len(meta.load_chunk_meta),
                sum(chunk.length for chunk in meta.load_chunk_meta),
            )
        return matched, async_load

    def get_hit_meta(self, request, new_cluster, best_cluster):
        def lookup(ids):
            try:
                return self._rank_consistency.lookup_on_prefix(self.store, ids)
            except Exception as exc:  # noqa: BLE001 - lookup failure must not disable normal compute
                logger.warning("KVB cache lookup failed: %s", exc)
                _record_counter("connector_lookup_errors_total")
                return -1

        result = []
        current, old = new_cluster.kvcluster_info, best_cluster.kvcluster_info
        for key in dict.fromkeys(current.chunk_hash_ids):
            if key not in old.chunk_lens:
                continue
            dst = current.chunk_start_positions[key]
            src = old.chunk_start_positions[key]
            lead = (-dst) % self.block_size
            src, dst = src + lead, dst + lead
            length = min(current.chunk_lens[key], old.chunk_lens[key]) - lead
            length = length // self.block_size * self.block_size
            if length <= 0:
                continue
            # A checkpoint at an interior token cannot be recovered from a block.
            if any(
                g.is_mamba_all and src % g.block_size
                for g in self.group_manager.groups_by_id
            ):
                continue
            verified = []
            for group in self.group_manager.groups_by_id:
                bs = group.block_size
                first = src // bs
                offset = src % bs
                original_src = old.chunk_start_positions[key]
                anchor_index = original_src // bs
                anchor = old.chunk_start_block_ids[key][group.group_id]
                next_block_offset = (anchor_index + 1) * bs - original_src
                new_start = current.chunk_start_positions[key]
                tokens = request.all_token_ids[
                    new_start + next_block_offset : new_start + old.chunk_lens[key]
                ]
                hashes = [anchor] + list(self.generate_hash(bs, tokens, anchor))
                skip = first - anchor_index
                count = (offset + length + bs - 1) // bs
                ids = hashes[skip : skip + count]
                previous = group.is_mamba_all and first > 0
                if not ids:
                    length = 0
                    break
                if previous:
                    preceding = (
                        hashes[skip - 1]
                        if skip > 0
                        else old.chunk_previous_block_ids[key][group.group_id]
                    )
                    if preceding is None:
                        length = 0
                        break
                    ids = [preceding] + ids
                # The preceding checkpoint must exist too. Prefix lookup therefore
                # covers [SA, B0, B1, ...] for nonzero-start Mamba chunks.
                hits = max(0, min(len(ids), lookup(list(ids)) + 1))
                available = max(0, (hits - int(previous)) * bs - offset)
                length = min(length, available) // self.block_size * self.block_size
                if length <= 0:
                    break
                verified.append(
                    GroupChunk(group.group_id, bs, tuple(ids[:hits]), offset, previous)
                )
            if length > 0 and len(verified) == self.group_manager.num_groups:
                trimmed = tuple(
                    replace(
                        g,
                        block_ids=g.block_ids[
                            : (g.source_offset + length + g.block_size - 1)
                            // g.block_size
                            + int(g.has_previous_state)
                        ],
                    )
                    for g in verified
                )
                result.append(LoadChunkMeta(key, src, dst, length, trimmed))
        return result

    def build_connector_meta(self, scheduler_output):
        cached = scheduler_output.scheduled_cached_reqs
        if isinstance(cached, list):
            entries = ((r.req_id, getattr(r, "num_output_tokens", 0)) for r in cached)
        else:
            entries = zip(
                cached.req_ids,
                getattr(cached, "num_output_tokens", [0] * len(cached.req_ids)),
            )
        for request_id, outputs in entries:
            meta = self.requests_meta.get(request_id)
            if outputs > 0 and isinstance(meta, KVBRequestMeta):
                self.kv_cluster_map.insert_cluster(meta.new_kv_cluster)
        return super().build_connector_meta(scheduler_output)

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
        if (
            start_block < 0
            or len(ucm_slice) != end_block - start_block
            or len(vllm_slice) != end_block - start_block
        ):
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
        manager: MambaAllGroupManager | None = self.group_manager  # type: ignore[assignment]
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
            if incoming_block_ids_are_full or not existing:
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

        step_start = req_meta.token_processed
        step_end = min(step_start + new_tokens, req_meta.num_token_ids)
        chunks = dispatch_chunks(
            getattr(req_meta, "load_chunk_meta", []), step_start, step_end, lcm
        )
        req_meta.token_processed += new_tokens

        return KVBRequestDispatchMeta(
            (load_ucm_ids, load_vllm_ids),
            (dump_ucm_ids, dump_vllm_ids),
            load_full_attn_count=load_full_attn_count,
            dump_full_attn_count=dump_full_attn_count,
            load_chunk_meta=chunks,
            group_vllm_block_ids=[list(ids) for ids in req_meta.group_vllm_block_ids],
        )

    def _get_kvb_rope(self, layer_name, spec, device):
        if layer_name in self._kvb_rope:
            return self._kvb_rope[layer_name]
        hf = self._vllm_config.model_config.hf_text_config
        params = getattr(hf, "rope_parameters", None)
        if params is not None and "full_attention" in params:
            params = params["full_attention"]
        scaling = getattr(hf, "rope_scaling", None)
        params = dict(params or scaling or {})
        params.setdefault("rope_theta", getattr(hf, "rope_theta", 10000.0))
        params.setdefault(
            "partial_rotary_factor", getattr(hf, "partial_rotary_factor", 1.0)
        )
        params.setdefault("rope_type", params.get("type", "default"))
        if params["rope_type"] not in ("default", "yarn"):
            raise ValueError("RotaryEmbedding3D supports default and YaRN RoPE only")
        fraction = params["partial_rotary_factor"]
        max_position = getattr(
            hf, "max_position_embeddings", self._vllm_config.model_config.max_model_len
        )
        use_yarn = params["rope_type"] == "yarn"
        if use_yarn:
            max_position = params.get("original_max_position_embeddings", max_position)
        rope = RotaryEmbedding3D(
            head_dim=params.get("rope_dim", int(spec.head_size * fraction)),
            max_position=max_position,
            base=params["rope_theta"],
            dtype=torch.float32,
            device=device,
            use_yarn=use_yarn,
            scaling_factor=params.get("factor", 1.0),
            beta_fast=params.get("beta_fast", 32.0),
            beta_slow=params.get("beta_slow", 1.0),
            extrapolation_factor=params.get("extrapolation_factor", 1.0),
        )
        self._kvb_rope[layer_name] = rope
        return rope

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if metadata is self._kvb_processed_metadata:
            return
        super().start_load_kv(forward_context, **kwargs)
        self._kvb_processed_metadata = metadata
        self.kvb_denoised_states.clear()
        jobs = []
        count = 0
        for request_id, request in metadata.request_meta.items():
            for chunk_index, chunk in enumerate(
                getattr(request, "load_chunk_meta", [])
            ):
                for group in chunk.groups:
                    ids = list(range(count, count + len(group.block_ids)))
                    jobs.append((request_id, chunk_index, request, chunk, group, ids))
                    count += len(ids)
        if not jobs:
            return
        first = next(iter(self.kv_caches.values()))
        device = first.device if isinstance(first, torch.Tensor) else first[0].device
        if self._kvb_staging is not None:
            self.device.synchronize()
        self._kvb_staging = KVCacheLayoutKVB(self.kv_cache_layout, count, device)
        # Make allocator reuse safe before a store-owned stream starts writing.
        self.device.synchronize()
        pending, failed = [], set()
        for job in jobs:
            request_id, chunk_index, request, chunk, group, ids = job
            spec = self._kv_cache_config.kv_cache_groups[group.group_id].kv_cache_spec
            fa_count = len(ids) if _is_full_attention_group(spec) and self.is_mla else 0
            keys = list(group.block_ids)
            _, scoped, scoped_ids = self._scope_blocks(
                keys, ids, fa_count, is_dump=False
            )
            try:
                ptrs = self._kvb_staging.extract_block_addrs(scoped_ids)
                task = self._rank_consistency.submit_load(
                    self.store, {request_id: keys}, scoped, [0] * len(scoped), ptrs
                )
                pending.append((job, task))
            except Exception as exc:  # noqa: BLE001 - drain other tasks after any store failure
                failed.add((request_id, chunk_index))
                logger.warning(
                    "KVB submit failed for %s chunk %s: %s",
                    request_id,
                    chunk_index,
                    exc,
                )
        completed = []
        # Drain every submitted task even after a different group fails.
        for job, task in pending:
            try:
                self._rank_consistency.wait_load(task)
                completed.append(job)
            except Exception as exc:  # noqa: BLE001 - every submitted task must be waited
                failed.add((job[0], job[1]))
                logger.warning(
                    "KVB wait failed for %s chunk %s: %s", job[0], job[1], exc
                )
        for job in completed:
            if (job[0], job[1]) in failed:
                continue
            try:
                self._copy_temp_to_kvcaches(job, forward_context)
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning(
                    "KVB postprocess failed for %s chunk %s: %s", job[0], job[1], exc
                )

    def _copy_temp_to_kvcaches(
        self, job, forward_context: "ForwardContext" = None
    ) -> None:
        _request_id, _chunk_index, request, chunk, group, ids = job
        cache_group = self._kv_cache_config.kv_cache_groups[group.group_id]
        spec_map = layer_name_to_kv_cache_spec(self._kv_cache_config)
        for name in cache_group.layer_names:
            if name not in self.kv_caches:
                continue  # pipeline-parallel layers on another worker
            spec = spec_map[name][0]
            if isinstance(spec, MambaSpec):
                self._denoise_mamba_chunk(name, job)
                continue
            if not isinstance(spec, FullAttentionSpec) or isinstance(
                spec, MLAAttentionSpec
            ):
                raise TypeError(
                    "KVB postprocess currently requires non-MLA FullAttentionSpec"
                )
            key, value = attention_components(
                self.kv_caches[name], spec, self._kv_cache_config.num_blocks
            )
            start = chunk.destination_start + chunk.output_offset
            length = chunk.output_length
            if start % group.block_size or length <= 0 or length % group.block_size:
                raise ValueError("KVB destination must cover complete aligned blocks")
            first_block = start // group.block_size
            blocks = request.group_vllm_block_ids[group.group_id]
            n = length // group.block_size
            if first_block < 0 or first_block + n > len(blocks):
                raise ValueError("KVB destination outside allocated block table")
            dst_ids = blocks[first_block : first_block + n]
            offset = group.source_offset + chunk.output_offset
            staging = self._kvb_staging
            temp_k = staging.tensor_view(name, key)[ids[0] : ids[-1] + 1]
            temp_v = staging.tensor_view(name, value)[ids[0] : ids[-1] + 1]
            src_k = temp_k.reshape(-1, *key.shape[2:])[offset : offset + length]
            src_v = temp_v.reshape(-1, *value.shape[2:])[offset : offset + length]
            if chunk.shift:
                delta_positions = torch.full(
                    (length,), chunk.shift, dtype=torch.long, device=key.device
                )
                src_k = self._get_kvb_rope(name, spec, key.device).shift(
                    src_k, delta_positions
                )
            # Compute/validate both views before writing either target tensor.
            src_k = src_k.reshape(n, group.block_size, *key.shape[2:])
            src_v = src_v.reshape(n, group.block_size, *value.shape[2:])
            key[dst_ids] = src_k
            value[dst_ids] = src_v

    def _denoise_mamba_chunk(self, name, job):
        request_id, chunk_index, _request, chunk, group, ids = job
        # Writer contract: conv, recurrent state, then token-wise log decay g.
        # Views supply offsets, strides and dtypes; no byte offsets are guessed.
        _conv, state, g = self.kv_caches[name]
        if g.shape[1:] != (group.block_size, state.shape[1]):
            raise ValueError("registered g must have shape [blocks, block_size, heads]")
        state_view = self._kvb_staging.tensor_view(name, state)[ids[0] : ids[-1] + 1]
        g_view = self._kvb_staging.tensor_view(name, g)[ids[0] : ids[-1] + 1]
        preceding = int(group.has_previous_state)
        initial = state_view[0] if preceding else None
        clean = denoise_gdn(state_view[preceding:], g_view[preceding:], initial)
        first = chunk.output_offset // group.block_size
        last = first + chunk.output_length // group.block_size
        self.kvb_denoised_states[(request_id, chunk_index, name)] = {
            "states": clean[first:last],
            "source_start": chunk.source_start + chunk.output_offset,
            "destination_start": chunk.destination_start + chunk.output_offset,
            "length": chunk.output_length,
        }

    def clear_connector_metadata(self):
        # Postprocess kernels may still reference staging views on the compute
        # stream. Keep allocations alive until those kernels have completed.
        if self._kvb_staging is not None:
            self.device.synchronize()
        self._kvb_staging = None
        self.kvb_denoised_states.clear()
        self._kvb_processed_metadata = None
        super().clear_connector_metadata()
