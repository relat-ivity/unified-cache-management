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
    """沿用原 KVB 的半维拆分 RoPE，对历史 K 做相对位置旋转。"""

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
        """校验旋转参数并初始化相位缓存，head_dim 可小于完整 head 维度。"""
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
        """生成 YaRN 插值与外推频率之间的线性过渡权重。"""
        if min_v == max_v:
            max_v += 0.001
        linear_func = (
            torch.arange(d, dtype=torch.float32, device=self.device) - min_v
        ) / (max_v - min_v)
        return torch.clamp(linear_func, 0, 1)

    def _compute_inv_freq(self):
        """计算 RoPE 逆频率，启用 YaRN 时混合插值与外推频率。"""
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
        """按相对位置生成 cos/sin 缓存，供历史 K 的位置修正使用。"""
        inv_freq = self._compute_inv_freq()
        if cache_len is None:
            cache_len = (
                int(self.max_position * self.scaling_factor) + 256
                if self.use_yarn
                else self.max_position + 1
            )
        t = torch.arange(cache_len, dtype=torch.float32, device=self.device)
        freqs = torch.outer(t, inv_freq)
        # 相对位置修正只使用相位，不对已旋转的 K 再次应用 mscale。
        self.cos_sin_cached = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(
            self.dtype
        )

    def _get_cos_sin(self, delta_positions):
        """查询相对位置的 cos/sin，并通过 sin 符号支持负向位移。"""
        if self.cos_sin_cached.device != delta_positions.device:
            self.device = delta_positions.device
            self.cos_sin_cached = self.cos_sin_cached.to(self.device)
        abs_positions = delta_positions.abs()
        if abs_positions.numel():
            required = int(abs_positions.max()) + 1
            if required > len(self.cos_sin_cached):
                # 仅扩展相位缓存，保留 YaRN 原始上下文长度和频率参数。
                self._build_cache(required)
        cos_sin = self.cos_sin_cached[abs_positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        sign = (delta_positions >= 0).to(cos.dtype) * 2 - 1
        # cos 对正负位移相同，sin 根据位移方向调整符号。
        sin = sin * sign.unsqueeze(-1)
        return cos.unsqueeze(-2), sin.unsqueeze(-2)

    def shift(self, key, delta_positions):
        """按 token 位置差旋转 K，key 形状为 [..., heads, dim]。"""
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
        # 仅修正 head 的旋转部分，其余维度保持原值。
        return torch.cat((output, key[..., self.head_dim :]), dim=-1)


class SingleVllmConfig:
    """按 VllmConfig 实例缓存 chunk 的结束标记和补齐配置。"""

    _instances: ClassVar[dict[int, "SingleVllmConfig"]] = {}

    def __new__(cls, vllm_config):
        """同一个配置对象复用同一个补齐配置实例。"""
        key = id(vllm_config)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, vllm_config):
        """首次使用时读取补齐配置，后续初始化调用直接返回。"""
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
    """判断 token 是否为 chunk 结束标记，支持单个或多个结束 ID。"""
    if isinstance(chunk_end_token_id, (list, tuple, set)):
        return token_id in chunk_end_token_id
    return chunk_end_token_id is not None and token_id == chunk_end_token_id


def pad_rag_chunks(token_ids: list[int], block_size: int, pad_id: int | None):
    """在 END token 前插入 pad，使已结束 chunk 的长度对齐 block。"""
    if not token_ids or pad_id is None:
        return token_ids
    if block_size <= 0:
        raise ValueError("KVB padding block_size must be positive")
    remainder = len(token_ids) % block_size
    if remainder == 0:
        return token_ids
    pad_len = block_size - remainder
    # 保留 END 为列表切片，pad 插在 END 前面。
    return token_ids[:-1] + [pad_id] * pad_len + token_ids[-1:]


def kvb_replace_padding(prompt_token_ids: list[int], vllm_config: "VllmConfig"):
    """在构造请求和 hash 前补齐已结束 chunk，重复调用不会重复补齐。"""
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
    # 未遇到 END 的尾部保留原样，不参与 chunk 补齐。
    if start < len(prompt_token_ids):
        chunks.append(prompt_token_ids[start:])
    return [token for chunk in chunks for token in chunk]


class KVCacheLayoutKVB:
    """沿用存储注册的字节布局，为 chunk 加载提供临时缓冲区。"""

    def __init__(self, layout, kvcaches):
        """按注册布局分配 1200 个临时 block，并使用缓存所在设备。"""
        self.layout = layout
        self.load_num_blocks = 1200
        self.num_blocks = self.load_num_blocks
        self.device = None
        for cache in kvcaches.values():
            tensor = cache if isinstance(cache, torch.Tensor) else cache[0]
            self.device = tensor.device
            break
        if self.device is None:
            raise ValueError("KVCacheLayoutKVB requires registered KV caches")
        self.buffers = [
            torch.empty(
                (self.load_num_blocks, int(n)), dtype=torch.uint8, device=self.device
            )
            for n in layout.tensor_size_lists
        ]

    def extract_block_addrs(self, block_ids):
        """返回临时 block 各字节分段的地址，供存储加载使用。"""
        ids = np.asarray(block_ids, dtype=np.uint64)
        base = np.asarray([b.data_ptr() for b in self.buffers], dtype=np.uint64)
        sizes = np.asarray([b.shape[1] for b in self.buffers], dtype=np.uint64)
        return ids[:, None] * sizes[None, :] + base[None, :]

    def tensor_view(self, layer_name, tensor):
        """根据注册张量的偏移和步长构造临时视图，首维必须是物理 block。"""
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
            # 复用原张量的块内步长，首维改为临时布局的 block 步长。
            raw = self.buffers[index].view(tensor.dtype)
            return raw.as_strided(
                (self.num_blocks, *tensor.shape[1:]),
                (size // item, *tensor.stride()[1:]),
                offset // item,
            )
        raise ValueError(f"{layer_name}: tensor is not described by the store layout")


class KVCluster:
    """记录一个请求的 chunk 索引及其有序内容标识。"""
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kvcluster_info: "KVClusterInfo",
        req_id="-1",
    ):
        """由请求的 chunk 信息创建 cluster，并记录来源请求 ID。"""
        self.kvcluster_info = kvcluster_info
        self.request_hasher = RequestHasher(vllm_config, 0)
        self._seed = self.request_hasher("KV_BRIDGE_HASH_SEED")
        self.cluster_id = self._get_cluster_hash(kvcluster_info.chunk_hash_ids)
        self.request_id = req_id

    def _get_cluster_hash(self, chunk_hash_ids: list[bytes]):
        """按 chunk 顺序串联 hash，使 cluster 标识包含内容及顺序。"""
        parent_block_hash_value = self._seed
        for chunk_hash_id in chunk_hash_ids:
            hash_value = self.request_hasher((parent_block_hash_value, chunk_hash_id))
            parent_block_hash_value = hash_value
        return parent_block_hash_value


class KVClusterMap:
    """维护历史 cluster 和从 chunk hash 到 cluster 的倒排索引。"""
    def __init__(self, capacity: int = MAX_CLUSTER_NUM):
        """初始化有容量上限的历史索引，按插入顺序淘汰旧记录。"""
        if capacity < 1:
            raise ValueError("cluster capacity must be positive")
        self.capacity = capacity
        self.candidates: OrderedDict[bytes, KVCluster] = OrderedDict()
        self.inverted: dict[bytes, dict[bytes, None]] = defaultdict(dict)

    def insert_cluster(self, cluster: KVCluster | None) -> None:
        """注册新 cluster，容量不足时同步删除最旧记录和倒排引用。"""
        if cluster is None or cluster.cluster_id in self.candidates:
            return
        # 容量已满时淘汰最早插入的 cluster，同时清理倒排索引。
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
        """按去重后的匹配 token 总数选一个历史 cluster，排除当前请求。"""
        scores: dict[bytes, int] = defaultdict(int)
        for chunk in dict.fromkeys(chunks):
            for key in self.inverted.get(chunk, ()):
                if self.candidates[key].request_id != exclude_request_id:
                    scores[key] += lengths[chunk]
        # 只选一个最高分 cluster，后续缓存缺失不会改选其他候选。
        return max(scores, key=scores.get) if scores else None


@dataclass
class KVClusterInfo:
    """保存 chunk 的位置、长度和各组起始及前置 block hash。"""
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
    """描述一组源缓存的存储 hashes、块内偏移和前置状态需求。"""
    group_id: int
    block_size: int
    block_ids: tuple[bytes, ...]
    source_offset: int
    # 前置状态作为第一个加载 block，后面才是复用范围的实际 blocks。
    has_previous_state: bool = False


@dataclass(frozen=True)
class LoadChunkMeta:
    """描述 chunk 的源目标映射，以及本次 dispatch 使用的输出子范围。"""
    chunk_hash: bytes
    source_start: int
    destination_start: int
    length: int
    groups: tuple[GroupChunk, ...]
    # 本 step 使用后半段时，仍可能需要前面 g 来扣除历史入口贡献。
    output_offset: int = 0
    output_length: int = 0

    @property
    def shift(self):
        """返回目标位置减源位置的 token 位移，用于 FA 的 RoPE 修正。"""
        return self.destination_start - self.source_start


@dataclass
class KVBRequestMeta(HLARequestMeta):
    """在 HLA 请求状态上增加当前 cluster 和 FA、Mamba 命中信息。"""
    new_kv_cluster: KVCluster | None = None
    # FA 按目标 block 对齐，Mamba 按源 block 对齐，命中范围分别保存。
    load_chunk_meta: list[LoadChunkMeta] = field(default_factory=list)
    mamba_load_chunk_meta: list[LoadChunkMeta] = field(default_factory=list)


@dataclass
class KVBRequestDispatchMeta(RequestDispatchMeta):
    """下发本 step 的复用范围、完整物理 block 表和下一步 FA 加载计划。"""
    load_chunk_meta: list[LoadChunkMeta] = field(default_factory=list)
    mamba_load_chunk_meta: list[LoadChunkMeta] = field(default_factory=list)
    group_vllm_block_ids: list[list[int]] = field(default_factory=list)
    fa_lookahead_chunk_meta: list[LoadChunkMeta] = field(default_factory=list)
    reset_kvb_state: bool = False


def build_step_chunk_metadata(chunks, start, end, alignment, *, source_aligned=False):
    """将请求级命中裁到本 step，按目标或源 block 边界生成加载描述。"""
    result = []
    for chunk in chunks:
        first = max(start, chunk.destination_start)
        last = min(end, chunk.destination_start + chunk.length)
        # FA 相对目标零点对齐，Mamba 相对源 block 映射起点对齐。
        origin = chunk.destination_start if source_aligned else 0
        first = origin + (first - origin + alignment - 1) // alignment * alignment
        last = origin + (last - origin) // alignment * alignment
        if first >= last:
            continue
        # 保留原映射起点及本步终点之前的 g，供后处理累计衰减。
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


def build_step_mamba_chunk_metadata(chunks, start, end, block_size, logits_limit):
    """逐个生成与本 step 相交的完整源 block 描述，并保留跨步前缀。"""
    result = []
    for chunk in chunks:
        origin = chunk.destination_start
        first = max(0, (start - origin) // block_size)
        # 短 step 恰好到达已推进终点时仍保留前一块描述，以恢复尚未落池的状态。
        if (
            origin % block_size
            and start > origin
            and (start - origin) % block_size == 0
        ):
            first -= 1
        stop = min(
            chunk.length // block_size, (end - origin + block_size - 1) // block_size
        )
        for index in range(first, max(first, stop)):
            target = origin + index * block_size
            last = target + block_size
            # 保留 logits block；本步非目标块边界结束时保守限制新增复用。
            if last > logits_limit or (
                target >= start
                and end % block_size
                and last > end // block_size * block_size
            ):
                continue
            result.extend(
                build_step_chunk_metadata([chunk], target, last, block_size, source_aligned=True)
            )
    return result


def denoise_gdn(states, g, initial_state=None):
    """按每 head 标量衰减约定扣除源入口状态贡献，不是通用 GDN 逆运算。"""
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
    # g 为 [blocks, tokens, heads] 的原始 log decay，累计后得到各检查点衰减。
    decay = g.float().sum(dim=1).cumsum(dim=0).exp()
    decay = decay.reshape(*decay.shape, *([1] * (states.ndim - 2)))
    # clean[j] = source_state[j] - decay[j] * source_entry，全部用 FP32。
    return states.float() - decay * initial_state.float().unsqueeze(0)


def get_attention_kv_views(cache, spec, num_blocks):
    """兼容分离或合并的 K/V 缓存，统一返回 [blocks, tokens, heads, dim] 视图。"""
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
    """展开统一类型的组配置，返回每层具体缓存规格。"""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return list(spec.kv_cache_specs.values())
    return [spec]


def _is_full_attention_group(spec: KVCacheSpec) -> bool:
    """检查组内所有缓存规格是否均为 Full Attention。"""
    specs = _concrete_specs(spec)
    return bool(specs) and all(isinstance(s, FullAttentionSpec) for s in specs)


def _is_mamba_all_group(spec: KVCacheSpec) -> bool:
    """检查组内所有缓存规格是否均为 all-mode Mamba。"""
    specs = _concrete_specs(spec)
    return bool(specs) and all(
        isinstance(s, MambaSpec) and s.mamba_cache_mode == "all" for s in specs
    )


@dataclass
class MambaAllGroupInfo:
    """保存各缓存组的类型、block 大小、层名和独立 hash 种子。"""
    group_id: int
    block_size: int
    layer_names: tuple[str, ...]
    seed: bytes
    is_mamba_all: bool

    @property
    def is_full_attention(self) -> bool:
        """将非 Mamba 组标记为 FA，组类型已在初始化时校验。"""
        return not self.is_mamba_all


class MambaAllGroupManager:
    """管理 FA 与 all-mode Mamba 各组的 hash 和共同前缀命中。"""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        connector: "UCMKvBridgeHybridConnector",
    ) -> None:
        """划分 FA 与 Mamba 组，并要求所有组使用相同 block 大小。"""
        self.connector = connector
        request_hasher = connector.request_hasher
        base_seed = connector._seed
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

        attention_block_size = self.full_attn_groups[0].block_size
        mamba_block_size = self.state_groups[0].block_size
        if any(g.block_size != attention_block_size for g in self.full_attn_groups):
            raise ValueError("KVB requires all Full Attention block sizes to match")
        if any(g.block_size != mamba_block_size for g in self.state_groups):
            raise ValueError("KVB requires all Mamba block sizes to match")
        if mamba_block_size != attention_block_size:
            raise ValueError("KVB requires mamba_block_size == block_size")
        self.block_size = mamba_block_size
        self.lcm_block_size = self.block_size

        logger.info(
            "MambaAllGroupManager initialized: mamba_block_size=%s, block_size=%s",
            mamba_block_size,
            attention_block_size,
        )

    @property
    def num_groups(self) -> int:
        """返回缓存组数量，供 block 表和 dispatch 检查使用。"""
        return len(self.groups_by_id)

    def compute_all_group_block_ids(self, request) -> list[list[bytes]]:
        """使用各组独立种子，为请求生成所有组的存储 block hashes。"""
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
        # all-mode 逐块检查状态，无需 align-mode 的反向单检查点查询。
        """查询各组外部前缀，取共同命中长度并收集 Mamba 预取 hashes。"""
        del lookup_on_reverse

        if num_computed_tokens % self.block_size != 0:
            raise ValueError(
                f"num_computed_tokens={num_computed_tokens} is not aligned to "
                f"block_size={self.block_size}"
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
            except Exception as exc:  # noqa: BLE001 - 各存储后端的异常类型不同，统一按未命中处理
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

        # 所有 FA 和 Mamba 组都必须命中，取最短的连续前缀。
        external_hit_tokens = min(candidates)
        external_hit_tokens = (external_hit_tokens // self.block_size) * self.block_size
        if external_hit_tokens <= 0:
            return 0, 0, []

        total_hit_tokens = num_computed_tokens + external_hit_tokens
        mamba_prefetch_hashes: list[bytes] = []
        for group in self.state_groups:
            end_block = total_hit_tokens // group.block_size
            mamba_prefetch_hashes.extend(group_block_ids[group.group_id][:end_block])

        return (
            external_hit_tokens,
            external_hit_tokens // self.block_size,
            mamba_prefetch_hashes,
        )


class UCMKvBridgeHybridConnector(UCMHybridLinearAttentionConnector):
    """在 HLA 前缀传输上增加 chunk 匹配、FA 位置修正和 GDN 状态复用。"""

    @classmethod
    def supports_kv_cache_layout(cls, kv_cache_config) -> bool:
        """检查支持的平台及 FA、all-mode Mamba 共享存储张量的布局。"""
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
            if any(
                isinstance(spec, FullAttentionSpec) for spec in shared_specs
            ) and any(
                isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "all"
                for spec in shared_specs
            ):
                return True
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # 直接初始化公共传输实现，避免 HLA 构造函数建立 align-mode 管理器。
        """初始化 all-mode 传输、chunk 索引和 worker 跨步状态容器。"""
        self.group_manager = None
        UCMDirectConnector.__init__(
            self,
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.vllm_config = vllm_config
        self.hf_text_config = vllm_config.model_config.hf_text_config

        # all-mode 的物理 block 0 有效，不能按 align-mode 空占位过滤。
        self._skip_null_vllm_blocks = False
        padding = self.launch_config.get("padding_config") or {}
        end = padding.get("chunk_end_token_id", -1)
        self.chunk_end_token_ids = set(end if isinstance(end, (list, tuple)) else [end])
        self.chunk_end_token_ids.discard(None)
        self.chunk_end_token_ids.discard(-1)
        self.chunk_pad_token_id = padding.get("chunk_pad_token_id")
        self.kv_cluster_map = KVClusterMap(MAX_CLUSTER_NUM)
        self.kv_cache_layout_kvb = None
        self.kvb_denoised_states = {}
        # 跨步状态独立于临时加载视图，不能在每步清理时无条件释放。
        self.kvb_cross_step_states = {}
        self.kvb_next_step_fa_cache = {}
        self._kvb_restored_cross_step_states = {}
        self.rotary_emb_3d = None
        self._kvb_processed_metadata = None

        if role == KVConnectorRole.SCHEDULER:
            self.group_manager = MambaAllGroupManager(
                kv_cache_config=kv_cache_config,
                connector=self,
            )
            self.block_size = self.group_manager.block_size
            self.hash_block_size = self.group_manager.block_size
            self._bind_request_block_hasher()

        logger.info("%s initialized for mamba_cache_mode=all", type(self).__name__)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """注册正常缓存，并建立 chunk 临时布局和 FP32 RoPE 修正缓存。"""
        super().register_kv_caches(kv_caches)
        self.kv_cache_layout_kvb = KVCacheLayoutKVB(
            self.kv_cache_layout, self.kv_caches
        )
        self.rotary_emb_3d = RotaryEmbedding3D(
            head_dim=self.vllm_config.model_config.get_head_size(),
            max_position=getattr(
                self.hf_text_config,
                "max_position_embeddings",
                self.max_model_len,
            ),
            base=getattr(self.hf_text_config, "rope_theta", 1000000.0),
            device=self.kv_cache_layout_kvb.device,
            dtype=torch.float32,
        )

    def compute_block_hashes(self, group: MambaAllGroupInfo, request) -> list[bytes]:
        """沿用 KVB token hash 链，使用当前缓存组的独立种子。"""
        return self.generate_hash(group.block_size, request.all_token_ids, group.seed)

    def generate_chunk_hash(self, token_ids):
        """仅按种子和 chunk 内容生成 hash，支持不同位置的内容匹配。"""
        return self.request_hasher((self._seed, tuple(token_ids)))

    def _process_req(self, all_token_ids, group_block_ids, prefix_len, request_id=""):
        """从前缀命中终点扫描已结束 chunk，建索引并选最佳历史 cluster。"""
        info = KVClusterInfo()
        # 已命中的 prefix 不再参与内容 chunk 匹配。
        start = prefix_len
        for end in range(prefix_len, len(all_token_ids)):
            if not is_token_in_chunk_end(all_token_ids[end], self.chunk_end_token_ids):
                continue
            # 仅登记 END 结束且不少于 512 token 的 chunk。
            if end + 1 - start >= MIN_CHUNK_LEN:
                key = self.generate_chunk_hash(all_token_ids[start : end + 1])
                info.chunk_hash_ids.append(key)
                # 同一 chunk hash 多次出现时，沿用原 KVB 的最后位置记录规则。
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
        """构建当前 cluster，并与最佳历史 cluster 检查 FA、Mamba 命中。"""
        info, best_id = self._process_req(
            request.all_token_ids, group_block_ids, prefix_hit_len, request.request_id
        )
        if info is None:
            return None, [], []
        cluster = KVCluster(self._vllm_config, info, request.request_id)
        fa_hits, mamba_hits = [], []
        if best_id is not None:
            fa_hits, mamba_hits = self.get_hit_meta(
                request, cluster, self.kv_cluster_map.candidates[best_id]
            )
        return cluster, fa_hits, mamba_hits

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """先查 prefix 再查 chunk，chunk 命中不增加返回给 scheduler 的命中数。"""
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
        # 用扩展对象替换 HLA metadata，保留前缀命中和各组 block 信息。
        self.requests_meta[request.request_id] = meta
        if not self.chunk_end_token_ids:
            return matched, async_load
        (meta.new_kv_cluster, meta.load_chunk_meta, meta.mamba_load_chunk_meta) = (
            self.get_kvbridge_matched_tokens(
                request,
                meta.group_ucm_block_ids,
                num_computed_tokens + matched,
            )
        )
        if meta.load_chunk_meta:
            logger.info(
                "KVB request=%s matched_chunks=%s matched_tokens=%s (normal compute enabled)",
                request.request_id,
                len(meta.load_chunk_meta),
                sum(chunk.length for chunk in meta.load_chunk_meta),
            )
        # chunk 命中仅生成候选，不增加 scheduler 认为已计算的 token 数。
        return matched, async_load

    def get_hit_meta(self, request, new_cluster, best_cluster):
        """检查历史缓存，分别返回 FA 和 GDN 的请求级复用范围。"""
        def lookup(ids):
            """查询连续前缀命中，异常按未命中处理以允许正常计算。"""
            try:
                return self._rank_consistency.lookup_on_prefix(self.store, ids)
            except Exception as exc:  # noqa: BLE001 - 查询失败按未命中处理，允许请求继续正常计算
                logger.warning("KVB cache lookup failed: %s", exc)
                _record_counter("connector_lookup_errors_total")
                return -1

        # current 是当前请求索引，old 是选中的历史请求索引。
        current, old = new_cluster.kvcluster_info, best_cluster.kvcluster_info

        def verify_range(key, src, length, groups):
            # src 和 length 以 token 为单位，返回各 group 共同可用的范围。
            """检查所有指定组的源缓存，返回共同可用长度和各组加载描述。"""
            verified = []
            for group in groups:
                bs = group.block_size
                # first 是源 block 下标，offset 是源 block 内的 token 偏移。
                first, offset = divmod(src, bs)
                original_src = old.chunk_start_positions[key]
                anchor_index = original_src // bs
                anchor = old.chunk_start_block_ids[key][group.group_id]
                # 起始 block 的 hash 已保存，从它之后的完整 block 开始续算。
                next_block_offset = (anchor_index + 1) * bs - original_src
                new_start = current.chunk_start_positions[key]
                tokens = request.all_token_ids[
                    new_start + next_block_offset : new_start + old.chunk_lens[key]
                ]
                # 用相同 chunk 的当前 tokens 和历史起始 hash 还原历史 hash 链。
                hashes = [anchor] + list(self.generate_hash(bs, tokens, anchor))
                skip = first - anchor_index
                count = (offset + length + bs - 1) // bs
                # ids 是覆盖源 token 范围的存储 hashes，不是物理 block IDs。
                ids = hashes[skip : skip + count]
                previous = group.is_mamba_all and first > 0
                if not ids:
                    return 0, ()
                if previous:
                    # 非零起点的 GDN 需要前一个 checkpoint 来扣除旧入口状态贡献。
                    preceding = (
                        hashes[skip - 1]
                        if skip > 0
                        else old.chunk_previous_block_ids[key][group.group_id]
                    )
                    if preceding is None:
                        return 0, ()
                    ids = [preceding] + ids
                # lookup 返回连续命中的最后下标，-1 表示未命中，+1 转为块数。
                hits = max(0, min(len(ids), lookup(list(ids)) + 1))
                # 前置 checkpoint 和源 block 内偏移不计入可复用 token 数。
                available = max(0, (hits - int(previous)) * bs - offset)
                # 每组检查后收缩长度，最终所有组都覆盖同一完整 block 范围。
                length = min(length, available) // self.block_size * self.block_size
                if length <= 0:
                    return 0, ()
                verified.append(
                    GroupChunk(group.group_id, bs, tuple(ids[:hits]), offset, previous)
                )
            # 之前检查的组也裁到最终长度，保留必要的前置 checkpoint。
            trimmed = tuple(
                replace(
                    g,
                    block_ids=g.block_ids[
                        : (g.source_offset + length + g.block_size - 1) // g.block_size
                        + int(g.has_previous_state)
                    ],
                )
                for g in verified
            )
            return length, trimmed

        fa_groups = [g for g in self.group_manager.groups_by_id if not g.is_mamba_all]
        state_groups = [g for g in self.group_manager.groups_by_id if g.is_mamba_all]
        fa_hits, mamba_hits = [], []
        # 相同 chunk hash 只检查一次，位置沿用索引中记录的最后一次出现。
        for key in dict.fromkeys(current.chunk_hash_ids):
            if key not in old.chunk_lens:
                continue
            dst = current.chunk_start_positions[key]
            src = old.chunk_start_positions[key]
            # FA 目标起点向上对齐；源起点同步移动，保持 token 内容对应。
            lead = (-dst) % self.block_size
            src, dst = src + lead, dst + lead
            length = min(current.chunk_lens[key], old.chunk_lens[key]) - lead
            # 目标终点向下对齐，例如 block3+30 到 block6+100 只复用块4、5。
            length = length // self.block_size * self.block_size
            if length <= 0:
                continue
            # 先确认所有 FA 组的缓存存在，再保留实际可用的目标对齐范围。
            length, groups = verify_range(key, src, length, fa_groups)
            if not length:
                continue
            fa_hits.append(LoadChunkMeta(key, src, dst, length, groups))
            # GDN 只取 FA 成功范围内的完整源 blocks，目标位置可以带块内偏移。
            first = (src + self.block_size - 1) // self.block_size * self.block_size
            last = (src + length) // self.block_size * self.block_size
            if first >= last:
                continue
            # Mamba 缓存缺失只关闭 GDN 复用，已确认的 FA 命中仍保留。
            mamba_length, groups = verify_range(key, first, last - first, state_groups)
            if mamba_length:
                # 用源范围的相对位移计算 GDN 目标起点，保持源目标内容对应。
                mamba_hits.append(
                    LoadChunkMeta(key, first, dst + first - src, mamba_length, groups)
                )
        return fa_hits, mamba_hits

    def build_connector_meta(self, scheduler_output):
        """注册已产生输出的 cluster，再由父类汇总本轮请求的 dispatch。"""
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
            # 请求首次产生输出后才登记 cluster，重复登记由索引去重。
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
        """校验存储 hash 与物理 block 映射等长，再追加指定区间。"""
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

        # 物理 block 0 是有效缓存地址，不能在追加时过滤。
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
    ) -> KVBRequestDispatchMeta:
        """生成单请求本 step 的前缀加载、保存及 chunk 复用候选计划。"""
        manager: MambaAllGroupManager | None = self.group_manager  # type: ignore[assignment]
        assert manager is not None

        if len(new_vllm_block_ids_per_group) != manager.num_groups:
            raise ValueError(
                f"new block group count={len(new_vllm_block_ids_per_group)} "
                f"does not match connector groups={manager.num_groups}"
            )

        # 请求保留完整物理表；传入完整表时替换，传入新增尾部时追加。
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

        block_size = manager.block_size
        external_hit_lcm_blocks = (
            req_meta.total_hit_block_num - req_meta.hbm_hit_block_num
        )
        hbm_hit_tokens = req_meta.hbm_hit_block_num * block_size
        total_hit_tokens = req_meta.total_hit_block_num * block_size

        # 只加载本地 HBM 命中终点之后、外部总命中终点之前的 prefix。
        if need_load and external_hit_lcm_blocks > 0:
            # 加载列表先放 FA，再放 Mamba all-mode 状态。
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

            # 加载所有命中的 Mamba blocks，保留后续 HBM 前缀复用所需的逐块状态。
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

        # 保存本步补齐的 blocks，末尾未完成的 block 留待后续 step。
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

            # all-mode 算子把每个已完成 block 的末尾状态写入对应物理槽位。
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

        step_start = req_meta.token_processed
        step_end = min(step_start + new_tokens, req_meta.num_token_ids)
        chunks = build_step_chunk_metadata(
            getattr(req_meta, "load_chunk_meta", []), step_start, step_end, block_size
        )
        # 最后一个 prompt block 必须保留正常计算，才能产生有效 logits。
        logits_limit = (req_meta.num_token_ids - 1) // block_size * block_size
        mamba_chunks = build_step_mamba_chunk_metadata(
            getattr(req_meta, "mamba_load_chunk_meta", []),
            step_start,
            step_end,
            block_size,
            logits_limit,
        )
        # GDN 提前组合跨步源 block 时，还需预加载下一整个目标 FA block。
        future_fa = []
        for chunk in mamba_chunks:
            first = chunk.destination_start + chunk.output_offset
            if first >= step_start and first + chunk.output_length > step_end:
                future_fa.extend(
                    build_step_chunk_metadata(
                        [
                            fa
                            for fa in getattr(req_meta, "load_chunk_meta", [])
                            if fa.chunk_hash == chunk.chunk_hash
                        ],
                        step_end,
                        step_end + block_size,
                        block_size,
                    )
                )
        # 推进 scheduler 已下发计划的位置，尚不代表 worker 已完成计算。
        req_meta.token_processed += new_tokens

        return KVBRequestDispatchMeta(
            load_block_ids=(load_ucm_ids, load_vllm_ids),
            dump_block_ids=(dump_ucm_ids, dump_vllm_ids),
            load_chunk_meta=chunks,
            mamba_load_chunk_meta=mamba_chunks,
            group_vllm_block_ids=[list(ids) for ids in req_meta.group_vllm_block_ids],
            fa_lookahead_chunk_meta=future_fa,
            reset_kvb_state=incoming_block_ids_are_full,
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """加载前缀及 chunk，等待所有已提交任务，再校验并发布复用信息。"""
        metadata = self._get_connector_metadata()
        # 同一个 dispatch 只处理一次，避免重复加载及重置临时状态。
        if metadata is self._kvb_processed_metadata:
            return
        super().start_load_kv(forward_context, **kwargs)
        self._kvb_processed_metadata = metadata
        self.kvb_denoised_states.clear()
        if not hasattr(self, "kvb_cross_step_states"):
            self.kvb_cross_step_states = {}
        self.kvb_next_step_fa_cache = {}
        self._kvb_restored_cross_step_states = {}
        for request_id in getattr(metadata, "preempted_req_ids", ()):
            self.kvb_cross_step_states.pop(request_id, None)
        for request_id, request in metadata.request_meta.items():
            if getattr(request, "reset_kvb_state", False):
                self.kvb_cross_step_states.pop(request_id, None)
        jobs = []
        count = 0
        for request_id, request in metadata.request_meta.items():
            for chunk_index, chunk in enumerate(
                [
                    *getattr(request, "load_chunk_meta", []),
                    *getattr(request, "mamba_load_chunk_meta", []),
                    *getattr(request, "fa_lookahead_chunk_meta", []),
                ]
            ):
                for group in chunk.groups:
                    ids = list(range(count, count + len(group.block_ids)))
                    jobs.append((request_id, chunk_index, request, chunk, group, ids))
                    count += len(ids)
        if not jobs:
            self._prepare_reuse_masks_and_metadata(forward_context, metadata, set())
            return
        # 临时容量不足时关闭本轮新增命中，已提交跨步状态仍会恢复。
        if count > self.kv_cache_layout_kvb.load_num_blocks:
            logger.warning(
                "KVB requires %s temporary blocks, exceeding registered capacity %s",
                count,
                self.kv_cache_layout_kvb.load_num_blocks,
            )
            self._prepare_reuse_masks_and_metadata(
                forward_context, metadata, {(j[0], j[1]) for j in jobs}
            )
            return
        self.device.synchronize()
        pending, failed = [], set()
        for job in jobs:
            request_id, chunk_index, request, chunk, group, ids = job
            keys = list(group.block_ids)
            # 普通 FA/GDN 各 TP rank 使用与保存端相同的分片哈希。
            scoped = (
                keys
                if self.tp_rank % self.tp_size == 0
                else [self.request_hasher(key) for key in keys]
            )
            try:
                ptrs = self.kv_cache_layout_kvb.extract_block_addrs(ids)
                task = self._rank_consistency.submit_load(
                    self.store, {request_id: keys}, scoped, [0] * len(scoped), ptrs
                )
                pending.append((job, task))
            except Exception as exc:  # noqa: BLE001 - 提交失败仍需等待其他已提交任务
                failed.add((request_id, chunk_index))
                logger.warning(
                    "KVB submit failed for %s chunk %s: %s",
                    request_id,
                    chunk_index,
                    exc,
                )
        completed = []
        # 即使其他组失败，也必须等待每个已提交任务结束。
        for job, task in pending:
            try:
                self._rank_consistency.wait_load(task)
                completed.append(job)
            except Exception as exc:  # noqa: BLE001 - 每个已提交任务都必须完成等待
                failed.add((job[0], job[1]))
                logger.warning(
                    "KVB wait failed for %s chunk %s: %s", job[0], job[1], exc
                )
        # 同一 chunk 任一组失败，就跳过该 chunk 的所有加载后处理。
        for job in completed:
            if (job[0], job[1]) in failed:
                continue
            try:
                self._copy_temp_to_kvcaches(job, forward_context)
            except (ValueError, TypeError, RuntimeError, MemoryError) as exc:
                failed.add((job[0], job[1]))
                logger.warning(
                    "KVB postprocess failed for %s chunk %s: %s", job[0], job[1], exc
                )
        self._prepare_reuse_masks_and_metadata(forward_context, metadata, failed)

    def _copy_temp_to_kvcaches(
        self, job, forward_context: "ForwardContext" = None
    ) -> None:
        """将临时缓存按类型后处理：FA 修正位置，Mamba 准备组合数据。"""
        request_id, chunk_index, request, chunk, group, ids = job
        is_next_step_fa_chunk = bool(
            getattr(request, "fa_lookahead_chunk_meta", ())
        ) and chunk_index >= (
            len(getattr(request, "load_chunk_meta", ()))
            + len(getattr(request, "mamba_load_chunk_meta", ()))
        )
        cache_group = self._kv_cache_config.kv_cache_groups[group.group_id]
        spec_map = layer_name_to_kv_cache_spec(self._kv_cache_config)
        for name in cache_group.layer_names:
            if name not in self.kv_caches:
                continue  # 该层可能由流水线中的其他 worker 持有
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
            key, value = get_attention_kv_views(
                self.kv_caches[name], spec, self._kv_cache_config.num_blocks
            )
            start = chunk.destination_start + chunk.output_offset
            length = chunk.output_length
            if start % group.block_size or length <= 0 or length % group.block_size:
                raise ValueError("KVB destination must cover complete aligned blocks")
            first_block = start // group.block_size
            blocks = request.group_vllm_block_ids[group.group_id]
            n = length // group.block_size
            if not is_next_step_fa_chunk and (first_block < 0 or first_block + n > len(blocks)):
                raise ValueError("KVB destination outside allocated block table")
            dst_ids = blocks[first_block : first_block + n]
            offset = group.source_offset + chunk.output_offset
            staging = self.kv_cache_layout_kvb
            temp_k = staging.tensor_view(name, key)[ids[0] : ids[-1] + 1]
            temp_v = staging.tensor_view(name, value)[ids[0] : ids[-1] + 1]
            src_k = temp_k.reshape(-1, *key.shape[2:])[offset : offset + length]
            src_v = temp_v.reshape(-1, *value.shape[2:])[offset : offset + length]
            # K 按目标减源的位置差旋转，V 保留原值。
            if chunk.shift:
                delta_positions = torch.full(
                    (length,), chunk.shift, dtype=torch.long, device=key.device
                )
                src_k = self.rotary_emb_3d.shift(src_k, delta_positions)
            # 两个视图均通过计算和形状校验后，再写目标 K/V。
            src_k = src_k.reshape(n, group.block_size, *key.shape[2:])
            src_v = src_v.reshape(n, group.block_size, *value.shape[2:])
            # 下一步物理 block 可能尚未分配，先持有修正后 K/V 的独立副本。
            if is_next_step_fa_chunk:
                future = self.kvb_next_step_fa_cache.setdefault(
                    (request_id, chunk.chunk_hash, start), {}
                )
                future[name] = (src_k.clone(), src_v.clone())
            else:
                key[dst_ids] = src_k
                value[dst_ids] = src_v

    def _denoise_mamba_chunk(self, name, job):
        """去除历史入口贡献并重定本段入口，保留 conv、原始 g 和源状态。"""
        request_id, chunk_index, request, chunk, group, ids = job
        # writer 依次注册 conv、recurrent state、原始 g，视图偏移和步长沿用注册布局。
        conv, state, g = self.kv_caches[name]
        if g.shape[1:] != (group.block_size, state.shape[1]):
            raise ValueError("registered g must have shape [blocks, block_size, heads]")
        state_view = self.kv_cache_layout_kvb.tensor_view(name, state)[
            ids[0] : ids[-1] + 1
        ]
        g_view = self.kv_cache_layout_kvb.tensor_view(name, g)[ids[0] : ids[-1] + 1]
        preceding = int(group.has_previous_state)
        initial = state_view[0] if preceding else None
        clean = denoise_gdn(state_view[preceding:], g_view[preceding:], initial)
        first = chunk.output_offset // group.block_size
        last = first + chunk.output_length // group.block_size
        if group.source_offset or chunk.output_offset % group.block_size:
            raise ValueError("GDN reuse must start at a checkpoint boundary")
        if chunk.output_length <= 0 or chunk.output_length % group.block_size:
            raise ValueError("GDN reuse must cover complete blocks")
        # 排除前置 checkpoint，并截出当前输出段的原始 g。
        selected_g = g_view[preceding + first : preceding + last]
        if (
            selected_g.shape[0] != last - first
            or clean[first:last].shape[0] != last - first
        ):
            raise ValueError("GDN staging does not cover the requested checkpoints")
        log_decay = selected_g.float().sum(dim=1).cumsum(dim=0)
        # 从区间中部开始时扣除前段贡献，新入口只乘本段累计衰减。
        selected_clean = clean[first:last]
        if first:
            decay = log_decay.exp().reshape(*log_decay.shape, *([1] * (clean.ndim - 2)))
            selected_clean = selected_clean - decay * clean[first - 1]
        start = chunk.destination_start + chunk.output_offset
        block_start = start // group.block_size
        slots = request.group_vllm_block_ids[group.group_id][
            block_start : block_start + last - first
        ]
        if len(slots) != last - first or any(
            s < 0 or s >= state.shape[0] for s in slots
        ):
            raise ValueError("GDN destination outside allocated block table")
        conv_view = self.kv_cache_layout_kvb.tensor_view(name, conv)[
            ids[0] : ids[-1] + 1
        ]
        # states 用于组合，source_states 用于非对齐槽位保存，g 按目标 token 写回。
        self.kvb_denoised_states[(request_id, chunk_index, name)] = {
            "chunk_hash": chunk.chunk_hash,
            "states": selected_clean,
            "source_states": state_view[preceding + first : preceding + last],
            "log_decay": log_decay,
            "conv": conv_view[preceding + first : preceding + last],
            "g": selected_g,
            "destination_block_ids": list(slots),
            "destination_start_offset": start % group.block_size,
            "block_size": group.block_size,
            "source_start": chunk.source_start + chunk.output_offset,
            "destination_start": chunk.destination_start + chunk.output_offset,
            "length": chunk.output_length,
        }

    def _get_tp_agreed_reuse_flags(self, ready):
        """对各 rank 的就绪标志取交集，任一 rank 失败就禁用该段新增复用。"""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            parallel = getattr(
                getattr(self, "_vllm_config", None), "parallel_config", None
            )
            return (
                ready
                if getattr(parallel, "tensor_parallel_size", 1) == 1
                else [False] * len(ready)
            )
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        flags = torch.tensor(ready, dtype=torch.int32, device="cpu")
        if group.world_size > 1:
            torch.distributed.all_reduce(
                flags, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
            )
        return flags.bool().tolist()

    def _validate_tp_cross_step_states(self, status):
        """校验各 rank 已提交跨步状态完整且一致，否则报错。"""
        if not status:
            return
        # 状态值 0 为无记录、1 为完整、-1 为无效；MIN/MAX 检查 rank 一致性。
        low = torch.tensor(status, dtype=torch.int32, device="cpu")
        high = low.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            from vllm.distributed import get_tp_group

            group = get_tp_group()
            if group.world_size > 1:
                torch.distributed.all_reduce(
                    low, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
                )
                torch.distributed.all_reduce(
                    high, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
                )
        if not torch.equal(low, high) or bool((low < 0).any()):
            raise RuntimeError(
                "KVB continuation is incomplete or inconsistent across TP ranks"
            )

    def _restore_cross_step_states(self, context, metadata, names, fa_ids, block_size):
        """恢复已提交跨步状态及预留 FA 缓存，发布本 step 的前缀复用信息。"""
        inputs = context.kvb_gdn_inputs
        status, pending = [], []
        for index, rid in enumerate(inputs["request_ids"]):
            cross_step_state = self.kvb_cross_step_states.get(rid)
            if cross_step_state is None:
                status.append(0)
                continue
            request = metadata.request_meta.get(rid)
            start = inputs["positions"][index]
            end = start + len(context.compute_masks[index])
            # 终点由原映射推导；只有所有 GDN 层完成组合的记录才允许恢复。
            terminal = cross_step_state["destination_start"] + cross_step_state["length"]
            valid = (
                request is not None
                and cross_step_state["completed_layer_names"] == set(names)
                and set(cross_step_state["layers"]) == set(names)
                and cross_step_state["fa_start"] <= start <= terminal
                and any(
                    c.chunk_hash == cross_step_state["chunk_hash"]
                    and c.destination_start + c.output_offset
                    == cross_step_state["destination_start"]
                    and c.output_length == cross_step_state["length"]
                    for c in getattr(request, "mamba_load_chunk_meta", ())
                )
            )
            writes = []
            if valid:
                for name in names:
                    meta = context.attn_metadata.get(name)
                    if (
                        not getattr(meta, "is_all_mode", False)
                        or getattr(meta, "spec_state_indices_tensor", None) is not None
                        or name not in self.kv_caches
                        or len(self.kv_caches[name]) != 3
                    ):
                        valid = False
                        break
                for gid in fa_ids:
                    group = self._kv_cache_config.kv_cache_groups[gid]
                    # 使用本 step 最新物理表，不能沿用上一步尚未分配的 block 地址。
                    block = cross_step_state["fa_start"] // block_size
                    table = request.group_vllm_block_ids[gid]
                    if block >= len(table) or table[block] < 0:
                        valid = False
                        break
                    for name in group.layer_names:
                        if name not in cross_step_state["fa"] or name not in self.kv_caches:
                            valid = False
                            break
                        key, value = get_attention_kv_views(
                            self.kv_caches[name],
                            group.kv_cache_spec,
                            self._kv_cache_config.num_blocks,
                        )
                        cached_k, cached_v = cross_step_state["fa"][name]
                        slot = table[block]
                        if (
                            slot >= key.shape[0]
                            or slot >= value.shape[0]
                            or cached_k.shape != (1, *key.shape[1:])
                            or cached_v.shape != (1, *value.shape[1:])
                        ):
                            valid = False
                            break
                        writes.append((key, value, slot, cached_k, cached_v))
            status.append(1 if valid else -1)
            pending.append((index, rid, start, end, terminal, cross_step_state, writes))
        # 已推进状态丢失或 rank 不一致时必须报错，不能回退到历史源状态。
        self._validate_tp_cross_step_states(status)
        for index, rid, start, end, terminal, cross_step_state, writes in pending:
            for key, value, slot, cached_k, cached_v in writes:
                key[slot].copy_(cached_k[0])
                value[slot].copy_(cached_v[0])
            # 预留的完整 FA block 也支持下一 step 只调度其中一小段。
            context.compute_masks[index][
                : min(end, cross_step_state["fa_start"] + block_size) - start
            ] = 1
            context.kvb_gdn_masks[index][: min(end, terminal) - start] = 1
            self._kvb_restored_cross_step_states[rid] = cross_step_state
            for name in names:
                context.kvb_gdn_cross_step_inputs.setdefault(name, {})[rid] = cross_step_state

    def _prepare_reuse_masks_and_metadata(self, context, metadata, failed):
        """分别校验 FA 和 GDN 候选，发布复用 mask、分段数据及跨步记录。"""
        inputs = getattr(context, "kvb_gdn_inputs", None)
        if inputs is None:
            return
        allow_reuse = getattr(context, "kvb_gdn_allow_reuse", True)
        if not hasattr(self, "kvb_cross_step_states"):
            self.kvb_cross_step_states = {}
        self._kvb_restored_cross_step_states = {}
        context.kvb_gdn_cross_step_inputs = {}
        context.kvb_gdn_meta = {}
        # 两种 mask 都以当前 batch 请求下标索引，1 表示复用、0 表示计算。
        context.compute_masks = {
            i: np.zeros(end - start, dtype=np.int64)
            for i, (start, end) in enumerate(
                zip(inputs["query_start_loc"], inputs["query_start_loc"][1:])
            )
        }
        context.kvb_gdn_masks = {
            i: np.zeros_like(mask) for i, mask in context.compute_masks.items()
        }
        indices = {rid: i for i, rid in enumerate(inputs["request_ids"])}
        specs = layer_name_to_kv_cache_spec(self._kv_cache_config)
        fa_ids = {
            i
            for i, g in enumerate(self._kv_cache_config.kv_cache_groups)
            if _is_full_attention_group(g.kv_cache_spec)
        }
        state_ids = set(range(len(self._kv_cache_config.kv_cache_groups))) - fa_ids
        names = [
            name for name, values in specs.items() if isinstance(values[0], MambaSpec)
        ]
        block_size = next(
            block_size_from_kv_cache_spec(g.kv_cache_spec)
            for g in self._kv_cache_config.kv_cache_groups
            if _is_mamba_all_group(g.kv_cache_spec)
        )
        fa_candidates, ready = [], []
        for rid, request in sorted(metadata.request_meta.items()):
            for job_index, chunk in enumerate(getattr(request, "load_chunk_meta", ())):
                index = indices.get(rid)
                first = chunk.destination_start + chunk.output_offset
                last = first + chunk.output_length
                valid = (
                    allow_reuse and index is not None and (rid, job_index) not in failed
                )
                valid = valid and {g.group_id for g in chunk.groups} == fa_ids
                valid = valid and all(
                    name in self.kv_caches
                    for gid in fa_ids
                    for name in self._kv_cache_config.kv_cache_groups[gid].layer_names
                )
                if valid:
                    position = inputs["positions"][index]
                    count = len(context.compute_masks[index])
                    # FA 只复用完整目标 blocks，最后一个 prompt block 保留计算 logits。
                    last = min(
                        last,
                        (inputs["prompt_lengths"][index] - 1)
                        // block_size
                        * block_size,
                    )
                    valid = position <= first < last <= position + count
                fa_candidates.append((index, first, last))
                ready.append(bool(valid))
        if ready:
            for accepted, (index, first, last) in zip(
                self._get_tp_agreed_reuse_flags(ready), fa_candidates
            ):
                if accepted:
                    offset = first - inputs["positions"][index]
                    context.compute_masks[index][offset : offset + last - first] = 1

        # 禁止新增复用时也必须消费已提交状态，保证状态与 token 位置一致。
        self._restore_cross_step_states(context, metadata, names, fa_ids, block_size)

        tables = {}
        for name in names:
            table = getattr(context.attn_metadata.get(name), "block_table_2d", None)
            if table is not None:
                tables[name] = table.detach().cpu().tolist()
        candidates, ready = [], []
        for rid, request in sorted(metadata.request_meta.items()):
            fa_count = len(getattr(request, "load_chunk_meta", ()))
            for chunk_index, chunk in enumerate(
                getattr(request, "mamba_load_chunk_meta", ())
            ):
                job_index = fa_count + chunk_index
                index = indices.get(rid)
                first = chunk.destination_start + chunk.output_offset
                last = first + chunk.output_length
                valid = (
                    allow_reuse
                    and index is not None
                    and (rid, job_index) not in failed
                    and bool(names)
                    and {g.group_id for g in chunk.groups} == state_ids
                    and all(name in self.kv_caches for name in names)
                )
                if valid:
                    offset = first - inputs["positions"][index]
                    # 本 step 的 GDN 复用 token 必须全部落在成功的 FA mask 内。
                    mask = context.compute_masks[index]
                    step_end = inputs["positions"][index] + len(mask)
                    valid = (
                        0 <= offset < len(mask)
                        and chunk.output_length % block_size == 0
                        and bool(
                            mask[
                                offset : min(len(mask), offset + chunk.output_length)
                            ].all()
                        )
                    )
                future = None
                # 跨步只允许在目标 block 边界提前组合，且必须备齐下一 FA block。
                if valid and last > step_end:
                    future_chunks = getattr(request, "fa_lookahead_chunk_meta", ())
                    future_start_index = fa_count + len(
                        getattr(request, "mamba_load_chunk_meta", ())
                    )
                    future = getattr(self, "kvb_next_step_fa_cache", {}).get(
                        (rid, chunk.chunk_hash, step_end)
                    )
                    fa_names = {
                        name
                        for gid in fa_ids
                        for name in self._kv_cache_config.kv_cache_groups[
                            gid
                        ].layer_names
                    }
                    valid = (
                        step_end % block_size == 0
                        and last - step_end < block_size
                        and last
                        <= (inputs["prompt_lengths"][index] - 1)
                        // block_size
                        * block_size
                        and future is not None
                        and set(future) == fa_names
                        and any(
                            c.chunk_hash == chunk.chunk_hash
                            and c.destination_start + c.output_offset == step_end
                            and c.output_length == block_size
                            and {g.group_id for g in c.groups} == fa_ids
                            and (rid, future_start_index + i) not in failed
                            for i, c in enumerate(future_chunks)
                        )
                    )
                # 校验每个 GDN 层的 all-mode 布局及物理表与加载描述一致。
                payload = {}
                if valid:
                    for name in names:
                        layer_meta = context.attn_metadata.get(name)
                        span = self.kvb_denoised_states.get((rid, job_index, name))
                        if (
                            layer_meta is None
                            or span is None
                            or not getattr(layer_meta, "is_all_mode", False)
                            or getattr(layer_meta, "spec_state_indices_tensor", None)
                            is not None
                            or len(self.kv_caches[name]) != 3
                            or getattr(layer_meta, "mamba_block_size", None)
                            != block_size
                            or index >= len(tables.get(name, ()))
                        ):
                            valid = False
                            break
                        count = chunk.output_length // block_size
                        first_block = first // block_size
                        table = tables[name][index]
                        if (
                            span["destination_start"] != first
                            or span["length"] != chunk.output_length
                            or table[first_block : first_block + count]
                            != span["destination_block_ids"]
                            or len(span["destination_block_ids"]) != count
                            or (min(last, step_end) - 1) // block_size >= len(table)
                            or any(
                                slot < 0 or slot >= self.kv_caches[name][1].shape[0]
                                for slot in table[
                                    first_block : (min(last, step_end) - 1)
                                    // block_size
                                    + 1
                                ]
                            )
                        ):
                            valid = False
                            break
                        payload[name] = {
                            **span,
                            "request_index": index,
                            "token_start": inputs["query_start_loc"][index] + offset,
                        }
                # 先分配跨步缓冲，forward 完成组合后才登记各层完成状态。
                if valid and future is not None:
                    try:
                        cross_step_state = {
                            "chunk_hash": chunk.chunk_hash,
                            "destination_start": first,
                            "length": chunk.output_length,
                            "fa_start": step_end,
                            "fa": future,
                            "layers": {
                                name: {
                                    "state": torch.empty_like(
                                        self.kv_caches[name][1][0]
                                    ),
                                    "conv": torch.empty_like(
                                        self.kv_caches[name][0][0]
                                    ),
                                    "g": span["g"]
                                    .reshape(-1, span["g"].shape[-1])
                                    .clone(),
                                }
                                for name, span in payload.items()
                            },
                            "completed_layer_names": set(),
                            "consumed_layer_names": set(),
                        }
                        for span in payload.values():
                            span["next_step_state"] = cross_step_state
                    except (RuntimeError, MemoryError) as exc:
                        logger.warning(
                            "KVB boundary buffering failed for %s: %s", rid, exc
                        )
                        valid = False
                candidates.append((rid, index, first, last, payload))
                ready.append(bool(valid))
        if not ready:
            return
        # 同一请求的重叠 GDN 区间全部拒绝，避免重复推进 recurrent state。
        windows = defaultdict(list)
        for i, ((rid, _index, first, last, _payload), valid) in enumerate(
            zip(candidates, ready)
        ):
            if valid:
                for other_first, other_last, other in windows[rid]:
                    if first < other_last and other_first < last:
                        ready[i] = ready[other] = False
                windows[rid].append((first, last, i))
        for accepted, (rid, index, first, last, payload) in zip(
            self._get_tp_agreed_reuse_flags(ready), candidates
        ):
            if not accepted:
                continue
            offset = first - inputs["positions"][index]
            # mask 仅覆盖本步已调度 token，未来 token 不重复计入跳过数量。
            context.kvb_gdn_masks[index][offset : offset + last - first] = 1
            for name, span in payload.items():
                if "next_step_state" in span:
                    self.kvb_cross_step_states[rid] = span["next_step_state"]
                context.kvb_gdn_meta.setdefault(name, {}).setdefault(rid, []).append(
                    span
                )
        reused = sum(int(mask.sum()) for mask in context.kvb_gdn_masks.values())
        if reused:
            logger.info(
                "KVB GDN forward reuses %s tokens (source-block aligned)", reused
            )

    def clear_connector_metadata(self):
        # 后处理算子可能仍引用临时视图，等待设备执行完成后再清理。
        """等待后处理结束，清理本轮临时数据并释放已消费的跨步记录。"""
        if self.kv_cache_layout_kvb is not None:
            self.device.synchronize()
        # 所有层保存后续状态后才释放旧记录，已被新记录替换时不误删。
        for rid, cross_step_state in getattr(self, "_kvb_restored_cross_step_states", {}).items():
            if (
                cross_step_state["consumed_layer_names"] == set(cross_step_state["layers"])
                and self.kvb_cross_step_states.get(rid) is cross_step_state
            ):
                self.kvb_cross_step_states.pop(rid)
        self._kvb_restored_cross_step_states = {}
        getattr(self, "kvb_next_step_fa_cache", {}).clear()
        self.kvb_denoised_states.clear()
        self._kvb_processed_metadata = None
        super().clear_connector_metadata()

    def get_finished(self, finished_req_ids):
        """删除已结束请求的跨步状态，再收集父类传输完成信息。"""
        for rid in finished_req_ids:
            getattr(self, "kvb_cross_step_states", {}).pop(rid, None)
        return super().get_finished(finished_req_ids)

    def handle_preemptions(self, kv_connector_metadata):
        """处理父类待保存任务，并删除被抢占请求的跨步状态。"""
        super().handle_preemptions(kv_connector_metadata)
        request_ids = (
            kv_connector_metadata
            if isinstance(kv_connector_metadata, set)
            else getattr(kv_connector_metadata, "preempted_req_ids", ())
        )
        for rid in request_ids or ():
            getattr(self, "kvb_cross_step_states", {}).pop(rid, None)
