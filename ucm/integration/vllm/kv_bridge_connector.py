import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
from enum import Enum, auto
import itertools
from vllm.config import VllmConfig
from collections import defaultdict
import math
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from ucm.shared.metrics import ucmmetrics
from vllm.v1.core.sched.output import SchedulerOutput
from ucm.logger import init_logger
import torch
from ucm.integration.vllm.ucm_connector import KVCacheLayout, UCMConnectorMetadata, RequestHasher, UCMDirectConnector, UCMLayerWiseConnector, UCMConnector
from ucm.utils import Config
from enum import IntEnum
from ucm.integration.vllm.perf_counter import PerfCounters
from vllm.v1.request import Request
import numpy as np

logger = init_logger(__name__)

MIN_CHUNK_LEN = 512
MAX_CLUSTER_NUM = 100000

class RotaryEmbedding3D:
    def __init__(
        self,
        head_dim: int,
        max_position: int = 8192,
        base: float = 1000000.0,
        device: str = 'cpu',
        # YaRN 特有参数
        use_yarn: bool = False,
        scaling_factor: float = 1.0,
        extrapolation_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
        dtype: torch.dtype = torch.float32
    ):
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.device = device
        self.dtype = dtype
        
        # YaRN 配置
        self.use_yarn = use_yarn
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
 
        # 初始化缓存 
        self.cos_sin_cached = None
        self._build_cache()

    def _yarn_linear_ramp_mask(self, min_v: float, max_v: float, d: int) -> torch.Tensor:
        if min_v == max_v:
            max_v += 0.001
        linear_func = (torch.arange(d, dtype=torch.float32, device=self.device) - min_v) / (max_v - min_v)
        return torch.clamp(linear_func, 0, 1)

    def _compute_inv_freq(self) -> torch.Tensor:
        # 1. 计算基础频率 (Common for both)
        # inv_freq = 1.0 / (base ** (arange / dim))
        pos_freqs = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim
        )
        inv_freq_standard = 1.0 / pos_freqs

        # 如果不使用 YaRN，直接返回标准频率
        if not self.use_yarn:
            return inv_freq_standard

        # === YaRN Logic Below ===
        inv_freq_extrapolation = inv_freq_standard
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)

        low = math.floor(self.head_dim * math.log(self.max_position / (self.beta_fast * 2 * math.pi)) / (2 * math.log(self.base)))
        high = math.ceil(self.head_dim * math.log(self.max_position / (self.beta_slow * 2 * math.pi)) / (2 * math.log(self.base)))
        low, high = max(low, 0), max(high, 0)

        inv_freq_mask = (1 - self._yarn_linear_ramp_mask(low, high, self.head_dim // 2)) * self.extrapolation_factor

        return inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    def _build_cache(self):
        inv_freq = self._compute_inv_freq()

        # 确定缓存大小
        # 如果是 YaRN，可能需要扩展 scaling_factor 倍
        # 如果是普通 RoPE，通常只需要 max_position
        if self.use_yarn:
            cache_len = int(self.max_position * self.scaling_factor) + 256
        else:
            cache_len = self.max_position + 1

        t = torch.arange(cache_len, dtype=torch.float32, device=self.device)

        freqs = torch.outer(t, inv_freq)

        # 纯净版：不乘 mscale，保证 Norm-preserving
        cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.cos_sin_cached = cos_sin.to(self.dtype)

    def _get_cos_sin(self, delta_positions: torch.Tensor):
        # 自动处理设备不匹配的情况
        if self.cos_sin_cached.device != delta_positions.device:
            self.cos_sin_cached = self.cos_sin_cached.to(delta_positions.device)

        # 1. 绝对值索引 (cos是对称的, sin是反对称的)
        abs_positions = delta_positions.abs()

        # 2. 从缓存获取
        # 注意：这里假设 abs_positions 不会超过 cache_len
        cos_sin = self.cos_sin_cached[abs_positions]  # [..., head_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)           # [..., head_dim//2]

        # 3. 负数位置处理：sin 取反
        # sign: 正数 -> 1.0, 负数 -> -1.0
        sign = (delta_positions >= 0).to(cos.dtype) * 2 - 1
        sin = sin * sign.unsqueeze(-1)

        return cos.unsqueeze(-2), sin.unsqueeze(-2)  # [..., 1, head_dim//2]

    def shift(self, key: torch.Tensor, delta_positions: torch.Tensor) -> torch.Tensor:
        """
        key: [..., num_heads, head_dim]
        delta_positions: [...] - 相对位置差
        """
        orig_dtype = key.dtype
        # 强制转为 float32 进行旋转计算以保证精度
        key_f32 = key.float()

        cos, sin = self._get_cos_sin(delta_positions)
        cos, sin = cos.float(), sin.float()

        x1, x2 = key_f32.chunk(2, dim=-1)

        # 标准旋转逻辑
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos

        output_f32 = torch.cat((o1, o2), dim=-1)
        return output_f32.to(orig_dtype)


class SingleVllmConfig:
    _instance = None

    def __new__(
            cls, 
            vllm_config: "VllmConfig"
        ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        vllm_config: "VllmConfig"
    ):
        # 注意：__init__ 每次都会调用，需做防护
        if not hasattr(self, '_initialized'):
            self.vllm_config = vllm_config
            self._initialized = True
            ucm_config = Config(vllm_config.kv_transfer_config)
            launch_config = ucm_config.get_config()
            self.padding_config = launch_config.get("padding_config", None)
            if self.padding_config is not None:
                self.chunk_end_token_id = self.padding_config.get("chunk_end_token_id", None)
                self.chunk_pad_token_id = self.padding_config.get("chunk_pad_token_id", None)
            else:
                self.chunk_end_token_id = -1
                self.chunk_pad_token_id = None


def is_token_in_chunk_end(
    token: int, 
    chunk_end_token: int
):
    return token == chunk_end_token

def pad_rag_chunks(
    token_ids: list[int],
    block_size: int,
    pad_id: int
):
    """
    pad token_ids with pad_id and end up with end_id
    """
    num_tokens = len(token_ids)
    remainder = num_tokens % block_size
    if pad_id is None or (remainder == 0 and num_tokens > 1):
        return token_ids

    pad_len = block_size - remainder
    padded = token_ids[:-1] + [pad_id] * pad_len + token_ids[-1]
    return padded

def kvb_replace_padding(
    prompt_token_ids: list[int],
    vllm_config: "VllmConfig"
):
    padding_config = SingleVllmConfig(vllm_config)
    if not padding_config.padding_config:
        return prompt_token_ids
    kvb_chunk_pad_token_id = padding_config.chunk_pad_token_id
    kvb_chunk_end_token_id = padding_config.chunk_end_token_id
    kvb_block_size = vllm_config.cache_config.block_size
    start = 0
    end = 0
    chunks = []
    for end, token in enumerate(prompt_token_ids):
        if is_token_in_chunk_end(token, kvb_chunk_end_token_id):
            chunks.append(prompt_token_ids[start:end + 1])
            start = end + 1
    chunks_padding = [pad_rag_chunks(x, kvb_block_size, kvb_chunk_pad_token_id) for x in chunks]
    if start < len(prompt_token_ids):
        chunks_padding.append(prompt_token_ids[start:])
    final_chunks = []
    for sublist in chunks_padding:
        final_chunks.extend(sublist)
    return final_chunks

class KVCacheLayoutKVB(KVCacheLayout):
    def __init__(
        self, kvcaches, use_layerwise: bool, vllm_config: "VllmConfig"
    ) -> None:
        self.load_num_blocks = 1200
        self.device = None
        super().__init__(kvcaches, use_layerwise, vllm_config)

    def _build_layout(self, kvcaches):
        raw_ptr_rows = [[] for _ in range(self.local_num_hidden_layers)]
        stride_rows = [[] for _ in range(self.local_num_hidden_layers)]

        # 必须将新申请的 Tensor 保存在实例属性中，否则函数结束后显存会被自动回收（产生野指针）
        self.load_kvcaches = {}

        for layer_name, kv_layer in kvcaches.items():
            ptrs = []
            strides = []

            def handle_tensor(t: torch.Tensor, size_dims):
                ptrs.append(t[0].data_ptr())
                stride = math.prod([t.shape[i] for i in size_dims]) * t.element_size()
                strides.append(stride)

            if isinstance(kv_layer, Tuple):
                # vllm_ascend >= 0.10.0, ([num_blocks, block_size, num_head, head_dim], ...)
                new_kv_tensors = []
                for tensor in kv_layer:
                    # 获取原有的 shape
                    shape = list(tensor.shape)
                    self.device = tensor.device

                    # 如果配置了特定的 num_blocks，则覆盖 shape 的第 0 维
                    if self.load_num_blocks is not None:
                        shape[0] = self.load_num_blocks

                    # 申请新的一块和 kvcaches 结构一致的 Tensor 作为暂存区
                    # 使用 torch.empty 可以避免不必要的零初始化，加快内存分配速度
                    new_tensor = torch.empty(
                        shape,
                        dtype=tensor.dtype,
                        device=tensor.device
                    )
                    new_kv_tensors.append(new_tensor)

                    # 对新申请的暂存区 tensor 提取底层指针和 stride
                    handle_tensor(new_tensor, (-3, -2, -1))

                # 将元组形式的(k_cache, v_cache)缓存起来
                self.load_kvcaches[layer_name] = tuple(new_kv_tensors)
            else:
                raise TypeError(f"Unsupported kv cache type: {type(kv_layer)}")

            local_layer_id = self.layer_name_to_id[layer_name] - self.first_layer_id
            raw_ptr_rows[local_layer_id].extend(ptrs)
            stride_rows[local_layer_id].extend(strides)

        self.base_ptrs = np.asarray(raw_ptr_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(stride_rows, dtype=np.uint64)

        logger.info(
            f"base_ptrs: {self.base_ptrs.shape}, tensor_size_lists: {self.tensor_size_lists.shape}"
        )


def save_kvcaches(kvcaches, layer_idx, save_name):
    kv_items = list(kvcaches.items())
    for idx in layer_idx:
        layer_name, kv_layer = kv_items[idx]
        kcache, vcache = kv_layer
        torch.save(kcache, f'{save_name}_layer{idx}_kcache.pt')
        torch.save(vcache, f'{save_name}_layer{idx}_vcache.pt')


@dataclass
class KVClusterInfo:
    # Fields restored from the conversation's dataclass table.
    chunk_num: int = 0
    chunk_hash_ids: list[bytes] = field(default_factory=list)
    chunk_start_positions: dict[str, int] = field(default_factory=dict)
    chunk_lens: dict[str, int] = field(default_factory=dict)
    chunk_start_block_ids: dict[str, bytes] = field(default_factory=dict)


class KVCluster:
    def __init__(
        self, 
        vllm_config: "VllmConfig", 
        kvcluster_info: KVClusterInfo, 
        req_id = '-1'
    ):
        self.kvcluster_info = kvcluster_info
        self.request_hasher = RequsetHasher(vllm_config, 0)
        self._seed = self.request_hasher("KV_BRIDGE_HASH_SEED")
        self.cluster_id = self._get_cluster_hash(kvcluster_info.chunk_hash_ids)
        self.request_id = req_id

    def _get_cluster_hash(
        self,
        chunk_hash_ids: List[bytes]
    ):
        parent_block_hash_value = self._seed
        hash_value = 0
        for chunk_hash_id in chunk_hash_ids:
            hash_value = self.request_hasher(
                (parent_block_hash_value, chunk_hash_id)
            )
            parent_block_hash_value = hash_value
        return parent_block_hash_value


class KVClusterMap:
    def __init__(
        self, 
        vllm_config: "VllmConfig", 
        block_size
    ):
        self.candidates = defaultdict(KVCluster)
        self.inverted = defaultdict(set)
        self.block_size = block_size
        self.max_cluster_num = max(1, MAX_CLUSTER_NUM)

    def search_top_one(
            self, 
            query_chunk_list: List[str], 
            chunk_lens: dict[bytes, int]
        ):
        query_set = set(query_chunk_list)
        counter = defaultdict(int)

        for chunk_hash_id in query_set:
            chunk_blocks_len = chunk_lens[chunk_hash_id]
            for cluster_hash_id in self.inverted[chunk_hash_id]:
                counter[cluster_hash_id] += chunk_blocks_len

        best_id = None
        best_inter = 0
        for cluster_hash_id, inter in counter.items():
            if inter > best_inter:
                best_inter, best_id = inter, cluster_hash_id

        return best_id

    def _ensure_capacity(self):
        if len(self.candidates) < self.max_cluster_num:
            return

        oldest_cluster_hash_id = next(iter(self.candidates))
        oldest_cluster = self.candidates[oldest_cluster_hash_id]
        chunk_hash_ids = oldest_cluster.kvcluster_info.chunk_hash_ids
        for chunk_hash_id in chunk_hash_ids:
            self.inverted[chunk_hash_id].discard(oldest_cluster_hash_id)
            if len(self.inverted[chunk_hash_id]) == 0:
                del self.inverted[chunk_hash_id]
        del self.candidates[oldest_cluster_hash_id]

    def insert_cluster(
        self,
        new_cluster: KVCluster
    ):
        if new_cluster is None:
            return
        self._ensure_capacity()
        cluster_hash_id = new_cluster.cluster_id
        chunk_hash_ids = new_cluster.kvcluster_info.chunk_hash_ids
        if cluster_hash_id in self.candidates:
            return
        self.candidates[cluster_hash_id] = new_cluster
        for chunk_hash_id in chunk_hash_ids:
            self.inverted[chunk_hash_id].add(cluster_hash_id)

@dataclass
class LoadChunkMeta:
    load_chunks_blocks: list = field(default_factory=list)
    load_src_offsets: list = field(default_factory=list)
    load_dst_chunks_start_positions: list = field(default_factory=list)
    load_chunks_shift_positions: list = field(default_factory=list)
    load_chunks_len: list = field(default_factory=list)


@dataclass
class RequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0
    current_vllm_block_ids: list[int] = field(default_factory=list)
    best_kv_cluster: KVCluster = None
    load_chunk_meta: LoadChunkMeta = None


@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[list[bytes], list[int]]
    dump_block_ids: tuple[list[bytes], list[int]]
    load_chunk_meta: LoadChunkMeta
    vllm_block_ids: list[int]
    need_load: bool = False


class UCMKvBridgeDirectConnector(UCMDirectConnector):
    """
    This connector means synchronize:
    load -> forward -> save
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole
    ):
        super().__init__(vllm_config=vllm_config, role=role)
        self.max_model_len = vllm_config.model_config.max_model_len
        self.kv_cluster_map = KVClusterMap(vllm_config, self.block_size)

        # save block info, avoid hash request twice, and track them until request finished
        self.vllm_config = vllm_config
        kv_bridge_config = self.launch_config.get("kv_bridge_config", [])
        padding_config = self.launch_config.get("padding_config", [])

        self.emb_config = kv_bridge_config["embedded"]
        self.chunk_end_token_id = padding_config.get("chunk_end_token_id", -1)

        self.hf_text_config = vllm_config.model_config.hf_text_config

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        self.kv_cache_layout_kvb = KVCacheLayoutKVB(
            self.kv_caches, self.use_layerwise, self._vllm_config
        )
        self.rotary_emb_3d = RotaryEmbedding3D(
            head_dim=self.vllm_config.model_config.get_head_size(),
            max_position=getattr(self.hf_text_config, "max_position_embeddings", self.max_model_len),
            base=getattr(self.hf_text_config, "rope_theta", 1000000.0),
            device=self.kv_cache_layout_kvb.device,
            dtype=torch.float32,
        )


    def generate_chunk_hash(
        self,
        token_ids: list[int]
    ) -> bytes:
        token_ids_tuple = tuple(token_ids)
        hash_value = self.request_hasher(
            (self._seed, token_ids_tuple)
        )
        return hash_value

    def _process_req(self, all_token_ids, ucm_block_ids, prefix_len):
        chunk_num: int = 0
        chunk_hash_ids: list[bytes] = []
        chunk_start_positions: dict[str, int] = {}
        chunk_lens: dict[str, int] = {}
        chunk_start_block_ids: dict[str, bytes] = {}
        start_token_idx: int = prefix_len
        for end_token_idx in range(prefix_len, len(all_token_ids)):
            if is_token_in_chunk_end(
                all_token_ids[end_token_idx],
                self.chunk_end_token_id
            ):
                if (end_token_idx + 1 - start_token_idx) >= MIN_CHUNK_LEN:
                    chunk_token_ids = all_token_ids[start_token_idx : (end_token_idx + 1)]
                    chunk_hash = self.generate_chunk_hash(chunk_token_ids)

                    chunk_num += 1
                    chunk_hash_ids.append(chunk_hash)
                    chunk_start_positions[chunk_hash] = start_token_idx
                    chunk_lens[chunk_hash] = len(chunk_token_ids)
                    chunk_start_block_ids[chunk_hash] = ucm_block_ids[start_token_idx // self.block_size]
                    logger.info(f"[kvbridge] chunk_hash {chunk_hash.hex()} block_id {ucm_block_ids[start_token_idx // self.block_size].hex()} {ucm_block_ids[start_token_idx // self.block_size + 1].hex()}")
                start_token_idx = end_token_idx + 1

        if chunk_num == 0:
            return None, None

        kvcluster_info = KVClusterInfo(
            chunk_num=chunk_num,
            chunk_hash_ids=chunk_hash_ids,
            chunk_start_positions=chunk_start_positions,
            chunk_lens=chunk_lens,
            chunk_start_block_ids=chunk_start_block_ids
        )
        best_kv_cluster_hash_id = self.kv_cluster_map.search_top_one(chunk_hash_ids, chunk_lens)
        return kvcluster_info, best_kv_cluster_hash_id

    def get_hit_meta(self, request, new_cluster, best_cluster):
        new_info = new_cluster.kvcluster_info
        best_info = best_cluster.kvcluster_info

        # 查找交集，保持 new_cluster 的顺序
        matched_hashes = [h for h in new_info.chunk_hash_ids if h in best_info.chunk_hash_ids]

        # 容器初始化
        meta_data = {
            "load_chunks_blocks": [],
            "load_src_offsets": [],
            "load_dst_chunks_start_positions": [],
            "load_chunks_shift_positions": [],
            "load_chunks_len": [],
        }

        for chunk_hash in matched_hashes:
            # 1. 结构化解构，减少深层访问
            best_start_pos = best_info.chunk_start_positions[chunk_hash]
            new_start_pos = new_info.chunk_start_positions[chunk_hash]
            chunk_raw_len = best_info.chunk_lens[chunk_hash]
            start_block_id = best_info.chunk_start_block_ids[chunk_hash]

            # 2. 块对齐与偏移量计算
            shift_len = (-new_start_pos) % self.block_size
            src_start = best_start_pos + shift_len
            dst_start = new_start_pos + shift_len

            # 3. 计算 Token 切片与 Block IDs
            # 使用整除计算切片起始位置，可进一步优化 start_pos
            start_pos = (best_start_pos // self.block_size + 1) * self.block_size
            lead = start_pos - best_start_pos

            tokens_slice = request.all_token_ids[(new_start_pos + lead):(new_start_pos + chunk_raw_len)]
            generated_blocks = self.generate_hash(self.block_size, tokens_slice, start_block_id)
            load_src_offset = src_start - start_pos + self.block_size
            if load_src_offset < self.block_size:
                external_hit_blocks = [start_block_id] + generated_blocks
            else:
                external_hit_blocks = generated_blocks
                load_src_offset -= self.block_size
            try:
                external_hit_block_num = self.store.lookup_on_prefix(external_hit_blocks) + 1
            except RuntimeError as e:
                external_hit_block_num = 0
            if external_hit_block_num == 0:
                continue
            external_hit_blocks = external_hit_blocks[:external_hit_block_num]
            chunk_hit_len = external_hit_block_num * self.block_size

            # 4. 组装结果
            meta_data["load_chunks_blocks"].append(external_hit_blocks)
            meta_data["load_src_offsets"].append(load_src_offset)
            meta_data["load_dst_chunks_start_positions"].append(dst_start)
            meta_data["load_chunks_shift_positions"].append(dst_start - src_start)
            meta_data["load_chunks_len"].append(((chunk_hit_len - load_src_offset) // self.block_size) * self.block_size)

        return LoadChunkMeta(**meta_data)

    def get_kvbridge_matched_tokens(self, request, ucm_block_ids, prefix_hit_len):
        new_kv_cluster = None
        best_kv_cluster = None
        load_chunk_meta = None

        kvcluster_info, best_cluster_hash_id = self._process_req(request.all_token_ids, ucm_block_ids, prefix_hit_len)

        if kvcluster_info is not None:
            new_kv_cluster = KVCluster(self.vllm_config, kvcluster_info, request.request_id)

        if best_cluster_hash_id is not None:
            best_kv_cluster = self.kv_cluster_map.candidates[best_cluster_hash_id]

        if new_kv_cluster is not None and best_kv_cluster is not None:
            load_chunk_meta = self.get_hit_meta(request, new_kv_cluster, best_kv_cluster)
            logger.info_once(f"[kvbridge] hit request: {best_kv_cluster.request_id}, hit chunks: {len(load_chunk_meta.load_chunks_blocks)}")
        return new_kv_cluster, load_chunk_meta

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        ucm_prefix_hit_block_num, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if request.request_id not in self.requests_meta:
            return 0, False
        num_prefix_hit_tokens = num_computed_tokens + ucm_prefix_hit_block_num
        ucm_block_ids = self.generate_hash(
            self.block_size, request.all_token_ids, self._seed
        )
        new_kv_cluster, load_chunk_meta = self.get_kvbridge_matched_tokens(request, ucm_block_ids, num_prefix_hit_tokens)

        self.requests_meta[request.request_id].new_kv_cluster = new_kv_cluster
        self.requests_meta[request.request_id].load_chunk_meta = load_chunk_meta

        return ucm_prefix_hit_block_num, False

    def _generate_dispatch_meta(
        self, req_meta, new_tokens, vllm_block_ids, need_load=True
    ):
        hbm_hit_block_num = req_meta.hbm_hit_block_num
        total_hit_block_num = req_meta.total_hit_block_num
        ucm_block_ids = req_meta.ucm_block_ids
        req_meta.vllm_block_ids.extend(vllm_block_ids)

        load_ucm_block_ids, load_vllm_block_ids = [], []
        dump_ucm_block_ids, dump_vllm_block_ids = [], []
        if need_load:
            req_meta.current_vllm_block_ids = vllm_block_ids
            load_ucm_block_ids = ucm_block_ids[hbm_hit_block_num:total_hit_block_num]
            load_vllm_block_ids = vllm_block_ids[hbm_hit_block_num:total_hit_block_num]

        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids = ucm_block_ids[start_idx:end_idx]
            dump_vllm_block_ids = req_meta.vllm_block_ids[start_idx:end_idx]
            req_meta.token_processed += new_tokens

        return RequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
            req_meta.load_chunk_meta,
            req_meta.current_vllm_block_ids,
            need_load,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids[0]
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    vllm_block_ids,
                )

        # for cached request, there are 3 situation:
        # 1. chunked prefill: we only need dump
        # 2. resumed: we need to handle like new request
        # 3. TODO decode stage: nothing happened
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            # >= 0.9.2
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    if scheduled_cached_reqs.num_output_tokens[i] > 0:
                        self.kv_cluster_map.insert_cluster(req_meta.new_kv_cluster)
                    new_block_ids = []
                    if scheduled_cached_reqs.new_block_ids[i] != None:
                        new_block_ids = scheduled_cached_reqs.new_block_ids[i][0]
                    if hasattr(scheduled_cached_reqs, "resumed_from_preemption"):
                        resumed_from_preemption = (
                            scheduled_cached_reqs.resumed_from_preemption[i]
                        )
                    else:
                        resumed_from_preemption = (
                            request_id in scheduled_cached_reqs.resumed_req_ids
                        )
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        new_block_ids,
                        resumed_from_preemption
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        super().start_load_kv(forward_context, **kwargs)
        input_batch = kwargs.get("input_batch", None)
        positions = kwargs.get("positions", None)
        if input_batch is None or positions is None:
            return

        request_to_task: dict[str, Task] = {}
        start_temp_block_id = 0

        pending_copy_tasks = {}
        metadata = self._get_connector_metadata()

        self._process_compute_mask(forward_context, metadata, input_batch, positions)

        # 1. 记录所有需要执行的拷贝任务的元数据
        for request_id, request in metadata.request_meta.items():
            # decode 阶段 KV 已写入目标块，无需再加载外部 chunk
            if getattr(request, "need_load", False):
                continue
            load_chunk_meta = request.load_chunk_meta
            if load_chunk_meta is None:
                continue
            vllm_block_ids = request.vllm_block_ids
            req_idx = input_batch.req_id_to_index[request_id]

            for idx, load_chunks_blocks in enumerate(load_chunk_meta.load_chunks_blocks):
                chunk_len = load_chunk_meta.load_chunks_len[idx]
                logger.info_once(f"[kvbridge] req: {request_id}, chunk_len: {chunk_len}")
                if chunk_len == 0:
                    continue
                if self.tp_rank != 0 and not self.is_mla:
                    for i, load_chunks_block in enumerate(load_chunks_blocks):
                        load_chunks_blocks[i] = self.request_hasher(load_chunks_block)

                end_temp_block_id = start_temp_block_id + len(load_chunks_blocks)
                temp_block_ids = list(range(start_temp_block_id, end_temp_block_id))
                start_temp_block_id = end_temp_block_id

                total_ptrs = self.kv_cache_layout_kvb.extract_block_addrs(temp_block_ids)
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(load_chunks_blocks)

                task_id = request_id + " kvb-chunk: " + str(idx)
                try:
                    # 提交异步加载任务到暂存区
                    task = self.store.load_data(load_chunks_blocks, shard_indexs, total_ptrs)
                    request_to_task[task_id] = task

                    # 记录该块的拷贝参数
                    pending_copy_tasks[task_id] = {
                        'temp_start': temp_block_ids[0],
                        'temp_end': temp_block_ids[-1] + 1,
                        'dst_start_pos': load_chunk_meta.load_dst_chunks_start_positions[idx],
                        'chunk_len': chunk_len,
                        'token_offset': load_chunk_meta.load_src_offsets[idx],
                        'shift_p': load_chunk_meta.load_chunks_shift_positions[idx],
                        'vllm_block_ids': vllm_block_ids,
                        'req_idx': req_idx
                    }
                except RuntimeError as e:
                    logger.error(f"{task_id} submit load task error. {e}")

        # 2. 统一 Wait 所有的 IO 任务，确保数据写入 temp_block
        for task_id, task in request_to_task.items():
            try:
                self.store.wait(task)
                self._copy_temp_to_kvcaches(pending_copy_tasks[task_id], forward_context)
            except RuntimeError as e:
                logger.error(f"[KVBridge] {task_id} wait load task error. {e}")

        # 3. 所有 chunk 已写完目标 KV，清除元数据，避免后续调度步（decode/重调度）重复拷贝
        for request_id, request in metadata.request_meta.items():
            if getattr(request, "need_load", False) or request.load_chunk_meta is None:
                continue
            req_meta = self.requests_meta.get(request_id)
            if req_meta is not None:
                req_meta.load_chunk_meta = None

    def _copy_temp_to_kvcaches(self, copy_task, forward_context: "ForwardContext") -> None:
        temp_start = copy_task['temp_start']
        temp_end = copy_task['temp_end']
        dst_start_pos = copy_task['dst_start_pos']
        chunk_len = copy_task['chunk_len']
        token_offset = copy_task['token_offset']
        shift_p = copy_task.get('shift_p', 0)
        vllm_block_ids = copy_task['vllm_block_ids']
        req_idx = copy_task['req_idx']

        # 计算目标系统所需的 Block 数和对应的 vllm_block_ids
        n_blocks = chunk_len // self.block_size
        dst_start_block = dst_start_pos // self.block_size
        if dst_start_block < 0 or dst_start_block + n_blocks > len(vllm_block_ids):
            logger.warning(
                f"[kvbridge] dst block window [{dst_start_block}, {dst_start_block+n_blocks}) "
                f"out of vllm_block_ids(len={len(vllm_block_ids)}), skip chunk copy."
            )
            return
        dst_block_ids = vllm_block_ids[dst_start_block : dst_start_block + n_blocks]

        # 待拷贝 chunk 的相对位置差（RoPE 位移量），由 get_hit_meta 计算为 dst_start - src_start
        delta_positions = shift_p * torch.ones(chunk_len, dtype=torch.long, device=self.kv_cache_layout_kvb.device).view(n_blocks, self.block_size)

        # 遍历每一层进行转移
        for layer_name, temp_kv_tensors in self.kv_cache_layout_kvb.load_kvcaches.items():
            dst_kv_tensors = self.kv_caches[layer_name]

            # 兼容处理：支持 Tuple[Tensor, Tensor] (K, V) 或者是单 Tensor
            if isinstance(temp_kv_tensors, torch.Tensor):
                temp_kv_tensors = (temp_kv_tensors,)
                dst_kv_tensors = (dst_kv_tensors,)

            for temp_tensor, dst_tensor in zip(temp_kv_tensors, dst_kv_tensors):
                # 1. 切片取出当前任务在临时区对应的 Block
                chunk_temp = temp_tensor[temp_start : temp_end]

                # 2. 展平成 Token 序列: view(-1, num_head, head_dim)
                flat_temp = chunk_temp.view(-1, *chunk_temp.shape[2:])

                # 3. 按 src offset 精确截取需要的 Token 数据
                src_tokens = flat_temp[token_offset : token_offset + chunk_len]

                # 4. 重新折叠成目标系统的 Block 结构: view(N_blocks, block_size, num_head, head_dim)
                src_blocks = src_tokens.view(n_blocks, self.block_size, *dst_tensor.shape[2:])

                # 5. 用 RoPE 相对位置差修正 KV，使其匹配目标 chunk 的位置编码
                if shift_p != 0:
                    src_blocks = self.rotary_emb_3d.shift(src_blocks, delta_positions)

                # 6. 离散映射写入到目标 KV cache，PyTorch 底层自动分发散点存取
                dst_tensor[dst_block_ids] = src_blocks

    def _process_compute_mask(
        self,
        forward_context: "ForwardContext",
        metadata,
        input_batch,
        positions: torch.Tensor,
        ) -> None:
        first_layer_metadata = next(iter(forward_context.attn_metadata.values()))
        query_start_loc = first_layer_metadata.query_start_loc
        forward_context.compute_masks = {}
        for request_id, request in metadata.request_meta.items():
            load_chunk_meta = request.load_chunk_meta
            if load_chunk_meta is None:
                continue
            req_idx = input_batch.req_id_to_index[request_id]
            req_start_pos = int(query_start_loc[req_idx].item())
            req_end_pos = int(query_start_loc[req_idx + 1].item())
            position = positions[req_start_pos:req_end_pos].tolist()
            if not position:
                forward_context.compute_masks[req_idx] = np.zeros(0, dtype=np.int64)
                continue
            forward_context.compute_masks[req_idx] = np.zeros(len(position), dtype=np.int64)

            for idx, load_chunks_blocks in enumerate(load_chunk_meta.load_chunks_blocks):
                chunk_len = load_chunk_meta.load_chunks_len[idx]
                if chunk_len == 0:
                    continue
                dst_start_pos = load_chunk_meta.load_dst_chunks_start_positions[idx]
                dst_end_pos = dst_start_pos + chunk_len
                if dst_start_pos > position[-1] or dst_end_pos <= position[0]:
                    continue
                dst_start_idx = max(dst_start_pos - position[0], 0)
                dst_end_idx = min(dst_end_pos - position[0], len(position))
                forward_context.compute_masks[req_idx][dst_start_idx : dst_end_idx] = 1
                logger.info_once(f"dst_start_idx: {dst_start_idx}, chunk_len: {chunk_len}")


class UCMKvBridgeLayerWiseConnector(
    UCMKvBridgeDirectConnector, UCMLayerWiseConnector
):
    pass


class UCMKvBridgeConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
        self.connector: KVConnectorBase_V1
        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()
        logger.info(f"self.launch_config: {self.launch_config}")

        use_layerwise = (
            self.launch_config.get("use_layerwise", False)
            if self.launch_config is not None
            else False
        )
        if use_layerwise:
            self.connector = UCMKvBridgeLayerWiseConnector(vllm_config, role)
        else:
            self.connector = UCMKvBridgeDirectConnector(vllm_config, role)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        return self.connector.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        self.connector.update_state_after_alloc(request, blocks, num_external_tokens)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.connector.register_kv_caches(kv_caches)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return self.connector.build_connector_meta(scheduler_output)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        self.connector.bind_connector_metadata(connector_metadata)

    def has_connector_metadata(self) -> bool:
        return self.connector.has_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        self.connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self.connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        self.connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        self.connector.wait_for_save()

    def clear_connector_metadata(self) -> None:
        self.connector.clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self.connector.get_block_ids_with_load_errors()
