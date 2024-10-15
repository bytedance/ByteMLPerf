from typing import List, Optional, Tuple, Union
import torch
import rocmKernels as ops

from dataclasses import dataclass
# from vllm.utils import is_hip
def is_hip():
    return True
# if HAS_TRITON:
# from vllm.attention.ops.prefix_prefill import context_attention_fwd

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512 if not is_hip() else 1024
_PARTITION_SIZE_ROCM = 512
_DEVICE_PROPERTIES = torch.cuda.get_device_properties("cuda")
_ON_NAVI = hasattr(_DEVICE_PROPERTIES, "gcnArchName") and \
            "gfx1" in torch.cuda.get_device_properties("cuda").gcnArchName


# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    ops.paged_attention_v1(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
        k_scale, v_scale, tp_rank, blocksparse_local_blocks,
        blocksparse_vert_stride, blocksparse_block_size,
        blocksparse_head_sliding_step)


def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    ops.paged_attention_v2(
        out, exp_sum, max_logits, tmp_out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step)


def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    ops.paged_attention_rocm(out, exp_sum, max_logits, tmp_out, query,
        key_cache, value_cache, num_kv_heads,
        scale, block_tables, seq_lens,
        block_size, max_seq_len, alibi_slopes,
        kv_cache_dtype, k_scale, v_scale)

@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


def _use_rocm_custom_paged_attention(qtype: torch.dtype, head_size: int,
                                    block_size: int, gqa_ratio: int,
                                    max_seq_len: int) -> bool:
    # rocm custom page attention not support on navi (gfx1*)
    return (not _ON_NAVI and (qtype == torch.half or qtype == torch.bfloat16)
            and (head_size == 64 or head_size == 128)
            and (block_size == 16 or block_size == 32)
            and (gqa_ratio >= 1 and gqa_ratio <= 16) and max_seq_len <= 65536)

class PagedAttention:
    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        # Whether to use rocm custom paged attention or not
        num_seqs, num_heads, head_size = query.shape
        block_size = value_cache.shape[3]
        gqa_ratio = num_heads // num_kv_heads
        use_custom = _use_rocm_custom_paged_attention(
            query.dtype, head_size, block_size, gqa_ratio,
            max_seq_len)
        output = torch.empty_like(query)
        if use_custom:
            max_num_partitions = (
                (max_seq_len + _PARTITION_SIZE_ROCM - 1) //
                _PARTITION_SIZE_ROCM)
            assert _PARTITION_SIZE_ROCM % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            out = output
            ops.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        else:
            max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                                _PARTITION_SIZE)
            if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
                # use blocksparse paged attention
                block_size = value_cache.size(-1)
                assert (blocksparse_block_size > 0 and
                        blocksparse_block_size % block_size == 0), \
                    (f"{blocksparse_block_size=} needs to be a multiple of"
                    f"{block_size=} used in block_tables.")

            # NOTE(woosuk): We use a simple heuristic to decide whether to use
            # PagedAttention V1 or V2. If the number of partitions is 1, we use
            # V1 to avoid the overhead of reduction. Also, if the number of
            # sequences or heads is large, we use V1 since there is enough work
            # to parallelize.
            # TODO(woosuk): Tune this heuristic.
            # For context len > 8192, use V2 kernel to avoid shared memory shortage.
            use_v1 = (max_seq_len <= 8192
                    and (max_num_partitions == 1 or num_seqs * num_heads > 512))

            if use_v1:
                # Run PagedAttention V1.
                ops.paged_attention_v1(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    num_kv_heads,
                    scale,
                    block_tables,
                    seq_lens,
                    block_size,
                    max_seq_len,
                    alibi_slopes,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                    tp_rank,
                    blocksparse_local_blocks,
                    blocksparse_vert_stride,
                    blocksparse_block_size,
                    blocksparse_head_sliding_step,
                )
            else:
                # Run PagedAttention V2.
                assert _PARTITION_SIZE % block_size == 0
                tmp_output = torch.empty(
                    size=(num_seqs, num_heads, max_num_partitions, head_size),
                    dtype=output.dtype,
                    device=output.device,
                )
                exp_sums = torch.empty(
                    size=(num_seqs, num_heads, max_num_partitions),
                    dtype=torch.float32,
                    device=output.device,
                )
                max_logits = torch.empty_like(exp_sums)
                ops.paged_attention_v2(
                    output,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    num_kv_heads,
                    scale,
                    block_tables,
                    seq_lens,
                    block_size,
                    max_seq_len,
                    alibi_slopes,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                    tp_rank,
                    blocksparse_local_blocks,
                    blocksparse_vert_stride,
                    blocksparse_block_size,
                    blocksparse_head_sliding_step,
                )
        return output

    # @staticmethod
    # def forward_prefix(
    #     query: torch.Tensor,
    #     key: torch.Tensor,
    #     value: torch.Tensor,
    #     kv_cache_dtype: str,
    #     key_cache: torch.Tensor,
    #     value_cache: torch.Tensor,
    #     block_tables: torch.Tensor,
    #     query_start_loc: torch.Tensor,
    #     seq_lens_tensor: torch.Tensor,
    #     context_lens: torch.Tensor,
    #     max_query_len: int,
    #     alibi_slopes: Optional[torch.Tensor],
    #     sliding_window: Optional[int],
    #     k_scale: float,
    #     v_scale: float,
    # ) -> torch.Tensor:
    #     output = torch.empty_like(query)
    #     context_attention_fwd(
    #         query,
    #         key,
    #         value,
    #         output,
    #         kv_cache_dtype,
    #         key_cache,
    #         value_cache,
    #         block_tables,
    #         # query_start_loc is (batch_size + 1,)
    #         query_start_loc[:-1],
    #         seq_lens_tensor,
    #         context_lens,
    #         max_query_len,
    #         k_scale,
    #         v_scale,
    #         alibi_slopes,
    #         sliding_window,
    #     )
    #     return output

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)