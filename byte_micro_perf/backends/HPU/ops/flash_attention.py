import itertools
import math
import os
import sys
import pathlib
from functools import partial
from time import time
from typing import Any, List, Optional
import torch

import habana_frameworks.torch as ht

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}

try:

    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    import vllm_hpu_extension.kernels as kernels
    import vllm_hpu_extension.ops as ops
    from vllm_hpu_extension.flags import enabled_flags
    from vllm_hpu_extension.utils import (Matmul, ModuleFusedSDPA, Softmax,
                                        VLLMKVCache)

    class FusedSDPAOP(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.mode == "prefill":

                self.input_tensor_info.update({
                    "q": OpTensorInfo(
                        shape=[self.batch_size, self.num_tokens // self.batch_size, self.q_head_num, self.head_dim], 
                        dtype=self.torch_dtype, 
                        device=self.backend.get_torch_device_name()
                    ), 
                    "k_cache": OpTensorInfo(
                        shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim], 
                        dtype=self.cache_torch_dtype, 
                        device=self.backend.get_torch_device_name(), 
                        creator=torch.empty
                    ), 
                    "v_cache": OpTensorInfo(
                        shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim], 
                        dtype=self.cache_torch_dtype, 
                        device=self.backend.get_torch_device_name(), 
                        creator=torch.empty
                    ),
                })

            elif self.mode == "decode":

                self.block_size=128

                self.input_tensor_info.update(
                    {
                        "q": OpTensorInfo(
                            shape=[
                                self.batch_size,
                                self.max_kv_len,
                                self.q_head_num,
                                self.head_dim,
                            ],
                            dtype=self.torch_dtype,
                            device=self.backend.get_torch_device_name(),
                        ),
                        "k_cache": OpTensorInfo(
                            shape=[
                                self.batch_size * self.max_kv_len // self.block_size
                                + 1,
                                self.block_size,
                                self.kv_head_num,
                                self.head_dim,
                            ],
                            dtype=self.cache_torch_dtype,
                            device=self.backend.get_torch_device_name(),
                            creator=torch.empty,
                        ),
                        "v_cache": OpTensorInfo(
                            shape=[
                                self.batch_size * self.max_kv_len // self.block_size
                                + 1,
                                self.block_size,
                                self.kv_head_num,
                                self.head_dim,
                            ],
                            dtype=self.cache_torch_dtype,
                            device=self.backend.get_torch_device_name(),
                            creator=torch.empty,
                        ),
                    }
                )

                self.k_cache = VLLMKVCache()
                self.v_cache = VLLMKVCache()
                self.matmul_qk_op = Matmul()
                self.matmul_av_op = Matmul()
                self.batch2block_matmul_op = Matmul()
                self.block2batch_matmul_op = Matmul()
                self.keys_fetch_func = self.k_cache.fetch_from_cache
                self.values_fetch_func = self.v_cache.fetch_from_cache

                (
                    self.block_list,
                    self.block_mapping,
                    self.block_bias,
                    self.block_scales,
                    self.block_groups,
                ) = set_flat_pa_inputs(
                    self.batch_size,
                    self.max_kv_len,
                    self.block_size,
                    self.dtype,
                    self.backend.get_torch_device_name(),
                )

            # currently not support prefill_session_cache mode
            # cause not support different q_lens
            if self.mode in ["prefill_session_cache"]:
                raise NotImplementedError("not support prefill_session_cache")

            # currently not support c8
            if self.dtype != self.cache_dtype:
                raise NotImplementedError("not support q_dtype != cache_dtype")

            if self.mode == "decode" and self.q_seq_len != 1:
                raise NotImplementedError("not support decode with query length more than 1")

        def flash_attention_run(self, tensor_mapping):
            # get pre-allocated tensors
            q = tensor_mapping["q"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cache_slot_ids = tensor_mapping["cache_slot_ids"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping.get("k_scale", None)
            v_scale = tensor_mapping.get("v_scale", None)

            if self.mode == "prefill" and self.cache_len == 0:
                out = FusedSDPA.apply(q, k_cache, v_cache, None, 0.0, True)
            else:

                out = ops.flat_pa(
                    q,
                    k_cache,
                    v_cache,
                    self.block_list,
                    self.block_mapping,
                    self.block_bias,
                    self.block_scales,
                    self.block_groups,
                    self.softmax_scale,
                    self.matmul_qk_op,
                    self.matmul_av_op,
                    self.batch2block_matmul_op,
                    self.block2batch_matmul_op,
                    self.keys_fetch_func,
                    self.values_fetch_func,
                )

            return out
    OP_MAPPING["fused_sdpa"] = FusedSDPAOP
except:
    pass


def flatten(in_list):
    return list(itertools.chain(*in_list))

def gather_list(input, indices, v):
    return [input[i] if i is not None else v for i in indices]

def set_flat_pa_inputs(num_seqs, seq_len, block_size, dtype, device): 
    
    max_num_blocks_per_seq = math.ceil(seq_len / block_size)
    
    # block_tables
    block_tables = []
    start_idx = 1
    idx = num_seqs * max_num_blocks_per_seq + 1
    for s in range( num_seqs):
        block_table = [i for i in range(start_idx, start_idx + max_num_blocks_per_seq)] + [idx + s]
        start_idx += max_num_blocks_per_seq        
        block_tables.append(block_table)
    #print(f"block_tables: {block_tables}")
    # block_list
    block_list = flatten(block_tables)
    
    # block_groups
    block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
    block_groups = flatten(block_groups) 
 
    VLLM_DECODE_BLOCK_BUCKET_STEP = 128
    #print(f"block_list len {len(block_list)} {block_list}")
    block_bucket_size = math.ceil(len(block_list) / VLLM_DECODE_BLOCK_BUCKET_STEP) * VLLM_DECODE_BLOCK_BUCKET_STEP
    #print(f"block_bucket_size {block_bucket_size}")
    indices: List[Any]
    indices = [None] * block_bucket_size
    for i, bid in enumerate(block_list):
        indices[bid] = i
    
    padding_fn = lambda tensor, pad_value: gather_list(
        tensor, indices, pad_value)    

    block_list = padding_fn(block_list, 0)
    block_groups = padding_fn(block_groups, -1)
 
    block_groups = torch.tensor(block_groups, dtype=torch.int, device=device)
    block_mapping = torch.nn.functional.relu(block_groups)
    block_mapping = torch.nn.functional.one_hot(block_mapping.to(torch.long),
                                                num_classes=num_seqs)
    
    block_bias = torch.zeros((block_bucket_size, block_size), dtype=dtype, device=device)
    
    # todo: here is eager mode, enable lazy mode
    # will it improve performance?
    lazy_mode = os.environ.get('PT_HPU_LAZY_MODE', 1)
    print(f"lazy_mode {lazy_mode}")
    #if not lazy_mode:
    oob_values = block_groups.lt(0)
    block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
    block_groups.masked_fill_(oob_values, num_seqs)
    #print(f"oob_values {oob_values}")
    
    # block_scales
    ones = torch.ones((block_mapping.size(0), ),
                  device=device,
                  dtype=torch.long)
    sums = ops.batch2block(ops.block2batch(ones, block_mapping), block_mapping)
    block_scales = torch.reciprocal(torch.maximum(ones, sums))
    block_scales = block_scales.to(dtype)

    block_list = torch.tensor(block_list, dtype=torch.int, device=device)
    block_mapping = block_mapping.to(dtype)
    
    return block_list, block_mapping, block_bias, block_scales, block_groups
