import sys
import pathlib
import torch
import random

from typing import List, Dict, Union, Tuple

from flash_attn_interface import flash_attn_func
from flash_mla import flash_mla_with_kvcache, get_mla_metadata
import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor


FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, OpSizeInfo, calc_tensor_size
from core.op import BasicOp
from core.ops.gemm_ops import GemmOp, GemmFP8Op
from core.ops.attn_ops import FlashAttentionOp


"""
gemm ops
"""
class GPUGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        if self.dtype == "float32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif self.dtype == "tfloat32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True





FP8_E4M3_MAX = 448.0  # Maximum representable value in FP8 E4M3 format

def per_token_cast_to_fp8(x: torch.Tensor, group_size=128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % group_size == 0
    m, n = x.shape
    x_view = x.view(m, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (
        (x_view * (FP8_E4M3_MAX / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n),
        (x_amax / FP8_E4M3_MAX).view(m, -1)
    )

def per_block_cast_to_fp8(x: torch.Tensor, group_size=128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, group_size) * group_size, ceil_div(n, group_size) * group_size), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, group_size, x_padded.size(1) // group_size, group_size)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (FP8_E4M3_MAX / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / FP8_E4M3_MAX).view(x_view.size(0), x_view.size(2))

def construct(m: int, k: int, n: int, group_size, device) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    y = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    x_fp8, y_fp8 = per_token_cast_to_fp8(x, group_size), per_block_cast_to_fp8(y, group_size)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out




class GPUGemmFP8Op(GemmFP8Op):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        self._custom_run = True
        self._run_func = self.gemm_fp8_run


    def gemm_fp8_run(self):
        def test_func():
            x_fp8, y_fp8, out = construct(self.M, self.K, self.N, self.quant_group_size, self.backend.get_torch_device_name())
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t * 1e6




"""
attn_ops
"""
class GPUFlashAttentionOp(FlashAttentionOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.flash_attention_run

    def flash_attention_run(self, tensor_mapping):
        q = tensor_mapping["q"]
        k = tensor_mapping["k"]
        v = tensor_mapping["v"]        
        return flash_attn_func(q, k, v, causal=self.is_causal)



class GPUFlashMLAOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        # llm args
        self.args_type = self.args_dict["args_type"]
        if self.args_type != "llm":
            raise NotImplementedError

        # llm phase: prefill or decode
        self.phase = self.args_dict["phase"]
        if self.phase not in ["prefill", "decode"]:
            raise NotImplementedError

        # dtype: bfloat16
        self.dtype = self.args_dict["dtype"]
        if self.dtype != "bfloat16":
            raise NotImplementedError
        self.torch_dtype = torch.bfloat16
        self.torch_dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()


        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.args_dict["kv_seq_len"]
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.qk_dim_size = self.args_dict["qk_dim_size"]
        self.v_dim_size = self.args_dict["v_dim_size"]

        self.is_causal = self.args_dict["is_causal"]
        if not self.is_causal:
            raise NotImplementedError

        self.varlen = self.args_dict["varlen"]
        if self.varlen:
            raise NotImplementedError



        # q: [batch_size, q_seq_len, q_head_num, qk_dim_size]
        self.q = torch.randn(
            self.batch_size, self.q_seq_len, self.q_head_num, self.qk_dim_size,
            dtype=self.torch_dtype,
            device=self.backend.get_torch_device_name()
        )

        # prefill, not absorb weight, use flash_attention
        if self.phase == "prefill":
            self.k = torch.randn(
                self.batch_size, self.kv_seq_len, self.kv_head_num, self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            )
            self.v = torch.randn(
                self.batch_size, self.kv_seq_len, self.kv_head_num, self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            )

            self.input_tensor_size = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ), 
                "k": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ),
                "v": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                )
            }
            self.output_tensor_size = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                )
            }

            self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_size.values()])
            self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_size.values()])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            self.attn_ratio = (1 + self.kv_seq_len) / 2 / self.kv_seq_len
            self.calc_flops = self.batch_size * self.q_head_num * self.q_seq_len * self.kv_seq_len * (self.qk_dim_size + self.v_dim_size) * 2 * self.attn_ratio


        # decode, absorb weight, use flash_mla
        elif self.phase == "decode":
            self.cache_seqlens = torch.full(
                (self.batch_size,), 
                self.kv_seq_len, 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name()
            )
            self.total_seqlens = self.cache_seqlens.sum().item()
            self.mean_seqlens = self.cache_seqlens.float().mean().int().item()
            self.max_seqlen = self.cache_seqlens.max().item()
            self.max_seqlen_pad = (self.max_seqlen + 255) // 256 * 256

            self.block_size = 64
            self.block_table = torch.arange(
                self.batch_size * self.max_seqlen_pad // self.block_size,
                dtype=torch.int32,
                device=self.backend.get_torch_device_name()
            ).view(self.batch_size, self.max_seqlen_pad // self.block_size)

            self.blocked_k = torch.randn(
                self.block_table.numel(), self.block_size, self.kv_head_num, self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            )
            for i in range(self.batch_size):
                self.blocked_k.view(self.batch_size, self.max_seqlen_pad, self.kv_head_num, self.qk_dim_size)[i, self.cache_seqlens[i].item():] = (
                    float("nan")
                )
            self.tile_scheduler_metadata, self.num_splits = get_mla_metadata(
                self.cache_seqlens, self.q_seq_len * self.q_head_num // self.kv_head_num, self.kv_head_num
            )

            # q:            [batch_size, q_seq_len, q_head_num, qk_dim_size]
            # blocked_k:    [batch_size * max_seqlen_pad // block_size, block_size, kv_head_num, qk_dim_size]
            # block_table:  [batch_size, max_seqlen_pad // block_size]
            # cache_seqlens:[batch_size]
            self.input_tensor_size = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ), 
                "blocked_k": OpTensorInfo(
                    shape=[self.block_table.numel(), self.block_size, self.kv_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ),
                "block_table": OpTensorInfo(
                    shape=[self.batch_size, self.max_seqlen_pad // self.block_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name()
                ),
                "cache_seqlens": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name()
                )
            }

            # out:          [batch_size, q_seq_len, q_head_num, v_dim_size]
            # softmax_lse   [batch_size, q_head_num, q_seq_len]
            self.output_tensor_size = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.v_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ),
                "softmax_lse": OpTensorInfo(
                    shape=[self.batch_size, self.q_head_num, self.q_seq_len],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                )
            }
            self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_size.values()])
            self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_size.values()])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            # q + kv_compress, ignore block_table and cache_seqlens
            self.read_bytes = \
                (self.batch_size * self.q_seq_len * self.q_head_num * self.qk_dim_size + \
                self.total_seqlens * self.kv_head_num * self.qk_dim_size) * self.torch_dtype_size
            # out + softmax_lse
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            # q * k, p * v
            self.calc_flops = self.total_seqlens * self.q_head_num * self.q_seq_len * (self.qk_dim_size + self.v_dim_size) * 2



        self._run_func = self.flash_mla_run


    def create_tensors(self, instance_num : int):
        all_tensor_list = []
        for i in range(instance_num):
            tensor_mapping = {}
            if self.phase == "prefill":
                tensor_mapping["q"] = self.q.clone()
                tensor_mapping["k"] = self.k.clone()
                tensor_mapping["v"] = self.v.clone()
            elif self.phase == "decode":
                tensor_mapping["q"] = self.q.clone()
                tensor_mapping["blocked_k"] = self.blocked_k.clone()
                tensor_mapping["block_table"] = self.block_table.clone()
                tensor_mapping["cache_seqlens"] = self.cache_seqlens.clone()
            all_tensor_list.append(tensor_mapping)
        return all_tensor_list



    @torch.inference_mode()
    def flash_mla_run(self, tensor_mapping):
        if self.phase == "prefill":
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            v = tensor_mapping["v"]
            return flash_attn_func(q, k, v, causal=self.is_causal)
        elif self.phase == "decode":
            q = tensor_mapping["q"]
            blocked_k = tensor_mapping["blocked_k"]
            block_table = tensor_mapping["block_table"]
            cache_seqlens = tensor_mapping["cache_seqlens"]
            return_vals = flash_mla_with_kvcache(
                q,
                blocked_k,
                block_table,
                cache_seqlens,
                self.v_dim_size,
                self.tile_scheduler_metadata,
                self.num_splits,
                causal=self.is_causal,
            )
            return return_vals
        
