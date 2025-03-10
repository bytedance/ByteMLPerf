import sys
import pathlib
import torch
import random
import triton

from flash_attn_interface import flash_attn_func
from flash_mla import flash_mla_with_kvcache, get_mla_metadata


FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))


from core.op import BasicOp
from core.ops.gemm_ops import GemmOp


"""
gemm ops
"""
class GPUGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        
        if self.dtype == "tfloat32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif self.dtype == "float32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


"""
attn ops
"""


"""
attn_ops
"""
class GPUFlashAttentionOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]

        if self.dtype == "bfloat16": 
            self.torch_dtype = torch.bfloat16
        else:
            raise NotImplementedError

        self.is_causal = self.args_dict["is_causal"]
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        self.batch_size = self.args_dict["batch_size"]
        
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.args_dict["kv_seq_len"]

        if self.torch_dtype == torch.bfloat16:
            self.input_tensor_info = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "k": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "v": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
            self.output_tensor_info = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )    
            }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.output_tensor_info = {}

        p_gemm_b = self.batch_size * self.q_head_num
        p_gemm_m = self.q_seq_len
        p_gemm_k = self.head_dim
        p_gemm_n = self.kv_seq_len
        p_gemm_calc_flops = p_gemm_b * p_gemm_m * p_gemm_k * p_gemm_n * 2

        o_gemm_b = self.batch_size * self.q_head_num
        o_gemm_m = self.q_seq_len
        o_gemm_k = self.kv_seq_len
        o_gemm_n = self.head_dim
        o_gemm_calc_flops = o_gemm_b * o_gemm_m * o_gemm_k * o_gemm_n * 2


        flops_ratio = (1 + self.kv_seq_len) * self.q_seq_len / 2 / (self.q_seq_len * self.kv_seq_len) if self.is_causal else 1
        self.calc_flops = (p_gemm_calc_flops + o_gemm_calc_flops) * flops_ratio

        self._run_func = self.flash_attention_run

    def flash_attention_run(self, tensor_mapping):
        q = tensor_mapping["q"]
        k = tensor_mapping["k"]
        v = tensor_mapping["v"]
        
        out = flash_attn_func(q, k, v, causal=self.is_causal)

        return out




class GPUFlashMLAOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]

        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            raise NotImplementedError

        self.h_kv = self.args_dict["h_kv"]
        self.d = self.args_dict["d"]
        self.dv = self.args_dict["dv"]
        self.causal = self.args_dict["causal"]

        self.b = self.args_dict["b"]
        self.s = self.args_dict["s"]
        self.h_q = self.args_dict["h_q"]
        self.s_q = self.args_dict["s_q"]
        self.varlen = self.args_dict["varlen"]

        self.mean_sk = self.s


        self.cache_seqlens = torch.full((self.b,), self.mean_sk, dtype=torch.int32, device=self.backend.get_device())
        if self.varlen:
            for i in range(self.b):
                cache_seqlens[i] = max(random.normalvariate(self.mean_sk, self.mean_sk / 2), self.s_q)

        self.total_seqlens = self.cache_seqlens.sum().item()
        self.mean_seqlens = self.cache_seqlens.float().mean().int().item()
        self.max_seqlen = self.cache_seqlens.max().item()
        self.max_seqlen_pad = triton.cdiv(self.max_seqlen, 256) * 256

        self.q = torch.randn(
            self.b, self.s_q, self.h_q, self.d, 
            dtype=self.torch_dtype, 
            device=self.backend.get_device()
        )
        self.block_size = 64
        self.block_table = torch.arange(
            self.b * self.max_seqlen_pad // self.block_size, 
            dtype=torch.int32, 
            device=self.backend.get_device()
        ).view(self.b, self.max_seqlen_pad // self.block_size)

        self.blocked_k = torch.randn(
            self.block_table.numel(), self.block_size, self.h_kv, self.d, 
            dtype=self.torch_dtype,
            device=self.backend.get_device()
        )

        for i in range(self.b):
            self.blocked_k.view(self.b, self.max_seqlen_pad, self.h_kv, self.d)[i, self.cache_seqlens[i].item():] = (
                float("nan")
            )
        self.blocked_v = self.blocked_k[..., :self.dv]

        self.tile_scheduler_metadata, self.num_splits = get_mla_metadata(
            self.cache_seqlens, self.s_q * self.h_q // self.h_kv, self.h_kv
        )

        self.input_tensor_info = {}
        self.output_tensor_info = {}

        self.tensor_size = (self.total_seqlens * self.h_kv * self.d + self.b * self.s_q * self.h_q * self.d + self.b * self.s_q * self.h_q * self.dv) * \
                           torch.tensor([], dtype=self.torch_dtype).element_size()
        self.io_bytes = self.tensor_size
        self.calc_flops = self.s_q * self.total_seqlens * self.h_q * (self.d + self.dv) * 2

        self._run_func = self.flash_mla_run



    def create_tensors(self, instance_num : int):
        all_tensor_list = []
        for i in range(instance_num):
            tensor_mapping = {}

            tensor_mapping["q"] = self.q.clone()
            tensor_mapping["blocked_k"] = self.blocked_k.clone()
            tensor_mapping["block_table"] = self.block_table.clone()
            tensor_mapping["cache_seqlens"] = self.cache_seqlens.clone()
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list


    @torch.inference_mode()
    def flash_mla_run(self, tensor_mapping):
        q = tensor_mapping["q"]
        blocked_k = tensor_mapping["blocked_k"]
        block_table = tensor_mapping["block_table"]
        cache_seqlens = tensor_mapping["cache_seqlens"]
        return_vals = flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            self.dv,
            self.tile_scheduler_metadata,
            self.num_splits,
            causal=self.causal,
        )
        return return_vals
        
