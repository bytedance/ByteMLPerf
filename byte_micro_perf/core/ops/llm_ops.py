import sys
import pathlib
import torch
import random
from functools import partial

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[2])
)

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


"""
llm:        [batch_size, q_seq_len, hidden_size]
            --> [num_tokens, hidden_size]
"""
class HeadRMSNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError
        
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.num_tokens = self.args_dict["num_tokens"]
        self.head_num = self.args_dict["head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.head_offset = self.args_dict["head_offset"]
        self.head_norm = self.args_dict["head_norm"]

        self.epsilon = 1e-5

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.num_tokens, self.head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "weight": OpTensorInfo(
                shape=[self.head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.num_tokens, self.head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

    def head_rms_norm_run(self, tensor_mapping):
        raise NotImplementedError



"""
flash_attention
"""
class FlashAttentionOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        # parse args
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError
        
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.is_causal = self.args_dict.get("is_causal", True)
        if not self.is_causal:
            raise NotImplementedError
        
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.q_seq_len
        self.cache_len = self.kv_seq_len - self.q_seq_len
        self.softmax_scale = self.head_dim ** (-0.5)

        # define basic input/outpus, could be overrided
        self.input_tensor_info = {
            "q": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
            "k": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.kv_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
            "v": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.kv_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            )
        }
        self.output_tensor_info = {
            "out": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

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

        flops_ratio = (self.cache_len + 1 + self.kv_seq_len) * self.q_seq_len / 2 / (self.q_seq_len * self.kv_seq_len) if self.is_causal else 1
        self.calc_flops = (p_gemm_calc_flops + o_gemm_calc_flops) * flops_ratio

        # specify create input/output tensors func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )
        
        # specify run function
        self._run_func = self.flash_attention_run


    def flash_attention_run(self, tensor_mapping):
        raise NotImplementedError
    

class FlashAttentionSessionCache(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm_cache_rate"]:
            raise NotImplementedError

        self.is_causal = self.args_dict.get("is_causal", True)
        if not self.is_causal:
            raise NotImplementedError

        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)


        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.kv_seq_len = self.args_dict["kv_seq_len"]
        self.accum_q_seq_len = self.args_dict["accum_q_seq_len"]
        self.cache_rate = self.args_dict["cache_rate"]
        
        self.cache_len = self.kv_seq_len * self.cache_rate // 100
        self.q_seq_len = self.kv_seq_len - self.cache_len

        # batch_size * q_seq_len >= kv_seq_len
        self.batch_size = (self.accum_q_seq_len + self.q_seq_len - 1) // self.q_seq_len

        self.softmax_scale = self.head_dim ** (-0.5)


        # define basic input/outpus, could be overrided
        self.input_tensor_info = {
            "q": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
            "k_cache": OpTensorInfo(
                shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
            "v_cache": OpTensorInfo(
                shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
            "cache_seqlens": OpTensorInfo(
                    shape=[self.batch_size], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * self.kv_seq_len
                )
        }
        self.output_tensor_info = {
            "out": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

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

        flops_ratio = (self.cache_len + 1 + self.kv_seq_len) * self.q_seq_len / 2 / (self.q_seq_len * self.kv_seq_len) if self.is_causal else 1
        self.calc_flops = (p_gemm_calc_flops + o_gemm_calc_flops) * flops_ratio

        # specify create input/output tensors func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )
        
        # specify run function
        self._run_func = self.flash_attention_session_cache_run

    def flash_attention_session_cache_run(self, tensor_mapping):
        raise NotImplementedError






"""
flash_decoding
"""
class FlashDecodingOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        # parse args
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError
        
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16", "bfloat16_c8"]:
            raise NotImplementedError
        
        if self.dtype == "bfloat16":
            self.torch_dtype = getattr(torch, self.dtype)
        elif self.dtype == "bfloat16_c8":
            self.torch_dtype = torch.int8

        self.is_causal = self.args_dict.get("is_causal", True)
        if not self.is_causal:
            raise NotImplementedError
        
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.args_dict["kv_seq_len"]
    
        self.softmax_scale = self.head_dim ** (-0.5)

        """
        define basic input/outpus, could be overrided
        assume that store_kv_cache already executed
        kv_seq_len = q_seq_len + cache_len
        """
        if self.dtype == "bfloat16":
            self.input_tensor_info = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim], 
                    dtype=torch.bfloat16, 
                    device=self.backend.get_torch_device_name()
                ), 
                "k_cache": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim], 
                    dtype=torch.bfloat16, 
                    device=self.backend.get_torch_device_name()
                ), 
                "v_cache": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim], 
                    dtype=torch.bfloat16, 
                    device=self.backend.get_torch_device_name()
                ), 
                "cache_seqlens": OpTensorInfo(
                    shape=[self.batch_size], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * self.kv_seq_len
                )
            }
            self.output_tensor_info = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim], 
                    dtype=torch.bfloat16, 
                    device=self.backend.get_torch_device_name()
                )
            }
        elif self.dtype == "bfloat16_c8":
            self.input_tensor_info = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim], 
                    dtype=torch.bfloat16, 
                    device=self.backend.get_torch_device_name()
                ), 
                "k_cache": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim], 
                    dtype=torch.int8, 
                    device=self.backend.get_torch_device_name()
                ), 
                "v_cache": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim], 
                    dtype=torch.int8, 
                    device=self.backend.get_torch_device_name()
                ), 
                "k_scale": OpTensorInfo(
                    shape=[self.kv_head_num, self.head_dim], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name()
                ), 
                "v_scale": OpTensorInfo(
                    shape=[self.kv_head_num, self.head_dim], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name()
                ), 
                "cache_seqlens": OpTensorInfo(
                    shape=[self.batch_size], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * self.kv_seq_len
                )
            }
            self.output_tensor_info = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim], 
                    dtype=torch.bfloat16, 
                    device=self.backend.get_torch_device_name()
                )
            }
            

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

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

        self.calc_flops = p_gemm_calc_flops + o_gemm_calc_flops

        # specify create input/output tensors func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )
        
        # specify run function
        self._run_func = self.flash_decoding_run


    def flash_decoding_run(self, tensor_mapping):
        raise NotImplementedError
    


"""
quant_matmul and moe_group_gemm
"""
class QuantMatmulOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError
        
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["int8"]:
            raise NotImplementedError

        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=torch.int8, 
                device=self.backend.get_torch_device_name()
            ), 
            "weight": OpTensorInfo(
                shape=[self.hidden_size, self.new_hidden_size], 
                dtype=torch.int8, 
                device=self.backend.get_torch_device_name()
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name()
            ), 
            "weight_scale": OpTensorInfo(
                shape=[self.new_hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name()
            ), 
        }
        self.output_tensor_info = {
            "out": OpTensorInfo(
                shape=[self.num_tokens, self.new_hidden_size], 
                dtype=torch.int8, 
                device=self.backend.get_torch_device_name()
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # gemm:         [M, K] * [K, N] = [M, N]
        # scale:        [M, 1] * [1, N] --> [M, N]
        # cast:         [M, N] --> [M, N]
        # dequantize:   [M, N] --> [M, N]
        self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.new_hidden_size

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )

        self._run_func = self.quant_matmul_run

    def quant_matmul_run(self, tensor_mapping):
        raise NotImplementedError







def generate_balanced_random_list(target_sum, length):
    average = target_sum // length
    remainder = target_sum % length
    adjust_max = average // 10

    result = [average] * length
    if remainder > 0:
        indices = random.sample(range(length), remainder)
        for idx in indices:
            result[idx] += 1
    
    source_list = []
    for _ in range(length // 2):
        random_value = random.randint(0, adjust_max)
        source_list.append(random_value)
        source_list.append(-random_value)
    if length % 2 == 1:
        source_list.append(0)

    target_adjust = random.sample(source_list, length)
    for i in range(length):
        result[i] += target_adjust[i]

    return result




"""
each rank has:
- [num_tokens // world_size, hidden_size]

after gating/softmax/topk, each rank has: 
- [num_tokens // world_size * topk, hidden_size]

dispatch tokens to corresponding rank, dispatched tokens are possibly unbalanced

"""
class MoeDispatchTokensOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        
        self.world_size = self.args_dict["world_size"]
        assert self.world_size % 2 == 0
        dist_module = self.backend.get_dist_module()
        self.local_rank = dist_module.get_rank(group=self.op_group)


        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.topk = self.args_dict["topk"]


        # 4096 // 8 = 512 tokens per rank
        self.num_tokens_per_rank = self.num_tokens // self.world_size

        # 1. 512 * topk tokens on each rank
        # 2. 512 * topk tokens may be dispatched to other ranks
        # 3. split 512 * topk tokens to list
        # 4. all_to_all
        self.after_num_tokens = self.num_tokens_per_rank * self.topk

        random.seed(1)
        all_ranks_distributions = []
        for _ in range(self.world_size):
            all_ranks_distributions.append(
                generate_balanced_random_list(self.after_num_tokens, self.world_size)
            )


        self.input_split_sizes = all_ranks_distributions[self.local_rank]
        self.output_split_sizes = [data[self.local_rank] for data in all_ranks_distributions]

        self.input_token_num = self.after_num_tokens
        self.output_token_num = sum(self.output_split_sizes)

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.input_token_num, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.output_token_num, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )

        self._run_func = self.moe_dispatch_tokens_run

    def moe_dispatch_tokens_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_to_all_single(
            dst, src, 
            self.output_split_sizes, 
            self.input_split_sizes, 
            group=self.op_group
        )
        return dst
            

        









