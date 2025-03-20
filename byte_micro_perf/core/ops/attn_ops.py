import sys
import pathlib
import torch

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


"""
attn ops
"""
class FlashAttentionOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        # llm args
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "llm":
            raise NotImplementedError

        # llm phase: prefill or decode
        self.phase = self.args_dict["phase"]
        if self.phase not in ["prefill"]:
            raise NotImplementedError

        # dtype: bfloat16
        self.dtype = self.args_dict["dtype"]
        if self.dtype != "bfloat16":
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)
        self.torch_dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()


        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.args_dict["kv_seq_len"]
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        
        self.is_causal = self.args_dict["is_causal"]
        if not self.is_causal:
            raise NotImplementedError

        self.softmax_scale = self.head_dim ** (-0.5)


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






