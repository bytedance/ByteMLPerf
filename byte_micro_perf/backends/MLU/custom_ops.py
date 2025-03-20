import sys
import pathlib
import torch
import torch_mlu_ops

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp
from core.ops.gemm_ops import GemmOp
from core.ops.attn_ops import FlashAttentionOp

class MLUGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        

    def prepare(self):
        self.dtype = self.args_dict["dtype"]

        if self.dtype == "float32":
            self.torch_dtype = torch.float32
            torch.backends.mlu.matmul.allow_tf32 = False
            torch.backends.cnnl.allow_tf32 = False
        elif self.dtype == "tfloat32":
            self.torch_dtype = torch.float32
            torch.backends.mlu.matmul.allow_tf32 = True
            torch.backends.cnnl.allow_tf32 = True
        else:
            self.torch_dtype = getattr(torch, self.dtype)

        self.M = self.args_dict["M"]
        self.K = self.args_dict["K"]
        self.N = self.args_dict["N"]

        if self.torch_dtype == torch.int8:        
            self.input_tensor_info = {
                "a": OpTensorInfo(
                    shape=[self.M, self.K],
                    dtype=torch.int8,
                    device=self.backend.get_device(),
                ),
                "b": OpTensorInfo(
                    shape=[self.N, self.K],
                    dtype=torch.int8,
                    device=self.backend.get_device(),
                ), 
                "a_scale": OpTensorInfo(
                    shape=[self.M],
                    dtype=torch.float32,
                    device=self.backend.get_device(),
                ), 
                "b_scale": OpTensorInfo(
                    shape=[self.N],
                    dtype=torch.float32,
                    device=self.backend.get_device(),
                ), 
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.M, self.N],
                    dtype=torch.bfloat16,
                    device=self.backend.get_device(),
                )
            }
        elif self.torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            self.input_tensor_info = {
                "a": OpTensorInfo(
                    shape=[self.M, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "b": OpTensorInfo(
                    shape=[self.K, self.N],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.M, self.N],
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
        self.algo_size = 0  
        self.bus_size = 0

        self.calc_flops = self.M * self.N * self.K * 2
        self._run_func = self.gemm_run



    def gemm_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        if self.torch_dtype == torch.int8:
            a_scale = tensor_mapping["a_scale"]
            b_scale = tensor_mapping["b_scale"]
            c = torch_mlu_ops.smooth_quant_matmul(
                a, a_scale, 
                b, b_scale, 
                c.dtype, 
                None, None, 'none', 
                1., 0., 
                False, 
            )
        else:
            torch.matmul(a, b, out=c)
        return c



class MLUFlashAttentionOp(FlashAttentionOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.flash_attention_run


    def create_tensors(self, instance_num : int):
        # q, k, v, out
        cu_seq_len_q = [0]
        cu_seq_len_k = [0]
        for i in range(self.batch_size):
            cu_seq_len_q.append(cu_seq_len_q[-1] + self.q_seq_len)
            cu_seq_len_k.append(cu_seq_len_k[-1] + self.kv_seq_len)
        cu_seq_len_q = torch.tensor(cu_seq_len_q, dtype=torch.int32, device="mlu")
        cu_seq_len_k = torch.tensor(cu_seq_len_k, dtype=torch.int32, device="mlu")

        all_tensor_list = []
        for i in range(instance_num):
            tensor_mapping = {}
            for key, value in self.input_tensor_info.items():
                tensor_mapping[key] = torch.zeros(
                    size=value.shape, 
                    dtype=value.dtype,
                    device=value.device
                )
                if value.device == "cpu":
                    tensor_mapping[key] = tensor_mapping[key].pin_memory()
            for key, value in self.output_tensor_info.items():
                tensor_mapping[key] = torch.zeros(
                    size=value.shape, 
                    dtype=value.dtype,
                    device=value.device
                )
                if value.device == "cpu":
                    tensor_mapping[key] = tensor_mapping[key].pin_memory()

            tensor_mapping["cu_seq_len_q"] = cu_seq_len_q
            tensor_mapping["cu_seq_len_k"] = cu_seq_len_k

            all_tensor_list.append(tensor_mapping)

        return all_tensor_list



    def flash_attention_run(self, tensor_mapping):
        q = tensor_mapping["q"].reshape(self.q_seq_len, self.q_head_num, self.head_dim)
        k = tensor_mapping["k"].reshape(self.kv_seq_len, self.kv_head_num, self.head_dim)
        v = tensor_mapping["v"].reshape(self.kv_seq_len, self.kv_head_num, self.head_dim)

        return torch_mlu_ops.flash_attention(
            q, k, v, 
            None, 
            tensor_mapping["cu_seq_len_q"], 
            tensor_mapping["cu_seq_len_k"], 
            None, None, 
            max_seq_len_q=self.q_seq_len, 
            max_seq_len_kv=self.kv_seq_len, 
            softmax_scale=self.softmax_scale, 
            is_causal=self.is_causal, 
            return_lse=False
        )