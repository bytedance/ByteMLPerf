import sys
import pathlib
import torch


FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, calc_tensor_size, ceil_div
from core.op import BasicOp

"""
gemm ops
"""
class GemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        
        self.M = self.args_dict["M"]
        self.K = self.args_dict["K"]
        self.N = self.args_dict["N"]

        # bf16 * bf16 --> bf16
        # fp16 * fp16 --> fp16
        if self.dtype in ["bfloat16", "float16"]:
            self.torch_dtype = getattr(torch, self.dtype)
            self.out_dtype = self.torch_dtype
        # fp32 * fp32 --> fp32
        elif self.dtype == "float32":
            self.torch_dtype = torch.float32
            self.out_dtype = torch.float32
            # use float32 gemm
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        # fp32(tf32) * fp32(tf32) --> fp32
        elif self.dtype == "tfloat32":
            self.torch_dtype = torch.float32
            self.out_dtype = torch.float32
            # use tfloat32 gemm
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # int8 (+scale) * int8 (+scale) --> bf16
        elif self.dtype == "int8":
            self.torch_dtype = torch.int8
            self.out_dtype = torch.bfloat16
        else:
            raise NotImplementedError

        if self.dtype in ["float32", "tfloat32", "float16", "bfloat16"]:
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
                ),
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.M, self.N],
                    dtype=self.out_dtype,
                    device=self.backend.get_device(),
                )
            }
        elif self.dtype == "int8":
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
                    dtype=self.out_dtype,
                    device=self.backend.get_device(),
                )
            }
        
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        
        self.calc_flops = self.M * self.N * self.K * 2
        self._run_func = self.gemm_run


    def gemm_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        if self.dtype in ["float32", "tfloat32", "float16", "bfloat16"]:
            torch.matmul(a, b, out=c)
        else:
            raise NotImplementedError
        return c



class GemmFP8Op(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]

        if self.dtype != "float8_e4m3fn":
            raise NotImplementedError

        # fp8 (+fp32 scale) * fp8 (+fp32 scale) --> bf16
        self.torch_dtype = torch.float8_e4m3fn
        self.scale_dtype = torch.float32
        self.out_dtype = torch.bfloat16

        self.quant_group_size = self.args_dict["quant_group_size"]
        self.M = self.args_dict["M"]
        self.K = self.args_dict["K"]
        self.N = self.args_dict["N"]


        self.K_BLOCK = ceil_div(self.K, self.quant_group_size)
        self.N_BLOCK = ceil_div(self.N, self.quant_group_size)

        self.input_tensor_info = {
            "a": OpTensorInfo(
                shape=[self.M, self.K],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            ),
            "b": OpTensorInfo(
                shape=[self.N, self.K],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            ),
            "a_scale": OpTensorInfo(
                shape=[self.M, self.K_BLOCK],
                dtype=self.scale_dtype,
                device=self.backend.get_device(),
            ), 
            "b_scale": OpTensorInfo(
                shape=[self.N_BLOCK, self.K_BLOCK],
                dtype=self.scale_dtype,
                device=self.backend.get_device(),
            ),
        }
        self.output_tensor_info = {
            "c": OpTensorInfo(
                shape=[self.M, self.N],
                dtype=self.out_dtype,
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self.calc_flops = self.M * self.N * self.K * 2




class GroupGemmFP8Op(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]

        if self.dtype != "float8_e4m3fn":
            raise NotImplementedError  

        # fp8 (+fp32 scale) * fp8 (+fp32 scale) --> bf16
        self.torch_dtype = torch.float8_e4m3fn
        self.scale_dtype = torch.float32
        self.out_dtype = torch.bfloat16

        self.quant_group_size = self.args_dict["quant_group_size"]
        self.mode = self.args_dict["mode"]
        self.num_groups = self.args_dict["num_groups"]
        self.M = self.args_dict["M"]
        self.K = self.args_dict["K"]
        self.N = self.args_dict["N"]

        self.K_BLOCK = ceil_div(self.K, self.quant_group_size)
        self.N_BLOCK = ceil_div(self.N, self.quant_group_size)

        if self.mode == "contiguous":
            self.input_tensor_info = {
                "a": OpTensorInfo(
                    shape=[self.num_groups * self.M, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "b": OpTensorInfo(
                    shape=[self.num_groups, self.N, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ), 
                "a_scale": OpTensorInfo(
                    shape=[self.num_groups * self.M, self.K_BLOCK],
                    dtype=self.scale_dtype,
                    device=self.backend.get_device(),
                ),
                "b_scale": OpTensorInfo(
                    shape=[self.num_groups, self.N_BLOCK, self.K_BLOCK],
                    dtype=self.scale_dtype,
                    device=self.backend.get_device(),
                ),
                "m_indices": OpTensorInfo(
                    shape=[self.num_groups * self.M],
                    dtype=torch.int32,
                    device=self.backend.get_device(),
                )
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.num_groups * self.M, self.N],
                    dtype=self.out_dtype,
                    device=self.backend.get_device(),
                )
            }
        elif self.mode == "masked":
            self.input_tensor_info = {
                "a": OpTensorInfo(
                    shape=[self.num_groups, self.M, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "b": OpTensorInfo(
                    shape=[self.num_groups, self.N, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ), 
                "a_scale": OpTensorInfo(
                    shape=[self.num_groups, self.M, self.K_BLOCK],
                    dtype=self.scale_dtype,
                    device=self.backend.get_device(),
                ), 
                "b_scale": OpTensorInfo(
                    shape=[self.num_groups, self.N_BLOCK, self.K_BLOCK],
                    dtype=self.scale_dtype,
                    device=self.backend.get_device(),
                ), 
                "masked_m": OpTensorInfo(
                    shape=[self.num_groups],
                    dtype=torch.int32, 
                    device=self.backend.get_device(),
                )
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.num_groups, self.M, self.N],
                    dtype=self.out_dtype,
                    device=self.backend.get_device(),
                )
            }
        else:
            raise NotImplementedError

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self.calc_flops = self.num_groups * self.M * self.N * self.K * 2

