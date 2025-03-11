import sys
import pathlib
import torch

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, OpSizeInfo, calc_tensor_size
from core.op import BasicOp

"""
gemm ops
"""
class GemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        
        if self.dtype == "tfloat32":
            self.torch_dtype = torch.float32
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
                    shape=[self.K, self.N],
                    dtype=torch.int8,
                    device=self.backend.get_device(),
                ), 
                "scale": OpTensorInfo(
                    shape=[self.M, self.N],
                    dtype=torch.bfloat16,
                    device=self.backend.get_device(),
                )
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
            scale = tensor_mapping["scale"]
            raise NotImplementedError
        else:
            torch.matmul(a, b, out=c)
        return c

