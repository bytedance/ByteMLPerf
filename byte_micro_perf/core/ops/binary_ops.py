import sys
import pathlib
import torch

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


"""
binary ops
"""
class AddOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "a": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_device(), 
            ), 
            "b": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_device(), 
            )
        }
        self.output_tensor_info = {
            "c": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
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

        self.calc_flops = self.batch_size * self.dim_size 

        self._run_func = self.add_run

    def add_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = torch.add(a, b, out=c)
        return c

class SubOp(AddOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.sub_run
    def sub_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = torch.sub(a, b, out=c)
        return c

class MulOp(AddOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.mul_run
    def mul_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = torch.mul(a, b, out=c)
        return c
    
class DivOp(AddOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.div_run
    def div_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = torch.div(a, b, out=c)
        return c
