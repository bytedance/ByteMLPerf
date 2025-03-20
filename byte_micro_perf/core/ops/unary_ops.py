import sys
import pathlib
import torch
from functools import partial

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


"""
unary ops
"""
class CastOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        # support:
        # fp32: fp32 --> bf16
        # fp16: fp16 --> fp32
        # bf16: bf16 --> fp32
        if self.dtype == "float32":
            self.src_dtype = torch.float32
            self.dst_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.src_dtype = torch.float16
            self.dst_dtype = torch.float32
        elif self.dtype == "bfloat16":
            self.src_dtype = torch.bfloat16
            self.dst_dtype = torch.float32
        else:
            raise NotImplementedError
                
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.src_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.dst_dtype,
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

        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True,
            create_outputs=False
        )
        self._run_func = self.cast_run

    def cast_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = src.to(dtype=self.dst_dtype)
        return dst
        


class CosOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
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
        self._run_func = self.cos_run

    def cos_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.cos(src, out=dst)
        return dst
    


class ExpOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.exp_run

    def exp_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.exp(src, out=dst)
        return dst


class GeluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.gelu_run
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
    def gelu_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.gelu(src)
        return dst


class LogOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.log_run
    def log_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.log(src, out=dst)
        return dst

class SiluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.silu_run
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
    def silu_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.silu(src)
        return dst

class SinOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.sin_run
    def sin_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.sin(src, out=dst)
        return dst

class SqrtOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.sqrt_run
    def sqrt_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.sqrt(src, out=dst)
        return dst





