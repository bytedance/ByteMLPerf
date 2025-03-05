import sys
import pathlib
import torch


FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, OpSizeInfo, calc_tensor_size
from core.op import BasicOp

class AddOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        self.dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()

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




class GemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)


    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        
        if self.dtype == "tfloat32":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = getattr(torch, self.dtype)
        self.dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()

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





class AllGatherOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        self.dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size // self.world_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
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

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0


        self._concurrent = True
        self._run_func = self.all_gather_run


    def all_gather_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_gather_into_tensor(dst, src, group=self.op_group)
        return dst


    def is_concurrent():
        return True





DEFAULT_OP_MAPPING = {
    # binary_ops
    "add": AddOp, 
    "sub": SubOp,
    "mul": MulOp,
    "div": DivOp,

    # gemm_ops
    "gemm": GemmOp,

    # xccl_ops
    "all_gather": AllGatherOp
}