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
ccl ops
"""
class AllGatherOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        
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



class AllReduceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {}
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
        self.read_bytes = self.output_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = 2 * (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._concurrent = True
        self._run_func = self.all_reduce_run


    def all_reduce_run(self, tensor_mapping):
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_reduce(dst, op=dist_module.ReduceOp.SUM, group=self.op_group)
        return dst

    def is_concurrent():
        return True