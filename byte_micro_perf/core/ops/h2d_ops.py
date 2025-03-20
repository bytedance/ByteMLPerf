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
h2d ops
"""
class Host2DeviceOp(BasicOp):
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
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device="cpu"
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = 0
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = 0
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.tensor_size

        self.calc_flops = 0

        self._run_func = self.host2device_run


    def host2device_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst

    def is_concurrent():
        return True


class Device2HostOp(BasicOp):
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
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device="cpu"
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = 0
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.tensor_size

        self.calc_flops = 0

        self._run_func = self.device2host_run


    def device2host_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst

    def is_concurrent():
        return True


class Device2DeviceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
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

        self.algo_size = 0
        self.bus_size = 0

        self.calc_flops = 0

        self._run_func = self.device2device_run

    def device2device_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst