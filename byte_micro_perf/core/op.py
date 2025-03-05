import sys
import time
import torch
import pathlib
from typing import List
from collections import namedtuple

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, OpSizeInfo, calc_tensor_size


class BasicOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        self.args_dict = args_dict
        self.backend = backend
        
        self.op_group = kwargs.get("op_group", None)
        self.group_size = kwargs.get("group_size", 1)

        # custom config for backend and op
        self._custom_run = False
        self._run_func = None

        # preset
        self.input_tensor_info = None
        self.output_tensor_info = None
        self.calc_flops = None

        


    @staticmethod
    def is_concurrent():
        return False


    def get_tensor_info(self):
        return self.input_tensor_info, self.output_tensor_info
        
    def get_size_info(self):
        return OpSizeInfo(
            input_tensor_size=self.input_tensor_size,
            output_tensor_size=self.output_tensor_size,
            tensor_size=self.tensor_size,
            read_bytes=self.read_bytes,
            write_bytes=self.write_bytes,
            io_bytes=self.io_bytes,
            algo_size=self.algo_size,
            bus_size=self.bus_size
        )

    def get_flops(self):
        return self.calc_flops



    def is_custom_run(self):
        return self._custom_run

    def core_run(self, *args, **kwargs):
        return self._run_func(*args, **kwargs)

    

    def prepare(self):
        pass

    def create_tensors(self, instance_num : int):
        input_tensor_info, output_tensor_info = self.get_tensor_info()
        torch_device_name = self.backend.get_torch_device_name()

        all_tensor_list = []
        for i in range(instance_num):
            tensor_mapping = {}
            for key, value in input_tensor_info.items():
                tensor_mapping[key] = torch.empty(
                    size=value.shape
                ).to(dtype=value.dtype, device=torch_device_name)
            for key, value in output_tensor_info.items():
                tensor_mapping[key] = torch.empty(
                    size=value.shape
                ).to(dtype=value.dtype, device=torch_device_name)
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list



    def summary(self, latency_us):
        result_dict = {
            "arguments": self.args_dict,
            "targets": {
                "latency(us)": latency_us,
                "qps": 1000000 / latency_us, 
                "mem_bw(GB/s)": self.io_bytes / (latency_us * 1e-6) / 1e9,
                "algo_bw(GB/s)": self.algo_size / (latency_us * 1e-6) / 1e9,
                "bus_bw(GB/s)": self.bus_size / (latency_us * 1e-6) / 1e9,
                "calc_flops_power(tflops)": self.calc_flops / (latency_us * 1e-6) / 1e12, 
                "calc_mem_ratio": self.calc_flops / self.io_bytes
            }
        }

        return result_dict
