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
        self._run_func = self.empty_run
        self._provider = "torch"

        # preset
        self.input_tensor_info = None
        self.output_tensor_info = None
        self.input_tensor_size = 0
        self.output_tensor_size = 0
        self.tensor_size = 0
        self.read_bytes = 0
        self.write_bytes = 0
        self.io_bytes = 0
        self.algo_size = 0
        self.bus_size = 0
        self.calc_flops = 0

        self.prepare()

    
    def empty_run(self):
        raise NotImplementedError

        
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


    def get_provider(self):
        return self._provider

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
                tensor_mapping[key] = torch.zeros(
                    size=value.shape, 
                    dtype=value.dtype,
                    device=torch_device_name
                )
            for key, value in output_tensor_info.items():
                tensor_mapping[key] = torch.zeros(
                    size=value.shape, 
                    dtype=value.dtype,
                    device=torch_device_name
                )
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list



    def summary(self, latency_us):
        result_dict = {
            "provider": self.get_provider(),
            "arguments": self.args_dict,
            "targets": {}
        }
        if latency_us > 0:
            result_dict["targets"] = {
                "latency(us)": round(latency_us, 3),
                "mem_bw(GB/s)": round(self.io_bytes / latency_us / 1e3, 3),
                "algo_bw(GB/s)": round(self.algo_size / latency_us / 1e3, 3),
                "bus_bw(GB/s)": round(self.bus_size / latency_us / 1e3, 3),
                "calc_flops_power(tflops)": round(self.calc_flops / latency_us / 1e6, 3),
                "calc_mem_ratio": round(self.calc_flops / self.io_bytes, 3) if self.io_bytes != 0 else 0
            }
        return result_dict
