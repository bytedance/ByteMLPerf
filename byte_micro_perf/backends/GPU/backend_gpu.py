import os
import sys
import pathlib
from datetime import timedelta

import torch
import torch.distributed as dist

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BACKEND_DIR = FILE_DIR.parent
MICRO_PERF_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))


from core.backend import Backend
from core.op_mapping import DEFAULT_OP_MAPPING



OP_MAPPING = DEFAULT_OP_MAPPING.copy()



class BackendGPU(Backend):
    def __init__(self):
        super().__init__()

    """
    device management related
    """
    def get_torch_device_name(self):
        return "cuda"
    
    def get_device_name(self, index = 0):
        return torch.cuda.get_device_name(index)
    
    def get_device_properties(self, index = 0):
        return torch.cuda.get_device_properties(index)

    def get_mem_info(self, index = 0):
        return torch.cuda.mem_get_info(index)

    def get_device_count(self):
        device_count = torch.cuda.device_count()
        return device_count, list(range(device_count))
    
    def set_device(self, device_index : int):
        torch.cuda.set_device(device_index)

    def get_device(self):
        return torch.cuda.current_device()

    def device_synchronize(self):
        torch.cuda.synchronize()

    def empty_cache(self):
        torch.cuda.empty_cache()



    """
    ccl related
    """
    def get_dist_module(self):
        return dist
    
    def get_dist_backend(self):
        return "nccl"




    def core_perf(self, op_instance, warmup_iterations, prefer_iterations, tensor_list):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(warmup_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])

        self.device_synchronize()
        self.op_group_barrier(op_group=op_group, group_size=group_size)

        start_event.record()
        for i in range(prefer_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        end_event.record()

        self.device_synchronize()
        self.op_group_barrier(op_group=op_group, group_size=group_size)

        return start_event.elapsed_time(end_event) * 1e3 / prefer_iterations