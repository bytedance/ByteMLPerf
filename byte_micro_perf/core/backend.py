import os
import sys
import math
import time
import pathlib
from datetime import timedelta
from abc import ABC, abstractmethod
from typing import List

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger


class Backend(ABC):
    def __init__(self):
        pass

    def __del__(self):
        self.destroy_process_group()


    """
    device management related
    """
    @abstractmethod
    def get_torch_device_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_device_name(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_device_properties(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_mem_info(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_device_count(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_device(self, index: int):
        raise NotImplementedError
    
    @abstractmethod
    def get_device(self):
        raise NotImplementedError
    
    @abstractmethod
    def device_synchronize(self):
        raise NotImplementedError
    
    @abstractmethod
    def empty_cache(self):
        raise NotImplementedError

    
    """
    ccl related
    """
    @abstractmethod
    def get_dist_module(self):
        raise NotImplementedError

    @abstractmethod
    def get_dist_backend(self):
        raise NotImplementedError
    

    def initialize_ccl(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "49373"
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist_module = self.get_dist_module()
        dist_backend_name = self.get_dist_backend()

        dist_module.init_process_group(
            backend=dist_backend_name,
            world_size=world_size,
            rank=rank, 
            timeout=timedelta(seconds=1800)
        )
        return True
    

    def new_group(self, ranks):
        dist_module = self.get_dist_module()
        
        if dist_module.is_initialized():
            return dist_module.new_group(ranks)
        else:
            return None


    def op_group_barrier(self, op_group=None, group_size=1):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized() and group_size > 1:
            dist_module.barrier(group=op_group)

    
    def destroy_process_group(self):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized():
            dist_module.destroy_process_group()
    


    def core_perf(self, op_instance, warmup_iterations, prefer_iterations, tensor_list):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        for i in range(warmup_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])

        self.device_synchronize()
        self.op_group_barrier(op_group=op_group, group_size=group_size)

        start_time = time.perf_counter_ns()
        for i in range(prefer_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])

        self.device_synchronize()
        self.op_group_barrier(op_group=op_group, group_size=group_size)
        end_time = time.perf_counter_ns()
        return (end_time - start_time) / 1e3 / prefer_iterations
            


    
    def perf(self, op_instance):
        if op_instance.is_custom_run():
            latency_us = op_instance.core_run()
            return latency_us
        
    

        # op
        op_size_info = op_instance.get_size_info()
        tensor_size = op_size_info.tensor_size

        # device
        device_mem_info = self.get_mem_info()
        avail_memory = device_mem_info[0]

        # assume
        assume_cache_size = 1 * (1024 ** 3)
        assume_avail_bytes = int(avail_memory * 0.9)


        latency_us = 0.
        try:
            max_data_cnt = 1
            if not type(op_instance).is_concurrent():
                if tensor_size > assume_avail_bytes:
                    raise RuntimeError("Not enough memory to run the op")
                elif 2 * tensor_size > assume_avail_bytes:
                    max_data_cnt = 1
                elif tensor_size > assume_cache_size:
                    max_data_cnt = 2
                else:
                    max_data_cnt = min(math.floor(assume_avail_bytes / tensor_size), 32)

            tensor_list = op_instance.create_tensors(max_data_cnt)
            latency_us = self.core_perf(op_instance, 2, 2, tensor_list)
            prefer_iters = min(max(int(1000000 / latency_us), 2), 32)

            if op_instance.group_size > 1:
                dist_module = self.get_dist_module()
                prefer_iters_list = [None for _ in range(op_instance.group_size)]
                dist_module.all_gather_object(prefer_iters_list, prefer_iters, group=op_instance.op_group)
                prefer_iters = max(prefer_iters_list)
                
            latency_us = self.core_perf(op_instance, 2, prefer_iters, tensor_list)
            del tensor_list
            self.empty_cache()
        except Exception as e:
            pass

        result_json = op_instance.summary(latency_us)
        return result_json


