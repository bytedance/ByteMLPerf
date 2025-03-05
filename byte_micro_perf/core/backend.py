import os
import sys
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


    def op_group_barrier(self, op_group, group_size):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized() and group_size > 1:
            dist_module.barrier(group=op_group)

    
    def destroy_process_group(self):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized():
            dist_module.destroy_process_group()
    



    
    def perf(self, op_instance, *args, **kwargs):
        return


        if op_instance.is_custom_run():
            latency_us = op_instance.core_run(*args, **kwargs)
        else:
            # op
            op_size_info = op_instance.get_size_info()

            # device
            device_mem_info = self.get_mem_info()
            avail_memory = device_mem_info[0]
            total_memory = device_mem_info[1]
            memory_limit = int(avail_memory * 0.9)


            if type(op_instance).is_concurrent():
                pass
            else:
                pass



            # print(avail_memory, total_memory, memory_limit)


            # tensor_size = op_size_info.tensor_size


    
    


