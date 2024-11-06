# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import logging

from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

from backends import module_store
from backends.backend import Backend


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")



class BackendGPU(Backend):
    def __init__(self, workload_dict, vendor_path):
        super().__init__(workload_dict, vendor_path)

        


    def get_device_name(self):
        return torch.cuda.get_device_name(0)

    def get_torch_device_name(self):
        return "cuda"

    def get_device_properties(self):
        return torch.cuda.get_device_properties(0)

    def get_device_count(self):
        return torch.cuda.device_count()

    def set_device(self, device_index : int):
        torch.cuda.set_device(device_index)
    
    def get_device(self):
        return torch.cuda.current_device()
    
    def device_synchronize(self):
        torch.cuda.synchronize()

    def empty_cache(self):
        torch.cuda.empty_cache()



    def get_dist_module(self):
        return dist

    def initialize_ccl(self, rank, world_size):
        # check device_count
        device_count = self.get_device_count()
        if world_size > device_count:
            world_size = device_count
        if rank >= world_size:
            return False
        self.set_device(rank)

        # set envs and internal vars
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "49373"
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # init process group
        self.get_dist_module().init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank, 
            timeout=timedelta(seconds=1800)
        )
        return True

