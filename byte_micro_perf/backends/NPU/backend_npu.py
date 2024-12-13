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
import random

from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch_npu
import torch.distributed as dist

from backends import module_store
from backends.backend import Backend

from .custom_ops import npu_gemm_compute_size, npu_gemm_create_tensors, NPUGemmOp
from .custom_ops import npu_batch_gemm_compute_size, npu_batch_gemm_create_tensors, NPUBatchGemmOp
from .custom_ops import npu_group_gemm_compute_size, npu_group_gemm_create_tensors, NPUGroupGemmOp


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")


npu_op_registry = module_store.op_registry.copy()
npu_op_registry.update({"gemm": NPUGemmOp()})
npu_op_registry.update({"gemv": NPUGemmOp()})
npu_op_registry.update({"batch_gemm": NPUBatchGemmOp()})
npu_op_registry.update({"group_gemm": NPUGroupGemmOp()})

npu_op_compute_size_funcs = module_store.op_compute_size_funcs.copy()
npu_op_compute_size_funcs.update({"gemm": npu_gemm_compute_size})
npu_op_compute_size_funcs.update({"gemv": npu_gemm_compute_size})
npu_op_compute_size_funcs.update({"batch_gemm": npu_batch_gemm_compute_size})
npu_op_compute_size_funcs.update({"group_gemm": npu_group_gemm_compute_size})

npu_op_create_tensors_funcs = module_store.op_create_tensors_funcs.copy()
npu_op_create_tensors_funcs.update({"gemm": npu_gemm_create_tensors})
npu_op_create_tensors_funcs.update({"gemv": npu_gemm_create_tensors})
npu_op_create_tensors_funcs.update({"batch_gemm": npu_batch_gemm_create_tensors})
npu_op_create_tensors_funcs.update({"group_gemm": npu_group_gemm_create_tensors})



class BackendNPU(Backend):
    def __init__(self, workload_dict):
        super().__init__(workload_dict)

    """
    op
    """
    def get_op_instance(self):
        if self.op_name in npu_op_registry:
            self.op = npu_op_registry[self.op_name]
        else:
            raise NotImplementedError

    def get_op_compute_size_func(self):
        if self.op_name in npu_op_compute_size_funcs:
            return npu_op_compute_size_funcs[self.op_name]
        else:
            raise NotImplementedError
        
    def get_op_create_tensors_func(self):
        if self.op_name in npu_op_create_tensors_funcs:
            return npu_op_create_tensors_funcs[self.op_name]
        else:
            raise NotImplementedError


    def get_device_name(self):
        return torch_npu.npu.get_device_name(0)

    def get_torch_device_name(self):
        return "npu"

    def get_device_properties(self, index = 0):
        return torch_npu.npu.get_device_properties(0)

    def get_mem_info(self):
        return torch_npu.npu.mem_get_info()

    def get_device_count(self):
        return torch_npu.npu.device_count()

    def set_device(self, device_index : int):
        torch_npu.npu.set_device(device_index)
    
    def get_device(self):
        return torch_npu.npu.current_device()
    
    def device_synchronize(self):
        torch_npu.npu.synchronize()

    def empty_cache(self):
        torch_npu.npu.empty_cache()


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
            backend="hccl",
            world_size=world_size,
            rank=rank, 
            timeout=timedelta(seconds=1800)
        )
        return True
    

    def core_perf(self, prefer_iterations, tensor_list):
        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)

        self.device_synchronize()
        self.barrier()

        start_event.record()
        for i in range(prefer_iterations):
            self._run_operation(self.op, random.choice(tensor_list))
        end_event.record()

        self.device_synchronize()
        self.barrier()
        
        return start_event.elapsed_time(end_event) * 1e3 / prefer_iterations