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
import time
import json
import math
import random
import logging
import traceback
from abc import ABC
from typing import Any, Dict, List

import torch

from backends import module_store
from backends.utils import dump_communication_ops_report, dump_computation_ops_report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")


default_op_registry = module_store.op_registry.copy()
default_op_compute_size_registry = module_store.op_compute_size_funcs.copy()
default_op_create_tensors_registry = module_store.op_create_tensors_funcs.copy()


class Backend(ABC):
    def __init__(
        self, 
        workload_dict, 
        device_index : int = 0,
        world_size : int = 1, 
        rank : int = 0
    ):
        self.workload = workload_dict
        self.op_name = workload_dict["operator"]
        self.iterations = workload_dict["iterations"]
        self.op = None
    
        self.device_index = device_index
        self.world_size = world_size
        self.rank = rank

        self.set_device(device_index)
        self.device_name = self.get_device_name(device_index)
        self.avail_memory, self.total_memory = self.get_mem_info(device_index)
        self.memory_limit = self.avail_memory * 0.9 // (1024**3)

        dist_module = self.get_dist_module()
        if self.world_size > 1:
            self.initialize_ccl(rank, world_size)
            avail_memory_list = [None for _ in range(world_size)]
            dist_module.all_gather_object(avail_memory_list, self.avail_memory)
            self.avail_memory = min(avail_memory_list)
            self.memory_limit = self.avail_memory * 0.9 // (1024**3)

        self.op_group = None
        self.op_group_size = 1

        self.get_op_instance()

    def __del__(self):
        self.destroy_process_group()


    def get_op_instance(self):
        if self.op_name in default_op_registry:
            self.op = default_op_registry[self.op_name]
        else:
            raise NotImplementedError

    def get_op_compute_size_func(self):
        if self.op_name in default_op_compute_size_registry:
            return default_op_compute_size_registry[self.op_name]
        else:
            raise NotImplementedError
        
    def get_op_create_tensors_func(self):
        if self.op_name in default_op_create_tensors_registry:
            return default_op_create_tensors_registry[self.op_name]
        else:
            raise NotImplementedError



    """
    device management related
    """
    def get_torch_device_name(self):
        raise NotImplementedError

    def get_device_name(self, index = 0):
        raise NotImplementedError

    def get_device_properties(self, index = 0):
        raise NotImplementedError

    def get_mem_info(self, index = 0):
        raise NotImplementedError

    def get_device_count(self):
        raise NotImplementedError
    
    def set_device(self, index = 0):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError

    def device_synchronize(self):
        raise NotImplementedError

    def empty_cache(self):
        raise NotImplementedError


    """
    ccl related
    """
    def get_dist_module(self):
        raise NotImplementedError

    def get_dist_backend(self):
        raise NotImplementedError

    def initialize_ccl(self, rank, world_size):
        raise NotImplementedError

    def new_group(self, ranks):
        raise NotImplementedError


    def destroy_process_group(self):
        dist = self.get_dist_module()
        if dist.is_initialized():
            dist.destroy_process_group()

    def op_group_barrier(self):
        dist = self.get_dist_module()
        if dist.is_initialized() and self.op_group_size > 1:
            dist.barrier(group=self.op_group)


    def _run_operation(self, operation, inputs):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized():
            result = operation(
                *inputs, 
                op_group=self.op_group, 
                op_group_size=self.op_group_size
            )
        else:
            result = operation(*inputs)
        return result


    def build_tensor(self, input_shapes, torch_dtype):
        # get funcs
        compute_size_func = self.get_op_compute_size_func()

        result = compute_size_func(
            input_shapes, torch_dtype,
            op_group=self.op_group,
            op_group_size=self.op_group_size
        )

        # 4: (bs, rw_bytes, r_bytes, w_bytes)  assume tensor_size = rw_bytes
        # 5: (bs, rw_bytes, r_bytes, w_bytes, tensor_size)
        if len(result) == 4:
            tensor_size = result[1]
        elif len(result) == 5:
            tensor_size = result[4]
            
        create_tensors_func = self.get_op_create_tensors_func()
        
        # avoid use cache, assume cache size is 1 GiB, and use 80% of available device memory
        assume_cache_size = 1 * 1024**3

        assume_avail_bytes = self.memory_limit * 0.9 * 1024**3



        if self.op_name in ["allreduce", "allgather", "reducescatter", "alltoall", "broadcast", "p2p", "device2host", "host2device", "hash_table"]:
            if tensor_size > assume_avail_bytes:
                return []
            else:
                max_data_cnt = 1
        else:
            if tensor_size > assume_avail_bytes:
                return [], 0, tensor_size
            elif 2 * tensor_size > assume_avail_bytes:
                max_data_cnt = 1
            elif tensor_size > assume_cache_size:
                max_data_cnt = 2
            else:
                max_data_cnt = min(math.floor(assume_avail_bytes / tensor_size), self.iterations)

        # create tensor_list for each op
        tensor_list = [
            create_tensors_func(
                input_shapes, torch_dtype, 
                self.get_torch_device_name(), 
                op_group=self.op_group, 
                op_group_size=self.op_group_size
            ) for _ in range(max_data_cnt)
        ]
        return tensor_list


    def warmup_and_test(self, tensor_list):
        warmup_iters = 3
        for _ in range(warmup_iters):
            self._run_operation(self.op, random.choice(tensor_list))

        test_iters = 5
        self.device_synchronize()
        self.op_group_barrier()
        start_time = time.perf_counter_ns()
        for _ in range(test_iters):
            self._run_operation(self.op, random.choice(tensor_list))
        self.device_synchronize()
        self.op_group_barrier()
        end_time = time.perf_counter_ns()
        avg_op_duration = (end_time - start_time) / 1e9 / test_iters

        total_op_duration_margin = 10.
        if avg_op_duration > total_op_duration_margin:
            prefer_iterations = 2
        else:
            prefer_iterations = min(math.ceil(total_op_duration_margin / avg_op_duration), self.iterations)
        return prefer_iterations







    def core_perf(self, prefer_iterations, tensor_list):
        self.device_synchronize()
        self.op_group_barrier()
        start_time = time.perf_counter_ns()
        for i in range(prefer_iterations):
            self._run_operation(self.op, tensor_list[i % len(tensor_list)])
        self.device_synchronize()
        self.op_group_barrier()
        end_time = time.perf_counter_ns()

        return (end_time - start_time) / 1e3 / prefer_iterations


    def perf(self, input_shapes: List[List[int]], dtype):
        error = ""
        torch_dtype = getattr(torch, dtype)

        # create input/output tensors for op
        # malloc multiple instances of tensors to avoid using last level cache
        tensor_list = self.build_tensor(
            input_shapes, torch_dtype
        )
        if len(tensor_list) == 0:
            raise RuntimeError("Not enough memory to run the op")

        prefer_iterations = self.warmup_and_test(tensor_list)
        latency = round(self.core_perf(prefer_iterations, tensor_list), 2)


        # clean tensors and device memory
        del tensor_list
        self.empty_cache()

        # create report for communication ops and computation ops
        if self.op_name in [
            "allreduce", "allgather", "reducescatter", "alltoall", "broadcast", "p2p", 
            "device2host", "host2device"
        ]:
            report = dump_communication_ops_report(
                self.op_name, torch_dtype, input_shapes, 
                self.get_op_compute_size_func(), 
                self.op_group_size, 
                None,
                latency,
                error
            )
        else:
            report = dump_computation_ops_report(
                self.op_name, torch_dtype, input_shapes, 
                self.get_op_compute_size_func(), 
                None, 
                latency, 
                error
            )
        return report

