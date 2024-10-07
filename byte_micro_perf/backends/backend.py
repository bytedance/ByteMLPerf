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
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, List, final

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

from backends import module_store
from backends.utils import dump_communication_ops_report, dump_computation_ops_report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")


default_op_registry = module_store.op_registry.copy()
default_op_compute_size_registry = module_store.op_compute_size_funcs.copy()
default_op_create_tensors_registry = module_store.op_create_tensors_funcs.copy()


class Backend(ABC):
    def __init__(self, workload_dict: Dict[str, Any], vendor_path: str):
        self.op_name = workload_dict["operator"]
        self.iterations = workload_dict["iterations"]
        self.op = None
    
        # communication params
        self.world_size = None
        self.rank = None

        # hardware info
        self.device_name = self.get_device_name()

        self.memory_limit = int(
            self.get_device_properties().total_memory / (1024**3)
        )
        if vendor_path is not None and os.path.exists(vendor_path) and (vendor_path).endswith(".json"):
            with open(vendor_path, "r") as f:
                self.hw_info_dict = json.load(f)
                # if the vendor path does not exist, please set this param manaually
                self.bandwidth_limit = self.hw_info_dict["内存参数"]["内存"]["内存带宽(GB/s)"]

    """
    op
    """
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
    # torch.get_device_name()
    def get_device_name(self):
        raise NotImplementedError
    
    # "cuda"
    def get_torch_device_name(self):
        raise NotImplementedError

    def get_device_properties(self):
        raise NotImplementedError

    def get_device_count(self):
        raise NotImplementedError
    
    def set_device(self, device_index : int):
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

    def initialize_ccl(self, rank, world_size):
        raise NotImplementedError
    
    def destroy_process_group(self):
        dist = self.get_dist_module()
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self):
        dist = self.get_dist_module()
        if dist.is_initialized():
            dist.barrier()



    def _run_operation(self, operation, inputs):
        result = operation(*inputs)
        return result


    def build_tensor(self, input_shapes, torch_dtype):
        # get funcs
        compute_size_func = self.get_op_compute_size_func()
        create_tensors_func = self.get_op_create_tensors_func()
        _, tensor_size, _, _ = compute_size_func(input_shapes, torch_dtype)

        # avoid use cache, assume cache size is 1 GiB, and use 80% of device memory
        assume_cache_size = 1 * 1024**3
        assume_avail_bytes = self.memory_limit * 0.9 * 1024**3

        if self.op_name in ["allreduce", "allgather", "reducescatter", "alltoall", "broadcast", "p2p", "device2host", "host2device"]:
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
            create_tensors_func(input_shapes, torch_dtype, self.get_torch_device_name()) for _ in range(max_data_cnt)
        ]
        return tensor_list


    def perf(self, input_shapes: List[List[int]], dtype):
        error = ""

        # create necessary tensors
        torch_dtype = getattr(torch, dtype)
        tensor_list = self.build_tensor(input_shapes, torch_dtype)


        if len(tensor_list) > 0:
            try:
                warm_iterations = 5
                test_iterations = 5
                max_total_duration = 10.
                prefer_iterations = self.iterations

                # warmup
                self.device_synchronize()
                self.barrier()
                for _ in range(warm_iterations):
                    self._run_operation(self.op, random.choice(tensor_list))

                # test perf
                self.device_synchronize()
                self.barrier()
                start_time = time.perf_counter_ns()
                for i in range(test_iterations):
                    self._run_operation(self.op, random.choice(tensor_list))
                self.device_synchronize()
                self.barrier()
                end_time = time.perf_counter_ns()
                avg_op_duration = (end_time - start_time) / 1e9 / test_iterations


                if avg_op_duration > max_total_duration:
                    prefer_iterations = 2
                else:
                    prefer_iterations = min(math.ceil(max_total_duration / avg_op_duration), self.iterations)

                # perf
                self.device_synchronize()
                self.barrier()
                start_time = time.perf_counter_ns()
                for i in range(prefer_iterations):
                    self._run_operation(self.op, tensor_list[i % len(tensor_list)])
                self.device_synchronize()
                self.barrier()
                end_time = time.perf_counter_ns()

                # time in us
                total_exec_time = (end_time - start_time) / 1e3
                latency = round(total_exec_time / prefer_iterations, 2)

            except Exception as e:
                traceback.print_exc()
                latency = 0
                error = "RUN_OP_ERROR"
        else:
            latency = 0
            error = "OOM"
        
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
                self.world_size, 
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

