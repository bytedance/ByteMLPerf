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
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from backends.utils import dump_communication_ops_report, dump_computation_ops_report

class Backend(ABC):
    def __init__(self, workload_dict: Dict[str, Any], vendor_path: str):
        self.op_name = workload_dict["operator"]
        self.iterations = workload_dict["iterations"]
        self.warmup = int(0.1 * workload_dict["iterations"])
        self.vendor_path = vendor_path
        self.op = None

        # communication params
        self.rank = None
        self.world_size = None
        self.group = None

        # hardware info
        self.hw_info_dict = None
        self.memory_limit = None
        self.bandwidth_limit = None
        self.get_backend_properties()

    @abstractmethod
    def get_device_name(self):
        pass

    @abstractmethod
    def get_backend_properties(self):
        pass

    @abstractmethod
    def build_tensor(self, input_shapes: List[List[int]], dtype):
        pass

    @abstractmethod
    def _run_operation(self, operation, inputs):
        pass

    @abstractmethod
    def device_synchronize(self):
        pass

    @abstractmethod
    def initialize_ccl(self, rank, world_size):
        pass

    @abstractmethod
    def setup_2d_group(self):
        pass



    # gemm ops
    def gemm(self):
        pass

    def batch_gemm(self):
        pass

    def group_gemm(self):
        pass


    # device/host ops
    def host2device(self):
        pass

    def device2host(self):
        pass


    # communication ops
    def allreduce(self):
        pass

    def allgather(self):
        pass

    def reducescatter(self):
        pass

    def alltoall(self):
        pass

    def broadcast(self):
        pass


    # other compute ops
    def add(self):
        pass

    def sin(self):
        pass

    def cos(self):
        pass

    def exp(self):
        pass

    def exponential(self):
        pass

    def gelu(self):
        pass

    def indexadd(self):
        pass

    def sort(self):
        pass

    def unique(self):
        pass

    def softmax(self):
        pass

    def layernorm(self):
        pass









    # perf specify input_shape for 
    def perf(self, input_shapes: List[List[int]], dtype):
        error = ""

        # create input tensors based on input_shapes and dtype
        tensor_list, tensor_cnt, tensor_size_perc_cnt = self.build_tensor(
            input_shapes, dtype
        )

        if tensor_cnt > 0:
            # random select input tensors
            input_index_list = [
                random.randint(0, tensor_cnt - 1) for _ in range(self.iterations)
            ]

            # warmup
            num_warm_up = 10
            for _ in range(num_warm_up):
                self._run_operation(self.op, tensor_list[0])

            # perf
            self.device_synchronize()
            start_time = time.perf_counter_ns()
            for i in range(self.iterations):
                result = self._run_operation(
                    self.op, 
                    tensor_list[input_index_list[i]]
                )
            self.device_synchronize()
            end_time = time.perf_counter_ns()

            # time in us
            total_exec_time = (end_time - start_time) / 1e3
            latency = round(total_exec_time / self.iterations, 2)
        else:
            latency = 0
            error = "OOM"

        
        if self.op_name in ["allreduce", "allgather", "reducescatter", "alltoall", "broadcast"]:
            report = dump_communication_ops_report(
                self.op_name,
                dtype,
                input_shapes,
                self.group.size(),
                self.bandwidth_limit,
                latency,
                error
            )
        else:
            report = dump_computation_ops_report(
                self.op_name, 
                dtype, 
                input_shapes, 
                self.bandwidth_limit, 
                latency, 
                error
            )
        return report

