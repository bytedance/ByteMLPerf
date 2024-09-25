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
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

from backends.utils import dump_communication_ops_report, dump_computation_ops_report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")


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

        self.target_dtype = None

    """
    device management related
    """
    @abstractmethod
    def get_device_count(self):
        raise NotImplementedError

    @abstractmethod
    def get_device_name(self):
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device_index : int):
        raise NotImplementedError

    @abstractmethod
    def get_device(self):
        raise NotImplementedError

    @abstractmethod
    def device_synchronize(self):
        raise NotImplementedError

    @abstractmethod
    def get_backend_properties(self):
        raise NotImplementedError


    @abstractmethod
    def initialize_ccl(self, rank, world_size):
        torch.cuda.set_device(rank)

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "49373"
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank, 
            timeout=timedelta(seconds=1800)
        )

        self.setup_2d_group()
        return True


    @abstractmethod
    def setup_2d_group(self):
        # get dist info
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.ranks = range(0, self.world_size)
        
        # set device
        torch.cuda.set_device(self.rank)

        # get original store_based_barrier
        origin_store_based_barrier = dist_c10d._store_based_barrier
        dist_c10d._store_based_barrier = lambda *a, **kw: None
        group = dist.new_group(self.ranks)
        if self.rank in self.ranks:
            self.group = group
        dist_c10d._store_based_barrier = origin_store_based_barrier

        # wait for all ranks finish group initializing
        torch.barrier()

    @abstractmethod
    def destroy_process_group(self):
        dist.destroy_process_group()

    @abstractmethod
    def barrier(self):
        dist.barrier(self.group)

    @abstractmethod
    def all_gather_object(self, obj):
        if dist.is_initialized() and self.world_size is not None and self.group is not None:
            gather_object_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(
                object_list=gather_object_list,
                obj=obj,
                group=self.group
            )
            return gather_object_list


    @abstractmethod
    def build_tensor(self, input_shapes: List[List[int]], dtype):
        raise NotImplementedError

    @abstractmethod
    def _run_operation(self, operation, inputs):
        return operation(*inputs)



    # perf specify input_shape for 
    def perf(self, input_shapes: List[List[int]], dtype):
        error = ""

        # create input tensors based on input_shapes and dtype
        tensor_list, tensor_cnt, tensor_size_perc_cnt = self.build_tensor(
            input_shapes, dtype)

        if tensor_cnt > 0:
            try:
                # warmup
                num_warm_up = 10
                for _ in range(num_warm_up):
                    self._run_operation(self.op, tensor_list[0])

                # test perf
                num_test_perf = 10
                self.device_synchronize()
                start_time = time.perf_counter_ns()
                for i in range(num_test_perf):
                    self._run_operation(self.op, tensor_list[0])
                self.device_synchronize()
                end_time = time.perf_counter_ns()

                prefer_iterations = self.iterations
                max_perf_seconds = 10.0
                op_duration = (end_time - start_time) / num_test_perf / 1e9
                if op_duration > max_perf_seconds:
                    prefer_iterations = 5
                else:
                    prefer_iterations = min(max(int(max_perf_seconds // op_duration), 10), self.iterations)

                # ccl ops need barrier
                if self.op_name in ["allreduce", "allgather", "reducescatter", "alltoall", "broadcast", "p2p"]:
                    self.barrier()

                # perf
                self.device_synchronize()
                start_time = time.perf_counter_ns()
                for i in range(prefer_iterations):
                    self._run_operation(self.op, tensor_list[i % tensor_cnt])
                self.device_synchronize()
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

        tensor_list = []
        
        if self.op_name in ["allreduce", "allgather", "reducescatter", "alltoall", "broadcast", "p2p"]:
            report = dump_communication_ops_report(
                self.op_name,
                dtype,
                input_shapes,
                self.group.size(),
                None,
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



    """
    gemm ops
    """
    def gemm(self):
        pass

    def gemv(self):
        pass

    def batch_gemm(self):
        pass

    def group_gemm(self):
        pass


    """
    communication ops
    """
    def host2device(self):
        pass

    def device2host(self):
        pass

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

    def p2p(self):
        pass

    # compute ops
    # unary ops
    def sin(self):
        pass

    def cos(self):
        pass

    def exp(self):
        pass

    def exponential(self):
        pass

    def silu(self):
        pass

    def gelu(self):
        pass

    def swiglu(self):
        pass

    def cast(self):
        pass

    def log(self):
        pass

    def sqrt(self):
        pass

    # binary ops
    def add(self):
        pass

    def mul(self):
        pass

    def sub(self):
        pass

    def div(self):
        pass


    # reduce ops
    def layernorm(self):
        pass

    def softmax(self):
        pass

    def reduce_sum(self):
        pass

    def reduce_min(self):
        pass

    def reduce_max(self):
        pass


    # index ops
    def index_add(self):
        pass

    def sort(self):
        pass

    def unique(self):
        pass

    def scatter(self):
        pass
        
    def gather(self):
        pass

