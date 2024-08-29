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

import json
import logging
import math
import os
from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

from backends.backend import Backend
from backends.module_store import *
from backends.utils import get_dtype_bytes

from .custom_ops import GPUGemmOp, GPUBatchGemmOp, GPUGroupGemmOp


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")


class BackendGPU(Backend):
    def get_device_count(self):
        return torch.cuda.device_count()

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)
    
    def get_device(self):
        return torch.cuda.current_device()

    def all_gather_object(self, obj):
        gather_object_list = [None for _ in range(self.world_size)]
        dist.all_gather_object(
            object_list=gather_object_list,
            obj=obj,
            group=self.group
        )
        return gather_object_list


    def get_device_name(self):
        return torch.cuda.get_device_name(0)

    def get_backend_properties(self):
        self.memory_limit = int(
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
        )

        if self.vendor_path is not None and os.path.exists(self.vendor_path) and (self.vendor_path).endswith(".json"):
            with open(self.vendor_path, "r") as f:
                self.hw_info_dict = json.load(f)
                # if the vendor path does not exist, please set this param manaually
                self.bandwidth_limit = self.hw_info_dict["内存参数"]["内存"]["内存带宽(GB/s)"]
        else:
            log.warning(
                "Vendor_path: [ {} ] was not found or not a full path points to json, please check your path!!! Otherwise, please set the hardware info manaually.".format(
                    self.vendor_path
                )
            )


    # device/host ops
    def host2device(self):
        self.op = Host2DeviceOp()

    def device2host(self):
        self.op = Device2HostOp()

    # communication ops
    def allreduce(self):
        self.setup_2d_group()
        self.op = AllReduceOp(self.group)

    def allgather(self):
        self.setup_2d_group()
        self.op = AllGatherOp(self.group)

    def reducescatter(self):
        self.setup_2d_group()
        self.op = ReduceScatterOp(self.group)

    def alltoall(self):
        self.setup_2d_group()
        self.op = AllToAllOp(self.group)

    def broadcast(self):
        self.setup_2d_group()
        self.op = BroadcastOp(self.group)

    def p2p(self):
        self.setup_2d_group()
        self.op = P2POp(self.group, self.ranks, self.rank)

    # compute ops
    # unary ops
    def sin(self):
        self.op = SinOp()

    def cos(self):
        self.op = CosOp()

    def exp(self):
        self.op = ExpOp()

    def exponential(self):
        self.op = ExponentialOp()

    def silu(self):
        self.op = SiluOp()

    def gelu(self):
        self.op = GeluOp()

    def swiglu(self):
        self.op = SwiGLUOp()

    def cast(self):
        self.op = CastOp()


    # binary ops
    def add(self):
        self.op = AddOp()

    def mul(self):
        self.op = MulOp()

    def sub(self):
        self.op = SubOp()

    def div(self):
        self.op = DivOp()


    # reduce ops
    def layernorm(self):
        self.op = LayerNormOp()

    def softmax(self):
        self.op = SoftmaxOp()

    def reduce_sum(self):
        self.op = ReduceSumOp()

    def reduce_min(self):
        self.op = ReduceMinOp()

    def reduce_max(self):
        self.op = ReduceMaxOp()


    # index ops
    def index_add(self):
        self.op = IndexAddOp()

    def sort(self):
        self.op = SortOp()

    def unique(self):
        self.op = UniqueOp()

    def scatter(self):
        self.op = ScatterOp()
    
    def gather(self):
        self.op = GatherOp()

    # gemm ops
    def gemm(self):
        self.op = GPUGemmOp()

    def gemv(self):
        self.op = GPUGemmOp()

    def batch_gemm(self):
        self.op = GPUBatchGemmOp()

    def group_gemm(self):
        self.op = GPUGroupGemmOp()



    # create input tensors
    def build_tensor(self, input_shapes, dtype):
        torch.cuda.empty_cache()
        torch_dtype = getattr(torch, dtype)

        # compute size of input and output tensors
        if hasattr(self.op, "compute_size"):
            bytes_per_cnt = self.op.compute_size(input_shapes, dtype)
        # default: input_tensors_size == output_tensor_size, all tensors have same dtype
        else:
            dtype_size = get_dtype_bytes(dtype)
            element_num = 2 * sum([math.prod(shape) for shape in input_shapes])
            bytes_per_cnt = dtype_size * element_num


        # avoid use L2 Cache: assume 256 MB currently
        # data_per_cnt > 256 MB, only one buffer
        # data_per_cnt < 256 MB, malloc multiple buffer to exceed 256MB, and use first and last buffer

        assume_l2_cache_size = 256 * 1024**2
        if bytes_per_cnt > self.memory_limit * 0.9 * 1024 ** 3:
            return [], 0, bytes_per_cnt
        elif bytes_per_cnt > assume_l2_cache_size:
            max_data_cnt = 1
        else:
            max_data_cnt = math.ceil(assume_l2_cache_size / bytes_per_cnt)

        # create input tensors for each op
        input_tensors_list = []
        for _ in range(max_data_cnt):
            # create input tensors
            if hasattr(self.op, "custom_create_tensors"):
                input_tensors = self.op.custom_create_tensors(input_shapes, torch_dtype, "cuda")
                input_tensors_list.append(input_tensors)
            # default: all input tensors have same dtype
            else:
                if torch_dtype in [torch.int8, torch.int32]:
                    input_tensors = [
                        torch.randint(-3, 3, size=shape, dtype=torch_dtype, device="cuda")
                        for shape in input_shapes
                    ]
                else:
                    input_tensors = [
                        torch.randn(shape, dtype=torch_dtype, device="cuda")
                        for shape in input_shapes
                    ]
                input_tensors_list.append(input_tensors)
        if hasattr(self.op, "process_inputs"):
            input_tensors_list = [
                self.op.process_inputs(*(input_tensor))
                for input_tensor in input_tensors_list
            ]


        if max_data_cnt > 2:
            max_data_cnt = 2
            new_tensor_list = []
            new_tensor_list.append(input_tensors_list[0])
            new_tensor_list.append(input_tensors_list[-1])
        else:
            new_tensor_list = input_tensors_list

        return new_tensor_list, max_data_cnt, bytes_per_cnt



    def _run_operation(self, operation, inputs):
        result = operation(*inputs)
        return result

    def device_synchronize(self):
        torch.cuda.synchronize()
        return True


    def initialize_ccl(self, rank, world_size):
        """
        initialize distributed process groups and relevant ENVs
        """

        # set cuda device
        torch.cuda.set_device(rank)

        # set envs
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "49373"
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Call the init process
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank
        )

        # create group
        self.setup_2d_group()

        log.info(f"DIST: rank {rank}, world_size {world_size}")
        return True


    def setup_2d_group(self):
        # get rank and set device
        self.rank = dist.get_rank()
        torch.cuda.set_device(self.rank)

        origin_store_based_barrier = dist_c10d._store_based_barrier
        dist_c10d._store_based_barrier = lambda *a, **kw: None

        self.world_size = dist.get_world_size()
        self.ranks = range(0, self.world_size)
        group = dist.new_group(self.ranks)
        if self.rank in self.ranks:
            self.group = group

        dist_c10d._store_based_barrier = origin_store_based_barrier

        # wait for all ranks finish group initializing
        torch.distributed.barrier()

    def destroy_process_group(self):
        dist.destroy_process_group()

    def barier(self):
        dist.barrier(self.group)