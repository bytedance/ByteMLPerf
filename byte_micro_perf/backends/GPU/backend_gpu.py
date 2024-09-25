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

from backends import module_store
from backends.backend import Backend
from backends.utils import get_dtype_bytes


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")


class BackendGPU(Backend):
    def __init__(self, workload_dict: Dict[str, Any], vendor_path: str):
        super().__init__(workload_dict, vendor_path)

    def get_device_count(self):
        return torch.cuda.device_count()

    def get_device_name(self):
        return torch.cuda.get_device_name(0)

    def set_device(self, device_index : int):
        torch.cuda.set_device(device_index)
    
    def get_device(self):
        return torch.cuda.current_device()

    def device_synchronize(self):
        torch.cuda.synchronize()

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

    def log(self):
        self.op = LogOp()

    def sqrt(self):
        self.op = SqrtOp()

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
        dist.barrier()

    def destroy_process_group(self):
        dist.destroy_process_group()

    def barrier(self):
        dist.barrier(self.group)


    def all_gather_object(self, obj):
        gather_object_list = [None for _ in range(self.world_size)]
        dist.all_gather_object(
            object_list=gather_object_list,
            obj=obj,
            group=self.group
        )
        return gather_object_list


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


        # avoid use L2 Cache: assume max 1GB currently
        # data_per_cnt > 1GB, use two buffers
        # data_per_cnt < 1GB, malloc multiple buffer to exceed 1GB
        assume_l2_cache_size = 1 * 1024**3
        assume_avail_bytes = self.memory_limit * 0.9 * 1024**3

        if bytes_per_cnt > assume_avail_bytes:
            return [], 0, bytes_per_cnt
        elif 2 * bytes_per_cnt > assume_avail_bytes:
            max_data_cnt = 1
        elif bytes_per_cnt > assume_l2_cache_size:
            max_data_cnt = 2
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

        return input_tensors_list, max_data_cnt, bytes_per_cnt


    def _run_operation(self, operation, inputs):
        result = operation(*inputs)
        return result



    # device/host ops
    def host2device(self):
        self.op = module_store.Host2DeviceOp()

    def device2host(self):
        self.op = module_store.Device2HostOp()

    # communication ops
    def allreduce(self):
        self.op = module_store.AllReduceOp(self.group)

    def allgather(self):
        self.op = module_store.AllGatherOp(self.group)

    def reducescatter(self):
        self.op = module_store.ReduceScatterOp(self.group)

    def alltoall(self):
        self.op = module_store.AllToAllOp(self.group)

    def broadcast(self):
        self.op = module_store.BroadcastOp(self.group)

    def p2p(self):
        self.op = module_store.P2POp(self.group, self.ranks, self.rank)

    # compute ops
    # unary ops
    def sin(self):
        self.op = module_store.SinOp()

    def cos(self):
        self.op = module_store.CosOp()

    def exp(self):
        self.op = module_store.ExpOp()

    def exponential(self):
        self.op = module_store.ExponentialOp()

    def silu(self):
        self.op = module_store.SiluOp()

    def gelu(self):
        self.op = module_store.GeluOp()

    def swiglu(self):
        self.op = module_store.SwiGLUOp()

    def cast(self):
        self.op = module_store.CastOp()


    # binary ops
    def add(self):
        self.op = module_store.AddOp()

    def mul(self):
        self.op = module_store.MulOp()

    def sub(self):
        self.op = module_store.SubOp()

    def div(self):
        self.op = module_store.DivOp()


    # reduce ops
    def layernorm(self):
        self.op = module_store.LayerNormOp()

    def softmax(self):
        self.op = module_store.SoftmaxOp()

    def reduce_sum(self):
        self.op = module_store.ReduceSumOp()

    def reduce_min(self):
        self.op = module_store.ReduceMinOp()

    def reduce_max(self):
        self.op = module_store.ReduceMaxOp()


    # index ops
    def index_add(self):
        self.op = module_store.IndexAddOp()

    def sort(self):
        self.op = module_store.SortOp()

    def unique(self):
        self.op = module_store.UniqueOp()

    def scatter(self):
        self.op = module_store.ScatterOp()
    
    def gather(self):
        self.op = module_store.GatherOp()


    # gemm ops
    def gemm(self):
        self.op = module_store.GemmOp()

    def gemv(self):
        self.op = module_store.GemmOp()

    def batch_gemm(self):
        self.op = module_store.BatchGemmOp()

    def group_gemm(self):
        self.op = module_store.GroupGemmOp()
