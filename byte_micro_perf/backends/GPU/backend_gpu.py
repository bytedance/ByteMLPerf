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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")


class BackendGPU(Backend):
    def get_device_name(self):
        return torch.cuda.get_device_name(0)

    def get_backend_properties(self):
        self.memory_limit = int(
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
        )

        if os.path.exists(self.vendor_path) and (self.vendor_path).endswith(".json"):
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

    def gemm(self):
        self.op = GemmOp()

    def add(self):
        self.op = AddOp()

    def sin(self):
        self.op = SinOp()

    def cos(self):
        self.op = CosOp()

    def exp(self):
        self.op = ExpOp()

    def exponential(self):
        self.op = ExponentialOp()

    def gelu(self):
        self.op = GeluOp()

    def sort(self):
        self.op = SortOp()

    def unique(self):
        self.op = UniqueOp()

    def indexadd(self):
        self.op = IndexAddOp()

    def softmax(self):
        self.op = SoftmaxOp()

    def layernorm(self):
        self.op = LayerNormOp()

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

    def host2device(self):
        self.op = Host2DeviceOp(torch.device("cuda"))

    def device2host(self):
        self.op = Device2HostOp()

    def build_tensor(self, input_shapes, dtype):
        torch_type = getattr(torch, dtype)
        if torch_type == torch.int32:
            dtype_size = torch.iinfo(torch_type).bits // 8
        else:
            dtype_size = torch.finfo(torch_type).bits // 8
        size = sum([math.prod(shape) for shape in input_shapes])
        data_amount = size * 2 * dtype_size
        data_cnt = (self.memory_limit - 4) * 1024**3 // data_amount
        data_cnt = min(data_cnt, self.iterations)
        input_tensors_list = []
        for _ in range(data_cnt):
            input_tensors = [
                torch.randn(shape).type(torch_type).to(torch.device("cuda"))
                for shape in input_shapes
            ]
            input_tensors_list.append(input_tensors)

        if hasattr(self.op, "process_inputs"):
            input_tensors_list = [
                self.op.process_inputs(*(input_tensor))
                for input_tensor in input_tensors_list
            ]

        return input_tensors_list, max(data_cnt, 1)

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
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "49373"
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        torch.cuda.set_device(rank)
        # Call the init process
        timeout_seconds = int(os.environ.get("MEGATRON_NCCL_TIMEOUT_SECOND", 30))
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=None,
            timeout=timedelta(seconds=timeout_seconds),
        )
        self.setup_2d_group()
        log.warning("DIST: rank {}, world_size {}".format(rank, world_size))

    def setup_2d_group(self):
        self.rank = dist.get_rank()
        torch.cuda.set_device(self.rank)
        origin_store_based_barrier = dist_c10d._store_based_barrier
        dist_c10d._store_based_barrier = lambda *a, **kw: None
        self.world_size = dist.get_world_size()
        ranks = range(0, self.world_size)
        group = dist.new_group(ranks)
        if self.rank in ranks:
            self.group = group
        dist_c10d._store_based_barrier = origin_store_based_barrier
        # wait for all ranks finish group initializing
        torch.distributed.barrier()
