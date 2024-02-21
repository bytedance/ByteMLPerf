from backends.backend import Backend
from backends.module_store import *
import torch
import torch.distributed as dist

import os
from datetime import timedelta

class BackendGPU(Backend):
    def gemm(self):
        self.op = GemmOp()

    def softmax(self):
        self.op = SoftmaxOp()

    def allreduce(self):
        self.setup_2d_group()
        self.op = AllReduceOp(self.group)        

    def build_tensor(self, input_shapes, dtype):
        tensors = [torch.randn(shape).type(getattr(torch, self.dtype)).cuda() for shape in input_shapes]
        return tensors

    def _run_operation(self, operation, inputs):
        return operation(*inputs)

    def initialize_ccl(self):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(self.rank)
        # Call the init process
        timeout_seconds = int(os.environ.get("MEGATRON_NCCL_TIMEOUT_SECOND", 30))
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=None,
            timeout=timedelta(seconds=timeout_seconds))
        print(f'DIST INFO: rank {self.rank}, world_size {self.world_size}', flush=True)

    def setup_2d_group(self):
        self.initialize_ccl()
        ranks = range(0, self.world_size)
        group = dist.new_group(ranks)
        if self.rank in ranks:
            self.group = group
        
