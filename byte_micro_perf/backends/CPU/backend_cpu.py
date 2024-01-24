from backends.backend import Backend
from backends.module_store import *
import torch

class BackendCPU(Backend):
    def gemm(self):
        self.op = GemmOp()

    def build_tensor(self, input_shapes, dtype):
        tensors = [torch.randn(shape) for shape in input_shapes]
        return tensors

    def _run_operation(self, operation, inputs):
        return operation(*inputs)