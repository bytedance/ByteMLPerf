from backends.backend import Backend
import torch

class AddMulOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, tensor_a, tensor_b, tensor_c):
        result = (tensor_a + tensor_b) * tensor_c
        return result

class LinearOp(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, tensor_a):
        tensor_c = self.linear(tensor_a)
        return tensor_c

class BackendCPU(Backend):
    def gemm(self, config):
        self.op = LinearOp()