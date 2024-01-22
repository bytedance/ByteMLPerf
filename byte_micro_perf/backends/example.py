import time
import torch
import time
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

def vector_perf_test(precision):
    model = AddMulOp().to('cuda')
    a = torch.rand(1, 2048, 12288, dtype=torch.float32).to(precision).to('cuda')
    b = torch.rand(1, 2048, 12288, dtype=torch.float32).to(precision).to('cuda')
    c = torch.rand(1, 2048, 12288, dtype=torch.float32).to(precision).to('cuda')
    output_gpu = model(a, b, c)
    torch.cuda.synchronize()
    s = time.time()
    for i in range(1000):
        output_gput = model(a, b, c)
    torch.cuda.synchronize()
    e = time.time()
    print(precision)
    print("1000 Round Elapsed(s): ", e - s)
    return e - s

if __name__ == "__main__":
    vector_perf_test(torch.bfloat16)
