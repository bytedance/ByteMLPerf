import torch

class AddMulOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, tensor_a, tensor_b, tensor_c):
        result = (tensor_a + tensor_b) * tensor_c
        return result


class GemmOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_a, input_b):
        logits = torch.matmul(input_a, input_b)
        return logits


class SoftmaxOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        logits = torch.nn.functional.softmax(hidden_states)
        return logits


