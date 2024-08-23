from typing import List

import torch

from backends.module_store import GemmOp, BatchGemmOp, GroupGemmOp


# gemm(pytorch) float32/float16/bfloat16 --> float32/float16/bfloat16
# gemm(cutlass) int8 --> int32
class GPUGemmOp(GemmOp):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor
    ):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype == torch.int8:
            output_tensor = input_tensor_a
        else:
            output_tensor = torch.mm(input_tensor_a, input_tensor_b)
        return output_tensor


# batch_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# batch_gemm(cutlass)   int8 --> int32
class GPUBatchGemmOp(BatchGemmOp):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor
    ):
        compute_dtype = input_tensor_a.dtype

        output_tensor = None
        if compute_dtype == torch.int8:
            output_tensor = input_tensor_a
        else:
            output_tensor = torch.bmm(input_tensor_a, input_tensor_b)
        return output_tensor


# group_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# group_gemm(cutlass)   int8 --> int32
class GPUGroupGemmOp(GroupGemmOp):
    def __init__(self):
        super().__init__()

    def forward(self, 
        a_list : List[torch.Tensor], 
        b_list : List[torch.Tensor]
    ):
        compute_dtype = a_list[0].dtype
        if compute_dtype == torch.int8:
            output_tensors = a_list
        else:
            output_tensors = [a @ b for a, b in zip(a_list, b_list)]
        return output_tensors