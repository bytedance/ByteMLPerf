from typing import List

import torch

#import cutlass
from .ixgemmblaslt import gemm88

from backends.module_store import GemmOp, BatchGemmOp, GroupGemmOp


# gemm(pytorch) float32/float16/bfloat16 --> float32/float16/bfloat16
# gemm(cutlass) int8 --> int32
class ILUVATARGemmOp(GemmOp):
    def __init__(self):
        super().__init__()

        try:
            self.blasLtIns = gemm88.gemm_init()
        except:
            self.blasLtIns = None
            raise Exception("ILUVATARGemmOp ixgemmblaslt error")
        
    def __del__(self):
        if not self.blasLtIns is None:
            gemm88.gemm_release(self.blasLtIns)

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor
    ):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype == torch.int8:
            output_tensor = gemm88.gemm_run(self.blasLtIns, [input_tensor_a], [input_tensor_b])[0]
        else:
            output_tensor = torch.mm(input_tensor_a, input_tensor_b)
        return output_tensor


# batch_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# batch_gemm(cutlass)   int8 --> int32
class ILUVATARBatchGemmOp(BatchGemmOp):
    def __init__(self):
        super().__init__()

        try:
            self.blasLtIns = gemm88.gemm_init()
        except:
            self.blasLtIns = None
            raise Exception("ILUVATARBatchGemmOp import ixgemmblaslt error")
        
    def __del__(self):
        if not self.blasLtIns is None:
            gemm88.gemm_release(self.blasLtIns)

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor
    ):
        compute_dtype = input_tensor_a.dtype

        output_tensor = None
        if compute_dtype == torch.int8:
            output_tensor = gemm88.gemm_run(self.blasLtIns, [input_tensor_a], [input_tensor_b])[0]
        else:
            output_tensor = torch.bmm(input_tensor_a, input_tensor_b)
        return output_tensor


# group_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# group_gemm(cutlass)   int8 --> int32
class ILUVATARGroupGemmOp(GroupGemmOp):
    def __init__(self):
        super().__init__()

        try:
            self.blasLtIns = gemm88.gemm_init()
        except:
            self.blasLtIns = None
            raise Exception("ILUVATARGroupGemmOp cutlass error")
        
    def __del__(self):
        if not self.blasLtIns is None:
            gemm88.gemm_release(self.blasLtIns)

    def forward(self, 
        a_list : List[torch.Tensor], 
        b_list : List[torch.Tensor]
    ):
        compute_dtype = a_list[0].dtype
        if compute_dtype == torch.int8:
            output_tensors = gemm88.gemm_run(self.blasLtIns, a_list, b_list)
        else:
            output_tensors = [a @ b for a, b in zip(a_list, b_list)]
        return output_tensors