from typing import List

import torch
import cutlass

from backends.module_store import GemmOp, BatchGemmOp, GroupGemmOp


# gemm(pytorch) float32/float16/bfloat16 --> float32/float16/bfloat16
# gemm(cutlass) int8 --> int32
class GPUGemmOp(GemmOp):
    def __init__(self):
        super().__init__()

        try:
            import cutlass
            dtype = torch.int8
            accum_dtype=torch.int32
            self.plan = cutlass.op.Gemm(
                alpha=1, beta=0,
                element_A=dtype,
                element_B=dtype,
                element_C=accum_dtype,
                element_D=accum_dtype,
                layout_A=cutlass.LayoutType.RowMajor,
                layout_B=cutlass.LayoutType.RowMajor,
                layout_C=cutlass.LayoutType.RowMajor
            )
            self.op = self.plan.construct()
            self.gemm_op_int8 = cutlass.emit.pytorch(
                self.op, name='gemm', cc=self.plan.cc, 
                jit=True, sourcedir='out'
            )
        except:
            self.gemm_op_int8 = None
            raise Exception("GPUGemmOp cutlass error")

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor
    ):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype == torch.int8:
            output_tensor = self.gemm_op_int8.run(input_tensor_a, input_tensor_b)
        else:
            output_tensor = torch.mm(input_tensor_a, input_tensor_b)
        return output_tensor


# batch_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# batch_gemm(cutlass)   int8 --> int32
class GPUBatchGemmOp(BatchGemmOp):
    def __init__(self):
        super().__init__()

        try:
            import cutlass
        except:
            raise Exception("GPUBatchGemmOp import cutlass error")

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor
    ):
        compute_dtype = input_tensor_a.dtype

        output_tensor = None
        if compute_dtype == torch.int8:
            bs, m, n = input_tensor_a.shape[0], input_tensor_a.shape[1], input_tensor_b.shape[2]
            c_tensor = torch.randint(-3, 3, [bs, m, n], dtype=torch.int32, device="cuda")
            output_tensor = torch.randint(-3, 3, [bs, m, n], dtype=torch.int32, device="cuda")
            plan = cutlass.op.Gemm(A=input_tensor_a, B=input_tensor_b, C=c_tensor, D=output_tensor, element_accumulator=cutlass.DataType.s32)
            plan.run(input_tensor_a, input_tensor_b, c_tensor, output_tensor, 1, 0)
        else:
            output_tensor = torch.bmm(input_tensor_a, input_tensor_b)
        return output_tensor


# group_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# group_gemm(cutlass)   int8 --> int32
class GPUGroupGemmOp(GroupGemmOp):
    def __init__(self):
        super().__init__()

        try:
            import cutlass
            dtype = torch.int8
            accum_dtype=torch.int32
            self.plan = cutlass.op.GroupedGemm(
                alpha=1, beta=0, 
                element_A=dtype, 
                element_B=dtype, 
                element_C=accum_dtype, 
                element_D=accum_dtype, 
                layout_A=cutlass.LayoutType.RowMajor, 
                layout_B=cutlass.LayoutType.RowMajor, 
                layout_C=cutlass.LayoutType.RowMajor
            )
            self.op = self.plan.construct()
            self.gemm_op_int8 = cutlass.emit.pytorch(
                self.op, name='group_gemm', cc=self.plan.cc,
                jit=True, sourcedir='out'
            )
        except:
            self.gemm_op_int8 = None
            raise Exception("GPUGroupGemmOp cutlass error")

    def forward(self, 
        a_list : List[torch.Tensor], 
        b_list : List[torch.Tensor]
    ):
        compute_dtype = a_list[0].dtype
        if compute_dtype == torch.int8:
            output_tensors = self.gemm_op_int8.run(a_list, b_list)
        else:
            output_tensors = [a @ b for a, b in zip(a_list, b_list)]
        return output_tensors