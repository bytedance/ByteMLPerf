from typing import List

import torch
import cutlass

from backends.module_store import GemmOp, BatchGemmOp, GroupGemmOp




# gemm(pytorch) float32/float16/bfloat16 --> float32/float16/bfloat16
# gemm(cutlass) int8 --> int32
class GPUGemmOp(GemmOp):
    def __init__(self):
        super().__init__()

        # cutlass int8 gemm
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

        
        # cutlass int8 gemm
        dtype = torch.int8
        accum_dtype=torch.int32
        self.plan = cutlass.op.GemmBatched(
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
            self.op, name='batch_gemm', cc=self.plan.cc, 
            jit=True, sourcedir='out'
        )

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor
    ):
        compute_dtype = input_tensor_a.dtype

        output_tensor = None
        if compute_dtype == torch.int8:
            cc = torch.bmm(input_tensor_a.to(torch.float32), input_tensor_b.to(torch.float32))
            output_tensor = self.gemm_op_int8.run(input_tensor_a, input_tensor_b)
            print(torch.nonzero(cc - output_tensor))
            import pdb;pdb.set_trace()
        else:
            output_tensor = torch.bmm(input_tensor_a, input_tensor_b)
        return output_tensor




# group_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# group_gemm(cutlass)   int8 --> int32
class GPUGroupGemmOp(GroupGemmOp):
    def __init__(self):
        super().__init__()

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