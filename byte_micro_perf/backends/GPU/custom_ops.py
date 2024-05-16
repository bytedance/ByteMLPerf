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
            layout_A=cutlass.LayoutType.ColumnMajorInterleaved32, 
            layout_B=cutlass.LayoutType.RowMajorInterleaved32, 
            layout_C=cutlass.LayoutType.RowMajor
        )
        self.op = self.plan.construct(
            alignment_A=16, 
            alignment_B=16, 
            alignment_C=8
        )
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
        if compute_dtype == torch.int8 and self.gemm_op_int8 is not None:
            output_tensor = self.gemm_op_int8.run(
                input_tensor_a, input_tensor_b
            )
        else:
            output_tensor = torch.mm(
                input_tensor_a, input_tensor_b
            )
        return output_tensor




# batch_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# batch_gemm(cutlass)   int8 --> int32
class GPUBatchGemmOp(BatchGemmOp):
    def __init__(self):
        super().__init__()

        # TODO: cutlass int8 batch_gemm
        pass

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor
    ):
        compute_dtype = input_tensor_a.dtype

        output_tensor = None
        if compute_dtype == torch.int8:
            # TODO
            pass
        else:
            output_tensor = torch.bmm(
                input_tensor_a, input_tensor_b
            )
        return output_tensor




# group_gemm(cutlass)   float32/float16/bfloat16 --> float32
# group_gemm(cutlass)   int8 --> int32
class GPUGroupGemmOp(GroupGemmOp):
    def __init__(self):
        super().__init__()

        self.group_gemm_fp32 = GPUGroupGemmOp.compile_mod(
            dtype=torch.float32, 
            accum_dtype=torch.float32, 
            mod_name="groupd_gemm_fp32"
        )
            
        self.group_gemm_fp16 = GPUGroupGemmOp.compile_mod(
            dtype=torch.float16, 
            accum_dtype=torch.float32, 
            mod_name="groupd_gemm_fp16"
        )

        self.group_gemm_bf16 = GPUGroupGemmOp.compile_mod(
            dtype=torch.bfloat16, 
            accum_dtype=torch.float32, 
            mod_name="groupd_gemm_bf16"
        )

        # TODO: cutlass int8 group_gemm
        self.group_gemm_int8 = None
        # if "int8" in dtype_list:
        #     self.group_gemm_int8 = GroupGemmOp.compile_mod(
        #         dtype=torch.int8, 
        #         accum_dtype=torch.int32, 
        #         mod_name="group_gemm_int8"
        #     )

    @staticmethod
    def compile_mod(dtype, accum_dtype, mod_name):

        if dtype == torch.int8:
            # TODO
            pass
            # plan = cutlass.op.Gemm(
            #     alpha=1, beta=0, 
            #     element_A=dtype, 
            #     element_B=dtype, 
            #     element_C=accum_dtype,
            #     element_D=accum_dtype, 
            #     layout_A=cutlass.LayoutType.ColumnMajorInterleaved32, 
            #     layout_B=cutlass.LayoutType.RowMajorInterleaved32, 
            #     layout_C=cutlass.LayoutType.RowMajor
            # )
            # op = plan.construct(
            #     alignment_A=16, 
            #     alignment_B=16, 
            #     alignment_C=8
            # )
            # grouped_gemm = cutlass.emit.pytorch(
            #     op, name=mod_name, 
            #     cc=plan.cc, jit=True, 
            #     sourcedir='out'
            # )
        else:
            plan = cutlass.op.GroupedGemm(
                alpha=1, beta=0, 
                element_A=dtype, 
                element_B=dtype, 
                element_C=accum_dtype, 
                element_D=accum_dtype, 
                layout_A=cutlass.LayoutType.RowMajor, 
                layout_B=cutlass.LayoutType.RowMajor, 
                layout_C=cutlass.LayoutType.RowMajor
            )
            op = plan.construct()
            grouped_gemm = cutlass.emit.pytorch(
                op, name=mod_name, 
                cc=plan.cc, jit=True, 
                sourcedir='./out'
            )
        return grouped_gemm


    def forward(self, 
        a_list : List[torch.Tensor], 
        b_list : List[torch.Tensor]
    ):
        compute_dtype = a_list[0].dtype
        if compute_dtype == torch.float32 and self.group_gemm_fp32 is not None:
            output_tensors = self.group_gemm_fp32.run(a_list, b_list)
        elif compute_dtype == torch.float16 and self.group_gemm_fp16 is not None:
            output_tensors = self.group_gemm_fp16.run(a_list, b_list)
        elif compute_dtype == torch.bfloat16 and self.group_gemm_bf16 is not None:
            output_tensors = self.group_gemm_bf16.run(a_list, b_list)
        elif compute_dtype == torch.int8 and self.group_gemm_int8 is not None:
            # TODO
            pass
            # output_tensors = self.group_gemm_int8.run(a_list, b_list)
        else:
            output_tensors = []
        return output_tensors