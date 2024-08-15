from typing import List

import torch
import torch_npu
import math

from backends.module_store import GemmOp, BatchGemmOp, GroupGemmOp
from backends.utils import get_dtype_bytes


# gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# gemm(torch_npu) int8 --> int8
class NPUGemmOp(GemmOp):
    def __init__(self):
        super().__init__()
        scale = torch.randn(16, dtype=torch.float32)

    def compute_size(self, input_shapes, dtype):
        # input_shapes: [[M, K], [K, N]]
        torch_dtype = getattr(torch, dtype)
        a_shape, b_shape = input_shapes
        M, K = a_shape
        K, N = b_shape
        d_shape = [M, N]
        dtype_size = get_dtype_bytes(dtype)
        input_element_num = sum([math.prod(shape) for shape in [a_shape, b_shape]])
        output_element_num = sum([math.prod(shape) for shape in [d_shape]])
        if torch_dtype == torch.int8:
            bytes_per_cnt = dtype_size * input_element_num + get_dtype_bytes("float32") * (output_element_num + N)
        else:
            bytes_per_cnt = dtype_size * (input_element_num + output_element_num)
        return bytes_per_cnt

    def custom_create_tensors(self, input_shapes, torch_dtype, xpu_device):
        """
        [
            [[M1, K1], [K1, N1]], 
            [[M2, K2], [K2, N2]]
        ]
        """
        if torch_dtype in [torch.int8, torch.int32]:
            input_tensors = [
                torch.randint(-3, 3, size=shape, dtype=torch_dtype, device=xpu_device)
                for shape in input_shapes
            ]
        else:
            input_tensors = [
                torch.randn(shape, dtype=torch_dtype, device=xpu_device)
                for shape in input_shapes
            ]
        if torch_dtype in [torch.int8]:
            N = input_shapes[1][1]
            input_tensors.append(torch.randn([N], dtype=torch.float32, device=xpu_device))
        else:
            input_tensors.append(None)

        return input_tensors

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor,
        scale = None
    ):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype == torch.int8:
            output_tensor = torch_npu.npu_quant_matmul(input_tensor_a, input_tensor_b, scale)
        else:
            output_tensor = torch.mm(input_tensor_a, input_tensor_b)
        return output_tensor


# batch_gemm(pytorch)     float32/float16/bfloat16 --> float32/float16/bfloat16
# batch_gemm(torch_npu)   int8 --> int8
class NPUBatchGemmOp(BatchGemmOp):
    def __init__(self):
        super().__init__()

    def compute_size(self, input_shapes, dtype):
        # input_shapes: [[bs, M, K], [bs, K, N]]
        torch_dtype = getattr(torch, dtype)
        a_shape, b_shape = input_shapes
        bs, M, K = a_shape
        bs, K, N = b_shape
        d_shape = [bs, M, N]
        dtype_size = get_dtype_bytes(dtype)
        input_element_num = sum([math.prod(shape) for shape in [a_shape, b_shape]])
        output_element_num = sum([math.prod(shape) for shape in [d_shape]])
        if torch_dtype == torch.int8:
            bytes_per_cnt = dtype_size * input_element_num + get_dtype_bytes("int32") * (output_element_num + N)
        else:
            bytes_per_cnt = dtype_size * (input_element_num + output_element_num)
        return bytes_per_cnt

    def custom_create_tensors(self, input_shapes, torch_dtype, xpu_device):
        """
        [
            [[M1, K1], [K1, N1]], 
            [[M2, K2], [K2, N2]]
        ]
        """
        if torch_dtype in [torch.int8, torch.int32]:
            input_tensors = [
                torch.randint(-3, 3, size=shape, dtype=torch_dtype, device=xpu_device)
                for shape in input_shapes
            ]
        else:
            input_tensors = [
                torch.randn(shape, dtype=torch_dtype, device=xpu_device)
                for shape in input_shapes
            ]
        if torch_dtype in [torch.int8]:
            N = input_shapes[1][2]
            input_tensors.append(torch.randn([N], dtype=torch.float32, device=xpu_device))
        else:
            input_tensors.append(None)

        return input_tensors
    

    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor,
        scale = None
    ):
        compute_dtype = input_tensor_a.dtype

        output_tensor = None
        if compute_dtype == torch.int8:
            output_tensor = torch_npu.npu_quant_matmul(input_tensor_a, input_tensor_b, scale)
        else:
            output_tensor = torch.bmm(input_tensor_a, input_tensor_b)
        return output_tensor


# group_gemm(pytorch)   float32--> float32
# batch_gemm(torch_npu) float16/bfloat16/int8 --> float16/bfloat16/int8
class NPUGroupGemmOp(GroupGemmOp):
    def __init__(self):
        super().__init__()

    def compute_size(self, input_shapes, dtype):
        # input_shapes: [[[M1, K1], [K1, N1]], [[M2, K2], [K2, N2]]]
        torch_dtype = getattr(torch, dtype)
        bytes_per_cnt = 0
        for problem_shape in input_shapes:
            a_shape, b_shape = problem_shape
            M, K = a_shape
            K, N = b_shape
            d_shape = [M, N]
            dtype_size = get_dtype_bytes(dtype)
            input_element_num = sum([math.prod(shape) for shape in [a_shape, b_shape]])
            output_element_num = sum([math.prod(shape) for shape in [d_shape]])
            if torch_dtype == torch.int8:
                bytes_per_cnt += dtype_size * (input_element_num + output_element_num) + get_dtype_bytes("int64") * N
            else:
                bytes_per_cnt += dtype_size * (input_element_num + output_element_num)
        return bytes_per_cnt

    def custom_create_tensors(self, input_shapes, torch_dtype, xpu_device):
        """
        [
            [[M1, K1], [K1, N1]],
            [[M2, K2], [K2, N2]]
        ]
        """
        left_tensors = []
        right_tensors = []
        scale_tensors = []

        for problem_shape in input_shapes:
            a_shape, b_shape = problem_shape
            if torch_dtype in [torch.int8, torch.int32]:
                left_tensor = torch.randint(-3, 3, size=a_shape, dtype=torch_dtype, device=xpu_device)
                right_tensor = torch.randint(-3, 3, size=b_shape, dtype=torch_dtype, device=xpu_device)
            else:
                left_tensor = torch.randn(a_shape, dtype=torch_dtype, device=xpu_device)
                right_tensor = torch.randn(b_shape, dtype=torch_dtype, device=xpu_device)
            if torch_dtype in [torch.int8]:
                N = b_shape[1]
                scale_tensors.append(torch.randn([N], dtype=torch.int64, device=xpu_device))
            else:
                scale_tensors.append(None)
            left_tensors.append(left_tensor)
            right_tensors.append(right_tensor)

        return [left_tensors, right_tensors, scale_tensors]

    def forward(self, 
        a_list : List[torch.Tensor], 
        b_list : List[torch.Tensor],
        c_list : List[torch.Tensor]
    ):
        compute_dtype = a_list[0].dtype
        if compute_dtype == torch.int8:
            output_tensors = torch_npu.npu_grouped_matmul(a_list, b_list, scale=c_list)
        elif compute_dtype == torch.float32:
            output_tensors = [a @ b for a, b in zip(a_list, b_list)]
        else:
            output_tensors = torch_npu.npu_grouped_matmul(a_list, b_list)
        return output_tensors