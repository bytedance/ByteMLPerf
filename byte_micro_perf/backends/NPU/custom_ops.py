import torch
import torch_npu
import math


def npu_gemm_compute_size(input_shapes, torch_dtype):
    # input_shapes: [[M, K], [K, N]]
    a_shape, b_shape = input_shapes
    M, K = a_shape
    K, N = b_shape
    d_shape = [M, N]
    
    a_tensor_size = math.prod(a_shape) * torch.tensor([], dtype=torch_dtype).element_size()
    b_tensor_size = math.prod(b_shape) * torch.tensor([], dtype=torch_dtype).element_size()

    input_tensor_size = a_tensor_size + b_tensor_size
    output_tensor_size = 0
    if torch_dtype == torch.int8:
        d_dtype = torch.int8
        d_tensor_size = math.prod(d_shape) * torch.tensor([], dtype=d_dtype).element_size()
        output_tensor_size += d_tensor_size

        scale_shape = [N]
        scale_dtype = torch.float32
        scale_tensor_size = math.prod(scale_shape) * torch.tensor([], dtype=scale_dtype).element_size()
        input_tensor_size += scale_tensor_size
    elif torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        d_dtype = torch_dtype
        d_tensor_size = math.prod(d_shape) * torch.tensor([], dtype=d_dtype).element_size()
        output_tensor_size += d_tensor_size
    else:
        raise NotImplementedError(f"dtype {torch_dtype} is not supported")

    tensor_size = input_tensor_size + output_tensor_size
    return M, tensor_size, input_tensor_size, output_tensor_size


def npu_gemm_create_tensors(input_shapes, torch_dtype, xpu_device):
    # input_shapes: [[M, K], [K, N]]
    a_shape, b_shape = input_shapes
    M, K = a_shape
    K, N = b_shape
    d_shape = [M, N]

    a_tensor = torch.randint(0, 7, size=a_shape, dtype=torch_dtype, device=xpu_device)
    b_tensor = torch.randint(0, 7, size=b_shape, dtype=torch_dtype, device=xpu_device)

    tensor_list = [a_tensor, b_tensor]
    if torch_dtype == torch.int8:
        d_dtype = torch.int8
        d_tensor = torch.empty(d_shape, dtype=torch_dtype, device=xpu_device)

        scale_shape = [N]
        scale_dtype = torch.float32
        scale_tensor = torch.rand(scale_shape, dtype=scale_dtype, device=xpu_device)

        tensor_list.append(d_tensor)
        tensor_list.append(scale_tensor)
    elif torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        d_dtype = torch_dtype
        d_tensor = torch.empty(d_shape, dtype=d_dtype, device=xpu_device)

        tensor_list.append(d_tensor)
    else:
        raise NotImplementedError(f"dtype {torch_dtype} is not supported")

    return tensor_list


class NPUGemmOp(torch.nn.Module): 
    def forward(
        self, 
        input_tensor_a : torch.Tensor, 
        input_tensor_b : torch.Tensor,
        input_tensor_d : torch.Tensor, 
        scale_tensor = None
    ):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype == torch.int8:
            input_tensor_d = torch_npu.npu_quant_matmul(
                x1=input_tensor_a, 
                x2=input_tensor_b, 
                scale=scale_tensor, 
                output_dtype=torch.int8
            )
        elif compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.mm(input_tensor_a, input_tensor_b, out=input_tensor_d)





def npu_batch_gemm_compute_size(input_shapes, torch_dtype):
    # input_shapes: [[B, M, K], [B, K, N]]
    a_shape, b_shape = input_shapes
    B, M, K = a_shape
    B, K, N = b_shape
    d_shape = [B, M, N]

    a_tensor_size = math.prod(a_shape) * torch.tensor([], dtype=torch_dtype).element_size()
    b_tensor_size = math.prod(b_shape) * torch.tensor([], dtype=torch_dtype).element_size()

    input_tensor_size = a_tensor_size + b_tensor_size
    output_tensor_size = 0

    if torch_dtype == torch.int8:
        d_dtype = torch.int8
        d_tensor_size = math.prod(d_shape) * torch.tensor([], dtype=d_dtype).element_size()
        output_tensor_size += d_tensor_size

        scale_shape = [N]
        scale_dtype = torch.float32
        scale_tensor_size = math.prod(scale_shape) * torch.tensor([], dtype=scale_dtype).element_size()
        input_tensor_size += scale_tensor_size
    elif torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        d_dtype = torch_dtype
        d_tensor_size = math.prod(d_shape) * torch.tensor([], dtype=d_dtype).element_size()
        output_tensor_size += d_tensor_size
    else:
        raise NotImplementedError(f"dtype {torch_dtype} is not supported")

    tensor_size = input_tensor_size + output_tensor_size
    return B, tensor_size, input_tensor_size, output_tensor_size


def npu_batch_gemm_create_tensors(input_shapes, torch_dtype, xpu_device):
    # input_shapes: [[B, M, K], [B, K, N]]
    a_shape, b_shape = input_shapes
    B, M, K = a_shape
    B, K, N = b_shape
    d_shape = [B, M, N]

    a_tensor = torch.randint(0, 7, size=a_shape, dtype=torch_dtype, device=xpu_device)
    b_tensor = torch.randint(0, 7, size=b_shape, dtype=torch_dtype, device=xpu_device)

    tensor_list = [a_tensor, b_tensor]
    if torch_dtype == torch.int8:
        d_dtype = torch.int8
        d_tensor = torch.empty(d_shape, dtype=torch_dtype, device=xpu_device)

        scale_shape = [N]
        scale_dtype = torch.float32
        scale_tensor = torch.rand(scale_shape, dtype=scale_dtype, device=xpu_device)

        tensor_list.append(d_tensor)
        tensor_list.append(scale_tensor)
    elif torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        d_dtype = torch_dtype
        d_tensor = torch.empty(d_shape, dtype=d_dtype, device=xpu_device)

        tensor_list.append(d_tensor)
    else:
        raise NotImplementedError(f"dtype {torch_dtype} is not supported")

    return tensor_list  


class NPUBatchGemmOp(torch.nn.Module):
    def forward(
        self, 
        input_tensor_a: torch.Tensor, 
        input_tensor_b: torch.Tensor, 
        input_tensor_d: torch.Tensor, 
        scale_tensor = None
    ):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype == torch.int8:
            input_tensor_d = torch_npu.npu_quant_matmul(
                x1=input_tensor_a, 
                x2=input_tensor_b, 
                scale=scale_tensor, 
                output_dtype=torch.int8
            )
        elif compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.bmm(input_tensor_a, input_tensor_b, out=input_tensor_d)




def npu_group_gemm_compute_size(input_shapes, torch_dtype):
    """
    [
        [[M1, K1], [K1, N1]], 
        [[M2, K2], [K2, N2]]
    ]
    """
    input_tensor_size = 0
    output_tensor_size = 0

    for problem_shape in input_shapes:
        a_shape, b_shape = problem_shape
        M, _ = a_shape
        _, N = b_shape
        d_shape = [M, N]

        a_tensor_size = math.prod(a_shape) * torch.tensor([], dtype=torch_dtype).element_size()
        b_tensor_size = math.prod(b_shape) * torch.tensor([], dtype=torch_dtype).element_size()
        
        input_tensor_size += a_tensor_size + b_tensor_size
        output_tensor_size += 0

        if torch_dtype == torch.int8:
            d_dtype = torch.int8
            d_tensor_size = math.prod(d_shape) * torch.tensor([], dtype=d_dtype).element_size()
            output_tensor_size += d_tensor_size

            scale_shape = [N]
            scale_dtype = torch.int64
            scale_tensor_size = math.prod(scale_shape) * torch.tensor([], dtype=scale_dtype).element_size()
            input_tensor_size += scale_tensor_size
        elif torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            d_dtype = torch_dtype
            d_tensor_size = math.prod(d_shape) * torch.tensor([], dtype=d_dtype).element_size()
            output_tensor_size += d_tensor_size
        else:
            raise NotImplementedError(f"dtype {torch_dtype} is not supported")

    batch_size = 1
    tensor_size = input_tensor_size + output_tensor_size
    
    return batch_size, tensor_size, input_tensor_size, output_tensor_size


def npu_group_gemm_create_tensors(input_shapes, torch_dtype, xpu_device):
    left_tensors = []
    right_tensors = []
    output_tensors = []
    scale_tensors = []

    for problem_shape in input_shapes:
        a_shape, b_shape = problem_shape
        M, _ = a_shape
        _, N = b_shape
        d_shape = [M, N]

        # create input tensors
        left_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)
        right_tensor = torch.randint(0, 7, b_shape, dtype=torch_dtype, device=xpu_device)

        left_tensors.append(left_tensor)
        right_tensors.append(right_tensor)

        if torch_dtype == torch.int8:
            d_dtype = torch.int8
            output_tensor = torch.empty(d_shape, dtype=d_dtype, device=xpu_device)
            output_tensors.append(output_tensor)

            scale_shape = [N]
            scale_dtype = torch.int64
            scale_tensor = torch.rand(scale_shape, dtype=scale_dtype, device=xpu_device)
            scale_tensors.append(scale_tensor)
        else:
            d_dtype = torch_dtype
            output_tensor = torch.empty(d_shape, dtype=d_dtype, device=xpu_device)
            output_tensors.append(output_tensor)

    return [left_tensors, right_tensors, output_tensors, scale_tensors]


class NPUGroupGemmOp(torch.nn.Module):
    def forward(self, 
        a_tensors, 
        b_tensors, 
        c_tensors, 
        scale_tensors
    ):
        compute_dtype = a_tensors[0].dtype
        if compute_dtype == torch.int8:
            c_tensors = torch_npu.npu_grouped_matmul(
                x=a_tensors, 
                weight=b_tensors, 
                scale=scale_tensors, 
                output_dtype=torch.int8
            )
        elif compute_dtype == torch.float32:
            for i in range(len(a_tensors)):
                c_tensors[i] = torch.mm(a_tensors[i], b_tensors[i])
        elif compute_dtype in [torch.float16, torch.bfloat16]:
            c_tensors = torch_npu.npu_grouped_matmul(
                x=a_tensors, 
                weight=b_tensors
            )