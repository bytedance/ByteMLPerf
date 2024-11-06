# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random

import torch
import torch.distributed as dist

def gemm_compute_size(input_shapes, torch_dtype):
    # input_shapes: [[M, K], [K, N]]
    a_shape, b_shape = input_shapes
    M, _ = a_shape
    _, N = b_shape
    d_shape = [M, N]

    # get element_size and dtype_size
    input_element_num = sum([math.prod(shape) for shape in [a_shape, b_shape]])
    output_element_num = sum([math.prod(shape) for shape in [d_shape]])
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()

    input_tensor_size = dtype_size * input_element_num
    if torch_dtype == torch.int8:
        output_tensor_size = 4 * output_element_num
    else:
        output_tensor_size = dtype_size * output_element_num
    batch_size = M
    tensor_size = input_tensor_size + output_tensor_size
    return (batch_size, tensor_size, input_tensor_size, output_tensor_size)

def gemm_create_tensors(input_shapes, torch_dtype, xpu_device):
    # input_shapes: [[M, K], [K, N]]
    a_shape, b_shape = input_shapes
    M, _ = a_shape
    _, N = b_shape
    d_shape = [M, N]

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)
    b_tensor = torch.randint(0, 7, b_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    d_tensor = torch.randint(0, 7, d_shape, dtype=torch_dtype, device=xpu_device)
    return [a_tensor, b_tensor, d_tensor]






def batch_gemm_compute_size(input_shapes, torch_dtype):
    # input_shapes: [[bs, M, K], [bs, K, N]]
    a_shape, b_shape = input_shapes
    bs, M, _ = a_shape
    bs, _, N = b_shape
    d_shape = [bs, M, N]

    # get element_size and dtype_size
    input_element_num = sum([math.prod(shape) for shape in [a_shape, b_shape]])
    output_element_num = sum([math.prod(shape) for shape in [d_shape]])
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()

    input_tensor_size = dtype_size * input_element_num
    if torch_dtype == torch.int8:
        output_tensor_size = 4 * output_element_num
    else:
        output_tensor_size = dtype_size * output_element_num
    batch_size = bs
    tensor_size = input_tensor_size + output_tensor_size
    return (batch_size, tensor_size, input_tensor_size, output_tensor_size)

def batch_gemm_create_tensors(input_shapes, torch_dtype, xpu_device):
    # input_shapes: [[bs, M, K], [bs, K, N]]
    a_shape, b_shape = input_shapes
    bs, M, _ = a_shape
    bs, _, N = b_shape
    d_shape = [bs, M, N]

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)
    b_tensor = torch.randint(0, 7, b_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    d_tensor = torch.randint(0, 7, d_shape, dtype=torch_dtype, device=xpu_device)
    return [a_tensor, b_tensor, d_tensor]



def group_gemm_compute_size(input_shapes, torch_dtype):
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
        
        # get element_size and dtype_size
        input_element_num = sum([math.prod(shape) for shape in [a_shape, b_shape]])
        output_element_num = sum([math.prod(shape) for shape in [d_shape]])
        dtype_size = torch.tensor([], dtype=torch_dtype).element_size()

        input_tensor_size += dtype_size * input_element_num
        if torch_dtype == torch.int8:
            output_tensor_size += 4 * output_element_num
        else:
            output_tensor_size += dtype_size * output_element_num
        batch_size = 1
        tensor_size = input_tensor_size + output_tensor_size

    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def group_gemm_create_tensors(input_shapes, torch_dtype, xpu_device):
    """
    [
        [[M1, K1], [K1, N1]],
        [[M2, K2], [K2, N2]]
    ]
    """
    left_tensors = []
    right_tensors = []
    output_tensors = []

    for problem_shape in input_shapes:
        a_shape, b_shape = problem_shape
        M, _ = a_shape
        _, N = b_shape
        d_shape = [M, N]

        # create input tensors
        left_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)
        right_tensor = torch.randint(0, 7, b_shape, dtype=torch_dtype, device=xpu_device)

        # create output tensors
        output_tensor = torch.randint(0, 7, d_shape, dtype=torch_dtype, device=xpu_device)

        left_tensors.append(left_tensor)
        right_tensors.append(right_tensor)
        output_tensors.append(output_tensor)

    return [left_tensors, right_tensors, output_tensors]



def sin_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    c_shape = a_shape

    input_element_num = sum([math.prod(shape) for shape in [a_shape]])
    output_element_num = sum([math.prod(shape) for shape in [c_shape]])
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num
    batch_size = c_shape[0]
    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def sin_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    c_shape = a_shape

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    c_tensor = torch.randint(0, 7, c_shape, dtype=torch_dtype, device=xpu_device)
    return [a_tensor, c_tensor]


def cast_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    c_shape = a_shape
    input_element_num = sum([math.prod(shape) for shape in [a_shape]])
    output_element_num = sum([math.prod(shape) for shape in [c_shape]])

    if torch_dtype == torch.float32:
        dst_torch_dtype = torch.bfloat16
    elif torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        dst_torch_dtype = torch.float32
    elif torch_dtype == torch.int8:
        dst_torch_dtype = torch.int32
    else:
        dst_torch_dtype = torch_dtype

    src_dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    dst_dtype_size = torch.tensor([], dtype=dst_torch_dtype).element_size()

    input_tensor_size = src_dtype_size * input_element_num
    output_tensor_size = dst_dtype_size * output_element_num

    batch_size = c_shape[0]
    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def cast_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    c_shape = a_shape

    if torch_dtype == torch.float32:
        dst_torch_dtype = torch.bfloat16
    elif torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        dst_torch_dtype = torch.float32
    elif torch_dtype == torch.int8:
        dst_torch_dtype = torch.int32
    else:
        dst_torch_dtype = torch_dtype  

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    c_tensor = torch.randint(0, 7, c_shape, dtype=dst_torch_dtype, device=xpu_device)

    return [a_tensor, c_tensor]


def swiglu_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape

    input_tensor_shape = [batch_size, hidden_size]
    output_tensor_shape = [batch_size, hidden_size]

    input_element_num = sum([math.prod(shape) for shape in [input_tensor_shape]])
    output_element_num = sum([math.prod(shape) for shape in [output_tensor_shape]])

    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num
    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size


def swiglu_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape

    input_tensor_shape = [batch_size, hidden_size]
    output_tensor_shape = [batch_size, hidden_size]

    # create input tensors
    input_tensor = torch.randint(0, 7, input_tensor_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    output_tensor = torch.randint(0, 7, output_tensor_shape, dtype=torch_dtype, device=xpu_device)

    return [input_tensor, output_tensor]



def add_compute_size(input_shapes, torch_dtype):
    a_shape, b_shape = input_shapes
    c_shape = a_shape
    batch_size, hidden_size = a_shape

    input_element_num = sum([math.prod(shape) for shape in [a_shape, b_shape]])
    output_element_num = sum([math.prod(shape) for shape in [c_shape]])
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()

    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num
    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def add_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, b_shape = input_shapes
    c_shape = a_shape

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)
    b_tensor = torch.randint(0, 7, b_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    c_tensor = torch.randint(0, 7, c_shape, dtype=torch_dtype, device=xpu_device)
    return [a_tensor, b_tensor, c_tensor]


def layer_norm_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = a_shape
    w_shape = a_shape[-1:]

    input_element_num = sum([math.prod(shape) for shape in [a_shape, w_shape]])
    output_element_num = sum([math.prod(shape) for shape in [c_shape]])

    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num
    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def layer_norm_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = a_shape
    w_shape = a_shape[-1:]

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    c_tensor = torch.randint(0, 7, c_shape, dtype=torch_dtype, device=xpu_device)

    # create weight tensors
    w_tensor = torch.randint(0, 7, w_shape, dtype=torch_dtype, device=xpu_device)

    return [a_tensor, c_tensor, w_tensor]



def softmax_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = a_shape

    input_element_num = sum([math.prod(shape) for shape in [a_shape]])
    output_element_num = sum([math.prod(shape) for shape in [c_shape]])
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()

    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num
    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def softmax_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = a_shape

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    c_tensor = torch.randint(0, 7, c_shape, dtype=torch_dtype, device=xpu_device)
    return [a_tensor, c_tensor]



def reduce_sum_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = [batch_size, 1]

    input_element_num = sum([math.prod(shape) for shape in [a_shape]])
    output_element_num = sum([math.prod(shape) for shape in [c_shape]])
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num
    tensor_size = input_tensor_size + output_tensor_size

    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def reduce_sum_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = [batch_size, 1]

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    c_tensor = torch.randint(0, 7, c_shape, dtype=torch_dtype, device=xpu_device)

    return [a_tensor, c_tensor]


def reduce_min_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape

    values_shape = [batch_size, 1]
    indices_shape = [batch_size, 1]
    
    input_element_num = sum([math.prod(shape) for shape in [a_shape]])
    values_element_num = sum([math.prod(shape) for shape in [values_shape]])
    indices_element_num = sum([math.prod(shape) for shape in [indices_shape]])
    
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    indices_dtype_size = torch.tensor([], dtype=torch.int64).element_size()

    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * values_element_num + indices_dtype_size * indices_element_num
    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def reduce_min_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    values_shape = [batch_size, 1]
    indices_shape = [batch_size, 1]

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    values_tensor = torch.randint(0, 7, values_shape, dtype=torch_dtype, device=xpu_device)
    indices_tensor = torch.randint(0, 7, indices_shape, dtype=torch.int64, device=xpu_device)

    return [a_tensor, values_tensor, indices_tensor]




def index_add_compute_size(input_shapes, torch_dtype):
    # src_tensor -->(index_tensor) dst_tensor
    dst_shape, src_shape = input_shapes

    src_batch_size = src_shape[0]
    dst_batch_size = dst_shape[0]

    index_shape = [src_batch_size]

    
    src_element_num = sum([math.prod(shape) for shape in [src_shape]])
    index_element_num = sum([math.prod(shape) for shape in [index_shape]])

    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    index_dtype_size = torch.tensor([], dtype=torch.int64).element_size()

    src_tensor_size = dtype_size * src_element_num
    index_tensor_size = index_dtype_size * index_element_num


    input_tensor_size = 2 * src_tensor_size + index_tensor_size
    output_tensor_size = src_tensor_size
    tensor_size = input_tensor_size + output_tensor_size

    return src_batch_size, tensor_size, input_tensor_size, output_tensor_size


def index_add_create_tensors(input_shapes, torch_dtype, xpu_device):
    # src_tensor -->(index_tensor) dst_tensor
    dst_shape, src_shape = input_shapes

    src_batch_size = src_shape[0]
    dst_batch_size = dst_shape[0]

    index_shape = [src_batch_size]

    # create output tensors
    dst_tensor = torch.randint(0, 7, dst_shape, dtype=torch_dtype, device=xpu_device)

    # create input tensors
    src_tensor = torch.randint(0, 7, src_shape, dtype=torch_dtype, device=xpu_device)
    index_tensor = torch.randint(0, dst_batch_size, index_shape, dtype=torch.int64, device=xpu_device)
    
    return [dst_tensor, src_tensor, index_tensor]


def sort_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = a_shape

    input_element_num = sum([math.prod(shape) for shape in [a_shape]])
    output_element_num = sum([math.prod(shape) for shape in [c_shape]])
    indice_element_num  = output_element_num

    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    indice_dtype_size = torch.tensor([], dtype=torch.int64).element_size()

    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num + indice_dtype_size * indice_element_num
    tensor_size = input_tensor_size + output_tensor_size

    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def sort_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = a_shape

    # create input tensors
    a_tensor = torch.randint(0, 7, a_shape, dtype=torch_dtype, device=xpu_device)

    # create output tensors
    c_tensor = torch.randint(0, 7, c_shape, dtype=torch_dtype, device=xpu_device)
    indice_tensor = torch.randint(0, 7, c_shape, dtype=torch.int64, device=xpu_device)
    return [a_tensor, c_tensor, indice_tensor]


def unique_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = a_shape

    input_element_num = sum([math.prod(shape) for shape in [a_shape]])
    output_element_num = sum([math.prod(shape) for shape in [c_shape]])

    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    indice_dtype_size = torch.tensor([], dtype=torch.int64).element_size()

    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num + indice_dtype_size * output_element_num
    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def unique_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    c_shape = a_shape

    # create input tensors
    torch.manual_seed(1)
    a_tensor = torch.randint(0, 1024, a_shape, dtype=torch_dtype, device="cpu").to(device=xpu_device)

    # create output tensors
    c_tensor = torch.empty(c_shape, dtype=torch_dtype, device=xpu_device)
    count_tensor = torch.empty(c_shape, dtype=torch.int64, device=xpu_device)
    return [a_tensor, c_tensor, count_tensor]




def scatter_compute_size(input_shapes, torch_dtype):
    tensor_shape = input_shapes[0]
    batch_size, hidden_size = tensor_shape
    index_shape = [batch_size]

    input_element_num = sum([math.prod(shape) for shape in [tensor_shape]])
    output_element_num = sum([math.prod(shape) for shape in [tensor_shape]])
    index_element_num = sum([math.prod(shape) for shape in [index_shape]])

    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    index_dtype_size = torch.tensor([], dtype=torch.int64).element_size()

    input_element_num = dtype_size * input_element_num + index_dtype_size * index_element_num
    output_element_num = dtype_size * output_element_num
    tensor_size = input_element_num + output_element_num
    return batch_size, tensor_size, input_element_num, output_element_num

def scatter_create_tensors(input_shapes, torch_dtype, xpu_device):
    tensor_shape = input_shapes[0]
    batch_size, hidden_size = tensor_shape
    index_shape = [batch_size]

    # create output tensors
    dst_tensor = torch.randint(0, 7, tensor_shape, dtype=torch_dtype, device=xpu_device)

    # create input tensors
    src_tensor = torch.randint(0, 7, tensor_shape, dtype=torch_dtype, device=xpu_device)
    index = [i for i in range(batch_size)]
    random.shuffle(index)
    index_tensor = torch.tensor(index, dtype=torch.int64, device=xpu_device)
    index_tensor = index_tensor.reshape(-1, 1).expand(-1, hidden_size)

    return [dst_tensor, src_tensor, index_tensor]








def host2device_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape

    output_element_num = sum([math.prod(shape) for shape in [a_shape]])

    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    output_tensor_size = dtype_size * output_element_num

    tensor_size = output_tensor_size
    return batch_size, tensor_size, 0, output_tensor_size

def host2device_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape

    host_tensor = torch.empty(a_shape, dtype=torch_dtype, device="cpu").pin_memory()
    device_tensor = torch.empty(a_shape, dtype=torch_dtype, device=xpu_device)
    
    return [host_tensor, device_tensor]


def allreduce_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    a_tensor = torch.zeros(a_shape, dtype=torch_dtype, device=xpu_device)
    return [a_tensor]


def allgather_compute_size(input_shapes, torch_dtype):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape

    output_element_num = sum([math.prod(shape) for shape in [a_shape]])
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    output_tensor_size = dtype_size * output_element_num
    tensor_size = output_tensor_size
    return batch_size, tensor_size, 0, output_tensor_size



def allgather_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, = input_shapes
    batch_size, hidden_size = a_shape
    
    world_size = dist.get_world_size()
    tensor = torch.empty([batch_size, hidden_size], dtype=torch_dtype, device=xpu_device)
    tensors = list(torch.chunk(tensor, world_size, dim=0))

    return [tensors]


def alltoall_compute_size(input_shapes, torch_dtype):
    a_shape, b_shape = input_shapes
    batch_size, hidden_size = a_shape

    world_size = dist.get_world_size()
    output_element_num = sum([math.prod(shape) for shape in [a_shape]]) * 2
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    output_tensor_size = dtype_size * output_element_num
    tensor_size = output_tensor_size
    return batch_size, tensor_size, 0, output_tensor_size


def alltoall_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, b_shape = input_shapes
    batch_size, hidden_size = a_shape

    world_size = dist.get_world_size()
    input_tensor = torch.empty([batch_size, hidden_size], dtype=torch_dtype, device=xpu_device)
    input_tensors = list(torch.chunk(input_tensor, world_size, dim=0))

    output_tensor = torch.empty([batch_size, hidden_size], dtype=torch_dtype, device=xpu_device)
    output_tensors = list(torch.chunk(output_tensor, world_size, dim=0))

    return [input_tensors, output_tensors]
    

def p2p_compute_size(input_shapes, torch_dtype):
    a_shape, b_shape = input_shapes
    batch_size, hidden_size = a_shape

    input_element_num = sum([math.prod(shape) for shape in [a_shape]])
    output_element_num = sum([math.prod(shape) for shape in [b_shape]])

    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    input_tensor_size = dtype_size * input_element_num
    output_tensor_size = dtype_size * output_element_num

    tensor_size = input_tensor_size + output_tensor_size
    return batch_size, tensor_size, input_tensor_size, output_tensor_size

def p2p_create_tensors(input_shapes, torch_dtype, xpu_device):
    a_shape, b_shape = input_shapes
    batch_size, hidden_size = a_shape

    a_tensor = torch.empty(a_shape, dtype=torch_dtype, device=xpu_device)
    b_tensor = torch.empty(b_shape, dtype=torch_dtype, device=xpu_device)

    return [a_tensor, b_tensor]



"""
gemm ops
"""
class GemmOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_d):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.mm(input_tensor_a, input_tensor_b, out=input_tensor_d)
        else:
            raise Exception(f"GemmOp with dtype {compute_dtype} is not implemented")

class BatchGemmOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_d):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.bmm(input_tensor_a, input_tensor_b, out=input_tensor_d)
        else:
            raise Exception(f"BatchGemmOp with dtype {compute_dtype} is not implemented")

class GroupGemmOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_d):
        compute_dtype = input_tensor_a[0].dtype
        for a, b, d in zip(input_tensor_a, input_tensor_b, input_tensor_d):
            if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
                torch.mm(a, b, out=d)
            else:
                raise Exception(f"GroupGemmOp with dtype {compute_dtype} is not implemented")



"""
unary ops
"""
class SinOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        torch.sin(input_tensor, out=output_tensor)

class CosOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        torch.cos(input_tensor, out=output_tensor)

class ExpOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        torch.exp(input_tensor, out=output_tensor)

class ExponentialOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        input_tensor.exponential_()

class LogOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        torch.log(input_tensor, out=output_tensor)

class SqrtOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        torch.sqrt(input_tensor, out=output_tensor)

class CastOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        output_tensor = input_tensor.to(output_tensor.dtype)

class SiluOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        output_tensor = torch.nn.functional.silu(input_tensor)

class GeluOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        output_tensor = torch.nn.functional.gelu(input_tensor)

class SwiGLUOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        torch.mul(torch.nn.functional.silu(input_tensor), input_tensor, out=output_tensor)

"""
Binary ops
"""
class AddOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_c):
        torch.add(input_tensor_a, input_tensor_b, out=input_tensor_c)

class MulOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_c):
        torch.mul(input_tensor_a, input_tensor_b, out=input_tensor_c)

class SubOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_c):
        torch.sub(input_tensor_a, input_tensor_b, out=input_tensor_c)

class DivOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_c):
        torch.div(input_tensor_a, input_tensor_b, out=input_tensor_c)



"""
reduction ops
"""
class LayerNormOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor, weight_tensor):
        output_tensor = torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight_tensor)

class SoftmaxOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        output_tensor = torch.nn.functional.softmax(input_tensor, dim=-1, dtype=output_tensor.dtype)

class ReduceSumOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor):
        torch.sum(input_tensor, dim=-1, keepdim=True, dtype=output_tensor.dtype, out=output_tensor)

class ReduceMinOp(torch.nn.Module):
    def forward(self, input_tensor, value_tensor, indice_tensor):
        torch.min(input_tensor, dim=-1, keepdim=True, out=(value_tensor, indice_tensor))

class ReduceMaxOp(torch.nn.Module):
    def forward(self, input_tensor, value_tensor, indice_tensor):
        torch.max(input_tensor, dim=-1, keepdim=True, out=(value_tensor, indice_tensor))



"""
index_ops
"""
class IndexAddOp(torch.nn.Module):
    def forward(self, dst_tensor, src_tensor, index_tensor):
        dst_tensor.index_add_(0, index_tensor, src_tensor)

class SortOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor, indice_tensor):
        torch.sort(input_tensor, dim=-1, out=(output_tensor, indice_tensor))

class UniqueOp(torch.nn.Module):
    def forward(self, input_tensor, output_tensor, count_tensor):
        output_tensor, count_tensor = torch.unique(
            input=input_tensor, 
            sorted=False, 
            return_counts=True, 
            return_inverse=False
        )

class ScatterOp(torch.nn.Module):
    def forward(self, dst_tensor, src_tensor, index_tensor):
        dst_tensor.scatter_(0, index_tensor, src_tensor)

class GatherOp(torch.nn.Module):
    def forward(self, dst_tensor, src_tensor, index_tensor):
        torch.gather(src_tensor, 0, index_tensor, out=dst_tensor)



"""
h2d_ops
"""
class Host2DeviceOp(torch.nn.Module):
    def forward(self, host_tensor, device_tensor):
        device_tensor.copy_(host_tensor)


class Device2HostOp(torch.nn.Module):
    def forward(self, host_tensor, device_tensor):
        host_tensor.copy_(device_tensor)



"""
communication ops
"""
class AllReduceOp(torch.nn.Module):
    def forward(self, input_tensor):
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)

class AllGatherOp(torch.nn.Module):
    def forward(self, input_tensors):
        dist.all_gather(input_tensors, input_tensors[dist.get_rank()])

class ReduceScatterOp(torch.nn.Module):
    def forward(self, input_tensors):
        dist.reduce_scatter(input_tensors[dist.get_rank()], input_tensors)

class AllToAllOp(torch.nn.Module):
    def forward(self, input_tensors, output_tensors):
        dist.all_to_all(output_tensors, input_tensors)


class BroadcastOp(torch.nn.Module):
    def forward(self, input_tensor):
        dist.broadcast(input_tensor, 0)


class P2POp(torch.nn.Module):
    def forward(self, send_tensor, recv_tensor):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        reqs = []
        if rank != world_size - 1:
            reqs.append(dist.isend(send_tensor, (rank + 1) % world_size))
        if rank != 0:
            reqs.append(dist.irecv(recv_tensor, (rank - 1 + world_size) % world_size))
        for req in reqs:
            req.wait()


op_registry = {
    # gemm ops
    "gemm": GemmOp(), 
    "gemv": GemmOp(),
    "batch_gemm": BatchGemmOp(),
    "group_gemm": GroupGemmOp(),

    # unary ops
    "sin": SinOp(),
    "cos": CosOp(),
    "exp": ExpOp(),
    "exponential": ExponentialOp(),
    "log": LogOp(),
    "sqrt": SqrtOp(),
    "cast": CastOp(),
    "silu": SiluOp(),
    "gelu": GeluOp(),
    "swiglu": SwiGLUOp(),

    # binary ops
    "add": AddOp(),
    "sub": SubOp(),
    "mul": MulOp(),
    "div": DivOp(),

    # reduction ops
    "layernorm": LayerNormOp(),
    "softmax": SoftmaxOp(),
    "reduce_sum": ReduceSumOp(),
    "reduce_max": ReduceMaxOp(),
    "reduce_min": ReduceMinOp(),

    # index_ops
    "index_add": IndexAddOp(),
    "sort": SortOp(),
    "unique": UniqueOp(),
    "scatter": ScatterOp(),
    "gather": GatherOp(),

    # h2d_ops
    "device2host": Device2HostOp(),
    "host2device": Host2DeviceOp(),

    # ccl ops
    "broadcast": BroadcastOp(),
    "allreduce": AllReduceOp(),
    "allgather": AllGatherOp(),
    "alltoall": AllToAllOp(),
    "reducescatter": ReduceScatterOp(),
    "p2p": P2POp(),
}


op_compute_size_funcs = {
    # gemm_ops
    "gemm": gemm_compute_size,
    "gemv": gemm_compute_size,
    "batch_gemm": batch_gemm_compute_size,
    "group_gemm": group_gemm_compute_size,

    # unary_ops
    "sin": sin_compute_size,
    "cos": sin_compute_size,
    "exp": sin_compute_size,
    "exponential": sin_compute_size,
    "log": sin_compute_size,
    "sqrt": sin_compute_size,
    "cast": cast_compute_size,
    "silu": sin_compute_size,
    "gelu": sin_compute_size,
    "swiglu": swiglu_compute_size,

    # binary_ops
    "add": add_compute_size,
    "mul": add_compute_size,
    "sub": add_compute_size,
    "div": add_compute_size,

    # reduction_ops
    "layernorm": layer_norm_compute_size,
    "softmax": softmax_compute_size,
    "reduce_sum": reduce_sum_compute_size,
    "reduce_min": reduce_min_compute_size,
    "reduce_max": reduce_min_compute_size,

    # index_ops
    "index_add": index_add_compute_size,
    "sort": sort_compute_size,
    "unique": unique_compute_size,
    "scatter": scatter_compute_size,
    "gather": scatter_compute_size,

    # h2d_ops
    "host2device": host2device_compute_size,
    "device2host": host2device_compute_size,

    # ccl_ops
    "broadcast": host2device_compute_size,
    "allreduce": host2device_compute_size,
    "allgather": allgather_compute_size,
    "alltoall": alltoall_compute_size,
    "reducescatter": allgather_compute_size, 
    "p2p": p2p_compute_size,
}

op_create_tensors_funcs = {
    # gemm ops
    "gemm": gemm_create_tensors,
    "gemv": gemm_create_tensors,
    "batch_gemm": batch_gemm_create_tensors,
    "group_gemm": group_gemm_create_tensors,

    # unary ops
    "sin": sin_create_tensors,
    "cos": sin_create_tensors,
    "exp": sin_create_tensors,
    "exponential": sin_create_tensors,
    "log": sin_create_tensors,
    "sqrt": sin_create_tensors,
    "cast": cast_create_tensors,
    "silu": sin_create_tensors,
    "gelu": sin_create_tensors,
    "swiglu": swiglu_create_tensors,

    # binary ops
    "add": add_create_tensors,
    "mul": add_create_tensors,
    "sub": add_create_tensors,
    "div": add_create_tensors,

    # reduction ops
    "layernorm": layer_norm_create_tensors,
    "softmax": softmax_create_tensors,
    "reduce_sum": reduce_sum_create_tensors,
    "reduce_min": reduce_min_create_tensors,
    "reduce_max": reduce_min_create_tensors,

    # index ops
    "index_add": index_add_create_tensors,
    "sort": sort_create_tensors,
    "unique": unique_create_tensors,
    "scatter": scatter_create_tensors,
    "gather": scatter_create_tensors,

    # h2d_ops
    "host2device": host2device_create_tensors,
    "device2host": host2device_create_tensors,

    # ccl_ops
    "broadcast": allreduce_create_tensors,
    "allreduce": allreduce_create_tensors,
    "allgather": allgather_create_tensors,
    "alltoall": alltoall_create_tensors,
    "reducescatter": allgather_create_tensors, 
    "p2p": p2p_create_tensors,
}
