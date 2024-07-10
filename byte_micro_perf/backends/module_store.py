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
from typing import List

import torch
import torch.distributed as dist

from .utils import get_dtype_bytes


class GemmOp(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
            bytes_per_cnt = dtype_size * input_element_num + get_dtype_bytes("float32") * output_element_num
        else:
            bytes_per_cnt = dtype_size * (input_element_num + output_element_num)
        return bytes_per_cnt

    def forward(self, input_tensor_a, input_tensor_b):
        compute_dtype = input_tensor_a.dtype
        output_tensor = None
        if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            output_tensor = torch.mm(input_tensor_a, input_tensor_b)
        else:
            raise Exception(f"GemmOp with dtype {compute_dtype} is not implemented")
        return output_tensor


class GemvOp(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
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
            bytes_per_cnt = dtype_size * input_element_num + get_dtype_bytes("float32") * output_element_num
        else:
            bytes_per_cnt = dtype_size * (input_element_num + output_element_num)
        return bytes_per_cnt

    def forward(self, input_tensor_a, input_tensor_b):
        compute_dtype = input_tensor_a.dtype
        output_tensor = None
        if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            output_tensor = torch.mm(input_tensor_a, input_tensor_b)
        else:
            raise Exception(f"GemvOp with dtype {compute_dtype} is not implemented")
        return output_tensor


class BatchGemmOp(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
            bytes_per_cnt = dtype_size * input_element_num + get_dtype_bytes("int32") * output_element_num * 2
        else:
            bytes_per_cnt = dtype_size * (input_element_num + output_element_num)
        return bytes_per_cnt

    def forward(self, input_tensor_a, input_tensor_b):
        compute_dtype = input_tensor_a.dtype
        output_tensor = None
        if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            output_tensor = torch.bmm(input_tensor_a, input_tensor_b)
        else:
            raise Exception(f"BatchGemmOp with dtype {compute_dtype} is not implemented")
        return output_tensor


class GroupGemmOp(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compute_size(self, input_shapes, dtype):
        """
        [
            [[M1, K1], [K1, N1]], 
            [[M2, K2], [K2, N2]]
        ]
        """
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
                bytes_per_cnt += dtype_size * input_element_num + get_dtype_bytes("float32") * output_element_num
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

        for problem_shape in input_shapes:
            a_shape, b_shape = problem_shape
            if torch_dtype in [torch.int8, torch.int32]:
                left_tensor = torch.randint(-3, 3, size=a_shape, dtype=torch_dtype, device=xpu_device)
                right_tensor = torch.randint(-3, 3, size=b_shape, dtype=torch_dtype, device=xpu_device)
            else:
                left_tensor = torch.randn(a_shape, dtype=torch_dtype, device=xpu_device)
                right_tensor = torch.randn(b_shape, dtype=torch_dtype, device=xpu_device)
            left_tensors.append(left_tensor)
            right_tensors.append(right_tensor)

        return [left_tensors, right_tensors]

    def forward(self, input_tensor_a, input_tensor_b):
        compute_dtype = input_tensor_a.dtype
        output_tensor_list = []
        for a, b in zip(input_tensor_a, input_tensor_b):
            if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
                output_tensor = torch.mm(a, b)
                output_tensor_list.append(output_tensor)
            else:
                raise Exception(f"GroupGemmOp with dtype {compute_dtype} is not implemented")
        return output_tensor_list


class Host2DeviceOp(torch.nn.Module):
    def __init__(self, xpu_device):
        super().__init__()
        self.xpu_device = xpu_device

    def process_inputs(self, input_tensors):
        new_inputs = input_tensors.cpu()
        return [new_inputs]

    def forward(self, input_tensors):
        assert input_tensors.device.type == "cpu"
        output_xpu = input_tensors.to(self.xpu_device)
        return output_xpu


class Device2HostOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        assert input_tensors.device.type != "cpu"
        output_cpu = input_tensors.cpu()
        return output_cpu


class AllReduceOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def forward(self, input_tensors):
        dist.all_reduce(input_tensors, group=self.group)
        return True


class AllGatherOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def process_inputs(self, input_tensors):
        input_tensor_list = list(
            torch.chunk(input_tensors, dist.get_world_size(self.group))
        )
        return [input_tensor_list]

    def forward(self, input_tensor_list):
        dist.all_gather(
            input_tensor_list,
            input_tensor_list[dist.get_rank(self.group)],
            group=self.group,
        )
        return True


class ReduceScatterOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def process_inputs(self, input_tensors):
        input_tensor_list = list(
            torch.chunk(input_tensors, dist.get_world_size(self.group))
        )
        return [input_tensor_list]

    def forward(self, input_tensor_list):
        dist.reduce_scatter(
            input_tensor_list[dist.get_rank(self.group)],
            input_tensor_list,
            group=self.group,
        )
        return True


class AllToAllOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def process_inputs(self, input_tensor, output_tensor):
        input_tensor_list = list(
            torch.chunk(input_tensor, dist.get_world_size(self.group))
        )
        output_tensor_list = list(
            torch.chunk(output_tensor, dist.get_world_size(self.group))
        )
        return [input_tensor_list, output_tensor_list]

    def forward(self, in_tensors_list, out_tensors_list):
        dist.all_to_all(out_tensors_list, in_tensors_list, group=self.group)
        return True


class BroadcastOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def forward(self, input_tensors):
        dist.broadcast(input_tensors, 0, self.group)
        return True


class P2POp(torch.nn.Module):
    def __init__(self, group, ranks, rank):
        super().__init__()
        self.group = group
        self.group_size = self.group.size()
        self.rank = rank
        self.ranks = ranks
        self.rank_size = len(ranks)

    def next_rank(self):
        return self.ranks[(self.rank + 1) % self.rank_size]

    def prev_rank(self):
        return self.ranks[(self.rank - 1) % self.rank_size]

    def forward(self, send_tensor, recv_tensor):
        reqs = []
        if self.rank != (self.group_size - 1):
            send_req = dist.isend(send_tensor, self.next_rank(), self.group)
            reqs.append(send_req)
        if self.rank != 0:
            recv_req = dist.irecv(recv_tensor, self.prev_rank(), self.group)
            reqs.append(recv_req)

        for req in reqs:
            req.wait()
        return True


class SinOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.sin(input_tensors)
        return result


class CosOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.cos(input_tensors)
        return result


class ExpOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.exp(input_tensors)
        return result


class ExponentialOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = input_tensors.exponential_()
        return result


class SiluOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.nn.functional.silu(input_tensors)
        return result


class GeluOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.nn.functional.gelu(input_tensors)
        return result


class SwiGLUOp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = 1
        self.v = 2

    def forward(self, input_tensors):
        result = (torch.nn.functional.sigmoid(input_tensors) * self.w) + (input_tensors * self.v)
        return result


class CastOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def set_dtype(self, src_dtype: str):
        target_dtype = "bfloat16" if src_dtype == "float32" else "float32"
        self.target_dtype = target_dtype
        self.target_torch_dtype = getattr(torch, target_dtype)

    def compute_size(self, input_shapes, dtype):
        torch_dtype = getattr(torch, dtype)
        self.set_dtype(dtype)
        dtype_size = get_dtype_bytes(dtype)
        target_dtype_size = get_dtype_bytes(self.target_dtype)
        element_num = sum([math.prod(shape) for shape in input_shapes])
        bytes_per_cnt = dtype_size * element_num + target_dtype_size * element_num
        return bytes_per_cnt

    def forward(self, input_tensors):
        result = input_tensors.to(self.target_torch_dtype)
        return result


class AddOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_a, input_tensor_b):
        result = input_tensor_a + input_tensor_b
        return result


class MulOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_a, input_tensor_b):
        result = input_tensor_a * input_tensor_b
        return result


class SubOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_a, input_tensor_b):
        result = input_tensor_a - input_tensor_b
        return result


class DivOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_a, input_tensor_b):
        result = input_tensor_a / input_tensor_b
        return result


class LayerNormOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.nn.functional.layer_norm(
            input_tensors, (input_tensors.shape[-1],)
        )
        return result


class SoftmaxOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.nn.functional.softmax(input_tensors, dim=-1)
        return result


class ReduceSumOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.sum(input_tensors, dim=-1)
        return result


class ReduceMinOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.min(input_tensors, dim=-1)
        return result


class ReduceMaxOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.max(input_tensors, dim=-1)
        return result


class IndexAddOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def process_inputs(self, input_tensor, source_tensor):
        index = torch.randint(0, input_tensor.shape[0], (source_tensor.shape[0],)).to(
            input_tensor.device
        )
        return [input_tensor, index, source_tensor]

    def forward(self, input_tensor, index, source_tensor):
        result = input_tensor.index_add_(0, index, source_tensor)
        return result


class SortOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.sort(input_tensors)
        return result


class UniqueOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.unique(input_tensors, return_counts=True)
        return result
