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

    def forward(self, input_tensor_a, input_tensor_b):
        compute_dtype = input_tensor_a.dtype
        output_tensor = None
        if compute_dtype == torch.int8:
            # to be realized
            pass
        elif compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            output_tensor = torch.mm(input_tensor_a, input_tensor_b)
        return output_tensor
    
    def compute_size(self, input_shapes, torch_dtype):
        # input_shapes: [[M, K], [K, N]]
        a_shape, b_shape = input_shapes
        M, K = a_shape
        K, N = b_shape
        d_shape = [M, N]
        dtype_size = get_dtype_bytes(torch_dtype)
        element_num = sum([math.prod(shape) for shape in [a_shape, b_shape, d_shape]])
        bytes_per_cnt = dtype_size * element_num
        return bytes_per_cnt



class BatchGemmOp(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input_tensor_a, input_tensor_b):
        compute_dtype = input_tensor_a.dtype
        output_tensor = None
        if compute_dtype == torch.int8:
            # to be realized
            pass
        elif compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            output_tensor = torch.bmm(input_tensor_a, input_tensor_b)
        return output_tensor

    def compute_size(self, input_shapes, torch_dtype):
        # input_shapes: [[bs, M, K], [bs, K, N]]
        a_shape, b_shape = input_shapes
        bs, M, K = a_shape
        bs, K, N = b_shape
        d_shape = [bs, M, N]
        dtype_size = get_dtype_bytes(torch_dtype)
        element_num = sum([math.prod(shape) for shape in [a_shape, b_shape, d_shape]])
        bytes_per_cnt = dtype_size * element_num
        return bytes_per_cnt
    


class GroupGemmOp(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input_tensor_a, input_tensor_b):
        compute_dtype = input_tensor_a.dtype
        output_tensor_list = []
        for a, b in zip(input_tensor_a, input_tensor_b):
            if compute_dtype == torch.int8:
                # to be realized
                pass
            elif compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
                output_tensor = torch.mm(a, b)
                output_tensor_list.append(output_tensor)
        return output_tensor_list


    def compute_size(self, input_shapes, torch_dtype):
        """
        [
            [[M1, K1], [K1, N1]], 
            [[M2, K2], [K2, N2]]
        ]
        """
        bytes_per_cnt = 0
        for problem_shape in input_shapes:
            a_shape, b_shape = problem_shape
            M, K = a_shape
            K, N = b_shape
            d_shape = [M, N]
            dtype_size = get_dtype_bytes(torch_dtype)
            element_num = sum([math.prod(shape) for shape in [a_shape, b_shape, d_shape]])
            bytes_per_cnt += dtype_size * element_num
        return bytes_per_cnt

    def custom_create_tensors(self, input_shapes, torch_dtype):
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
            left_tensors.append(
                torch.randint(0, 3, size=a_shape).type(torch_dtype).to(torch.device("cuda"))
            )
            right_tensors.append(
                torch.randint(0, 3, size=b_shape).type(torch_dtype).to(torch.device("cuda"))
            )
        return [left_tensors, right_tensors]





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
    




class AddOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_a, input_tensor_b):
        result = input_tensor_a + input_tensor_b
        return result


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


class GeluOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.nn.functional.gelu(input_tensors)
        return result


class ExponentialOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = input_tensors.exponential_()
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


class ExpOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.exp(input_tensors)
        return result



class SoftmaxOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        logits = torch.nn.functional.softmax(hidden_states, dim=-1)
        return logits


class LayerNormOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        logits = torch.nn.functional.layer_norm(
            hidden_states, (hidden_states.shape[-1],)
        )
        return logits




