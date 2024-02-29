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

import torch
import torch.distributed as dist


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
        result = torch.unique(input_tensors)
        return result


class ExpOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        result = torch.exp(input_tensors)
        return result


class GemmOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_a, input_tensor_b):
        logits = torch.matmul(input_tensor_a, input_tensor_b)
        return logits


class SoftmaxOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        logits = torch.nn.functional.softmax(hidden_states, dim=-1)
        return logits


class AllReduceOp(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group

    def forward(self, input_tensors):
        dist.all_reduce(input_tensors[0], group=self.group)
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


class Device2HostOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensors):
        assert input_tensors[0].device.type != "cpu"
        output_cpu = input_tensors[0].cpu()
        return output_cpu


class Host2DeviceOp(torch.nn.Module):
    def __init__(self, xpu_device):
        super().__init__()
        self.xpu_device = xpu_device

    def process_inputs(self, input_tensors):
        new_inputs = input_tensors[0].cpu()
        return [new_inputs]

    def forward(self, input_tensors):
        assert input_tensors[0].device.type == "cpu"
        output_xpu = input_tensors[0].to(self.xpu_device)
        return output_xpu
