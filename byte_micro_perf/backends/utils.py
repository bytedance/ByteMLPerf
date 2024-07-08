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

import numpy as np
import torch


def get_dtype_bytes(torch_type):
    dtype_size = 0
    if torch_type in [torch.int32, torch.int8]:
        dtype_size = torch.iinfo(torch_type).bits // 8
    elif torch_type in [torch.float32, torch.float16, torch.bfloat16]:
        dtype_size = torch.finfo(torch_type).bits // 8
    else:
        # not supported yet
        pass
    return dtype_size


def dump_communication_ops_report(
    op_name: str,
    dtype: torch.dtype,
    input_shapes: List[List[int]],
    group_size: List[int],
    bandwidth_limit: float,
    latency: float,
    error: str = ""
):
    size = math.prod(input_shapes[0])
    dtype_size = get_dtype_bytes(dtype)
    mb = dtype_size * size / 1024 / 1024
    if error == "":
        algo_bw = dtype_size * size / latency / 1e3

        """
        allreduce:      2 * (group_size - 1) * (tensor_size / group_size)
        allgather:      1 * (group_size - 1) * (tensor_size / group_size)
        reducescatter:  1 * (group_size - 1) * (tensor_size / group_size)
        alltoall:       1 * (group_size - 1) * (tensor_size / group_size)
        broadcast:      tensor_size
        """
        bus_bw = algo_bw * (group_size - 1) / group_size
        if op_name == "broadcast":
            bus_bw = algo_bw
        if op_name == "allreduce":
            bus_bw *= 2

        bandwidth_utils = None
        if bandwidth_limit is not None:
            bandwidth_utils = round((algo_bw / bandwidth_limit) * 1e2, 2)
        report = {
            "Dtype": str(dtype),
            "Tensor Shapes": input_shapes,
            "Memory Size(MB)": round(mb, 2),
            "Group": group_size,
            "Kernel bandwidth(GB/s)": round(algo_bw, 2),
            "Bus bandwidth(GB/s)": round(bus_bw, 2),
            "Bandwidth Utilization(%)": bandwidth_utils,
            "Avg latency(us)": round(latency, 2),
        }
    else:
        report = {
            "Dtype": str(dtype),
            "Tensor Shapes": input_shapes,
            "Memory Size(MB)": round(mb, 2),
            "Group": group_size,
            "Kernel bandwidth(GB/s)": 0,
            "Bus bandwidth(GB/s)": 0,
            "Bandwidth Utilization(%)": None,
            "Avg latency(us)": 0,
            "Error": error,
        }
    return report


def dump_computation_ops_report(
    op_name: str,
    dtype: torch.dtype,
    input_shapes: List[List[int]],
    bandwidth_limit: float,
    latency: float,
    error: str = ""
):
    if op_name in ["add", "mul", "sub", "div"]:
        # c = a + b
        # MAC_total = MAC_a + MAC_b + MAC_c
        size = sum(
            [math.prod(shape) for shape in input_shapes], math.prod(input_shapes[0])
        )
    elif op_name == "gemm":
        # c = gemm(a, b)
        # MAC_total = MAC_a + MAC_b + MAC_c
        M = input_shapes[0][0]
        K = input_shapes[0][1]
        N = input_shapes[1][1]
        size = M * K + K * N + M * N
    elif op_name == "batch_gemm":
        # c = batch_gemm(a, b)
        bs = input_shapes[0][0]
        M = input_shapes[0][1]
        K = input_shapes[0][2]
        N = input_shapes[1][2]
        size = bs * (M * K + K * N + M * N)
    elif op_name == "group_gemm":
        # c_list = group_gemm(a_list, b_list)
        size_list = []
        for problem_shape in input_shapes:
            M = problem_shape[0][0]
            K = problem_shape[0][1]
            N = problem_shape[1][1]
            size_list.append(M * K + K * N + M * N)
        size = sum(size_list)
    elif op_name in ["unique", "device2host", "host2device"]:
        size = sum([math.prod(shape) for shape in input_shapes])
    elif op_name == "cast":
        size = sum([math.prod(shape) for shape in input_shapes])
        size = size * (1 + 2) if dtype == torch.float32 else int(size * (1 + 0.5))
    else:
        # out = func(in)
        # MAC_total = MAC_in + MAC_out
        size = sum([math.prod(shape) for shape in input_shapes]) * 2

    dtype_size = get_dtype_bytes(dtype)
    mb = dtype_size * size / 1024 / 1024
    if error == "":
        algo_bw = dtype_size * size / latency / 1e3

        bandwidth_utils = None
        if bandwidth_limit is not None:
            bandwidth_utils = round((algo_bw / bandwidth_limit) * 1e2, 2)
        report = {
            "Dtype": str(dtype),
            "Tensor Shapes": input_shapes,
            "Memory Size(MB)": round(mb, 2),
            "Kernel bandwidth(GB/s)": round(algo_bw, 2),
            "Bandwidth Utilization(%)": bandwidth_utils,
            "Avg latency(us)": round(latency, 2),
        }
    else:
        report = {
            "Dtype": str(dtype),
            "Tensor Shapes": input_shapes,
            "Memory Size(MB)": round(mb, 2),
            "Kernel bandwidth(GB/s)": 0,
            "Bandwidth Utilization(%)": None,
            "Avg latency(us)": 0,
            "Error": error,
        }
    return report
