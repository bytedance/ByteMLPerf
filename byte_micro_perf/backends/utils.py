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


def dump_communication_ops_report(
    op_name: str,
    dtype: str,
    input_shapes: List[List[int]],
    group_size: List[int],
    bandwidth_limit: float,
    latency: float,
):
    size = math.prod(input_shapes[0])
    torch_type = getattr(torch, dtype)
    if torch_type == torch.int32:
        dtype_size = torch.iinfo(torch_type).bits // 8
    else:
        dtype_size = torch.finfo(torch_type).bits // 8
    mb = dtype_size * size / 1024 / 1024
    algo_bw = dtype_size * size / latency / 1e3
    bus_bw = algo_bw * (group_size - 1) / group_size
    if op_name == "allreduce":
        bus_bw *= 2

    bandwidth_utils = None
    if bandwidth_limit is not None:
        bandwidth_utils = round((algo_bw / bandwidth_limit) * 1e2, 2)

    report = {
        "Dtype": dtype,
        "Memory Size(MB)": round(mb, 2),
        "Group": group_size,
        "Algo bandwidth(GB/s)": round(algo_bw, 2),
        "Bus bandwidth(GB/s)": round(bus_bw, 2),
        "Bandwidth Utilization(%)": bandwidth_utils,
        "Avg latency(us)": round(latency, 2),
    }
    return report


def dump_computation_ops_report(
    op_name: str,
    dtype: str,
    input_shapes: List[List[int]],
    bandwidth_limit: float,
    latency: float,
):
    if op_name == "add":
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
    elif op_name == "unique":
        size = sum([math.prod(shape) for shape in input_shapes])
    else:
        # out = func(in)
        # MAC_total = MAC_in + MAC_out
        size = sum([math.prod(shape) for shape in input_shapes]) * 2

    torch_type = getattr(torch, dtype)
    if torch_type == torch.int32:
        dtype_size = torch.iinfo(torch_type).bits // 8
    else:
        dtype_size = torch.finfo(torch_type).bits // 8
    mb = dtype_size * size / 1024 / 1024
    algo_bw = dtype_size * size / latency / 1e3

    bandwidth_utils = None
    if bandwidth_limit is not None:
        bandwidth_utils = round((algo_bw / bandwidth_limit) * 1e2, 2)

    report = {
        "Dtype": dtype,
        "Memory Size(MB)": round(mb, 2),
        "Algo bandwidth(GB/s)": round(algo_bw, 2),
        "Bandwidth Utilization(%)": bandwidth_utils,
        "Avg latency(us)": round(latency, 2),
    }
    return report
