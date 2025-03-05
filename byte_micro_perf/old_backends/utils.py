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

from backends import module_store


def dump_communication_ops_report(
    op_name: str,
    torch_dtype,
    input_shapes: List[List[int]],
    compute_size_func, 
    group_size: int,
    bandwidth_limit: float,
    latency: float,
    error: str = ""
):
    # get dtype name and dtype_size
    dtype_name = str(torch_dtype).split(".")[-1]
    
    # ignore compute_size_func
    dtype_size = torch.tensor([], dtype=torch_dtype).element_size()
    element_num = math.prod(input_shapes[0])
    tensor_size = dtype_size * element_num

    mb = tensor_size / 1024 / 1024
    if error == "":
        algo_bw = tensor_size / latency / 1e3

        """
        allreduce:      2 * (group_size - 1) * (tensor_size / group_size)
        allgather:      1 * (group_size - 1) * (tensor_size / group_size)
        reducescatter:  1 * (group_size - 1) * (tensor_size / group_size)
        alltoall:       1 * (group_size - 1) * (tensor_size / group_size)
        broadcast:      tensor_size
        p2p:            tensor_size
        """
        if op_name in ["allgather", "reducescatter", "alltoall"]:
            bus_bw = algo_bw * (group_size - 1) / group_size
        elif op_name in ["allreduce"]:
            bus_bw = 2 * algo_bw * (group_size - 1) / group_size
        elif op_name in ["broadcast", "p2p", "device2host", "host2device"]:
            bus_bw = algo_bw

        bandwidth_utils = None
        if bandwidth_limit is not None:
            bandwidth_utils = round((algo_bw / bandwidth_limit) * 1e2, 2)
        report = {
            "Dtype": str(dtype_name),
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
            "Dtype": str(dtype_name),
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
    torch_dtype: str,
    input_shapes: List[List[int]], 
    compute_size_func, 
    bandwidth_limit: float,
    latency: float,
    error: str = ""
):
    # get dtype name and dtype_size
    dtype_name = str(torch_dtype).split(".")[-1]
    result = compute_size_func(input_shapes, torch_dtype)
    
    batch_size = result[0]
    tensor_size = result[1]
    input_tensor_size = result[2]
    output_tensor_size = result[3]
    
    if error == "":
        qps = round(1e6 / latency * batch_size, 2)
        algo_bw = tensor_size / latency / 1e3

        bandwidth_utils = None
        if bandwidth_limit is not None:
            bandwidth_utils = round((algo_bw / bandwidth_limit) * 1e2, 2)
        report = {
            "Dtype": str(dtype_name),
            "Tensor Shapes": input_shapes,
            "Read IO Size(MB)": round(input_tensor_size / 1024 / 1024, 2),
            "Write IO Size(MB)": round(output_tensor_size / 1024 / 1024, 2),
            "Memory Size(MB)": round(tensor_size / 1024 / 1024, 2),
            "Kernel bandwidth(GB/s)": round(algo_bw, 2),
            "Bandwidth Utilization(%)": bandwidth_utils,
            "Avg latency(us)": round(latency, 2),
            "QPS": qps,
        }
    else:
        report = {
            "Dtype": str(dtype_name),
            "Tensor Shapes": input_shapes,
            "Read IO Size(MB)": round(input_tensor_size / 1024 / 1024, 2),
            "Write IO Size(MB)": round(output_tensor_size / 1024 / 1024, 2),
            "Memory Size(MB)": round(tensor_size / 1024 / 1024, 2),
            "Kernel bandwidth(GB/s)": 0,
            "Bandwidth Utilization(%)": None,
            "Avg latency(us)": 0,
            "QPS": 0,
            "Error": error,
        }
    return report
