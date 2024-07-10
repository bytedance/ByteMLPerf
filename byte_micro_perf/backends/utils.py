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


def get_dtype_bytes(dtype: str):
    torch_dtype = getattr(torch, dtype)
    dtype_size = 0
    if torch_dtype in [torch.int32, torch.int8]:
        dtype_size = torch.iinfo(torch_dtype).bits // 8
    elif torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        dtype_size = torch.finfo(torch_dtype).bits // 8
    else:
        # not supported yet
        pass
    return dtype_size


def get_io_amount(op_name, input_shapes, dtype):
    batch_size = input_shapes[0][0]
    dtype_size = get_dtype_bytes(dtype)
    if op_name in ["add", "mul", "sub", "div"]:
        # c = a + b
        read_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes])
        write_io_amount = dtype_size * math.prod(input_shapes[0])
    elif op_name == "gemm":
        M = input_shapes[0][0]
        K = input_shapes[0][1]
        N = input_shapes[1][1]
        read_io_amount = dtype_size * (M * K + K * N)
        if dtype != torch.int8:
            write_io_amount = dtype_size * (M * N)
        else:
            write_io_amount = get_dtype_bytes("int32") * (M * N)
    elif op_name == "batch_gemm":
        bs = input_shapes[0][0]
        M = input_shapes[0][1]
        K = input_shapes[0][2]
        N = input_shapes[1][2]
        read_io_amount = dtype_size * bs * (M * K + K * N)
        if dtype != torch.int8:
            write_io_amount = dtype_size * bs * (M * N)
        else:
            write_io_amount = get_dtype_bytes("int32") * bs * (M * N)
    elif op_name == "group_gemm":
        in_size_list = []
        out_size_list = []
        m_list = []
        for problem_shape in input_shapes:
            M = problem_shape[0][0]
            K = problem_shape[0][1]
            N = problem_shape[1][1]
            in_size_list.append(M * K + K * N)
            out_size_list.append(M * N)
            m_list.append(M)
        batch_size = sum(m_list)
        read_io_amount = dtype_size * sum(in_size_list)
        if dtype != torch.int8:
            write_io_amount = dtype_size * sum(out_size_list)
        else:
            write_io_amount = get_dtype_bytes("int32") * sum(out_size_list)
    elif op_name in ["device2host"]:
        read_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes])
        write_io_amount = 0
    elif op_name in ["host2device"]:
        read_io_amount = 0
        write_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes])
    elif op_name in ["reduce_sum", "reduce_max", "reduce_min"]:
        read_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes])
        write_io_amount = dtype_size * sum([math.prod(shape[:-1]) for shape in input_shapes])
    elif op_name in ["unqiue", "sort"]:
        read_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes])
        write_io_amount = 2 * dtype_size * sum([math.prod(shape) for shape in input_shapes])
    elif op_name == "cast":
        read_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes])
        write_io_amount = read_io_amount / 2 if dtype == torch.float32 else read_io_amount * 2
    elif op_name in ["index_add"]:
        read_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes]) + get_dtype_bytes("int32") * input_shapes[1][0]
        write_io_amount = dtype_size * math.prod(input_shapes[0])
    else:
        read_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes])
        write_io_amount = dtype_size * sum([math.prod(shape) for shape in input_shapes])

    total_io_amount = read_io_amount + write_io_amount

    return batch_size, total_io_amount, read_io_amount, write_io_amount


def dump_communication_ops_report(
    op_name: str,
    dtype: str,
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
        p2p:            tensor_size
        """
        bus_bw = algo_bw * (group_size - 1) / group_size
        if op_name in ["broadcast", "p2p"]:
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
    dtype: str,
    input_shapes: List[List[int]],
    bandwidth_limit: float,
    latency: float,
    error: str = ""
):
    batch_size, total_io_amount, read_io_amount, write_io_amount = get_io_amount(op_name, input_shapes, dtype)

    if error == "":
        qps = round(1000 / latency * batch_size, 2)
        algo_bw = total_io_amount / latency / 1e3

        bandwidth_utils = None
        if bandwidth_limit is not None:
            bandwidth_utils = round((algo_bw / bandwidth_limit) * 1e2, 2)
        report = {
            "Dtype": str(dtype),
            "Tensor Shapes": input_shapes,
            "Read IO Size(MB)": round(read_io_amount / 1024 / 1024, 2),
            "Write IO Size(MB)": round(write_io_amount / 1024 / 1024, 2),
            "Memory Size(MB)": round(total_io_amount / 1024 / 1024, 2),
            "Kernel bandwidth(GB/s)": round(algo_bw, 2),
            "Bandwidth Utilization(%)": bandwidth_utils,
            "Avg latency(us)": round(latency, 2),
            "QPS": qps,
        }
    else:
        report = {
            "Dtype": str(dtype),
            "Tensor Shapes": input_shapes,
            "Read IO Size(MB)": round(read_io_amount / 1024 / 1024, 2),
            "Write IO Size(MB)": round(write_io_amount / 1024 / 1024, 2),
            "Memory Size(MB)": round(total_io_amount / 1024 / 1024, 2),
            "Kernel bandwidth(GB/s)": 0,
            "Bandwidth Utilization(%)": None,
            "Avg latency(us)": 0,
            "QPS": 0,
            "Error": error,
        }
    return report
