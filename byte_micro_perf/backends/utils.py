from typing import List
import math
import torch

def write_communication_ops_report(op_name: str, dtype: str, input_shape: List[List[int]], group_size: List[int], latency: int):
    if type(input_shape[0]) == list:
        input_shape = input_shape[0]    
    size = math.prod(input_shape)
    dtype_size = torch.finfo(getattr(torch, dtype)).bits // 8
    mb = dtype_size * size / 1024 / 1024
    algo_bw = dtype_size * size / latency / 1e9
    bus_bw = algo_bw * (group_size - 1) / group_size
    if op_name == 'allreduce':
        bus_bw *=2
    report = {
        "dtype": "{}".format(dtype),
        "shape": input_shape,
        "bus bandwidth" : bus_bw,
        "avg latency(ms)" : latency,
        "theoretical bandwidth" : 178,
        "theoretical latency"  : 1.3,
        "mfu" : 0.87
    }
    return report

def write_computation_ops_report(dtype: str, input_shapes: List[List[int]], iterations: int, execution_history: List[int]):
    report = {
        "dtype": "{}".format(dtype),
        "shape": input_shapes,
        "ops(ops per-sec)" : round(iterations / sum(execution_history), 2),
        "avg latency(ms)" : round(sum(execution_history) * 1000 / len(execution_history), 2),
        "theoretical ops" : 178,
        "theoretical latency"  : 1.3,
        "theoretical io"  : 2.3,
        "mfu" : 0.87
    }
    return report