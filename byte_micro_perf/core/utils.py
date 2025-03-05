import sys
import logging
import torch
from collections import namedtuple


# logger functions
logger = logging.getLogger("bytemlperf_micro_perf")
def setup_logger(loglevel: str):
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(loglevel.upper())
    logger.propagate = False





OpTensorInfo = namedtuple("TensorInfo", ["shape", "dtype", "device"])
OpSizeInfo = namedtuple("OpSizeInfo", [
    # tensor_size
    "input_tensor_size", 
    "output_tensor_size", 
    "tensor_size", 

    # io_size
    "read_bytes",
    "write_bytes",
    "io_bytes",

    # comm_size
    "algo_size",
    "bus_size"
])



def calc_tensor_size(tensor_info: OpTensorInfo):
    tensor_size = 1
    for dim in tensor_info.shape:
        tensor_size *= dim
    dtype_size = torch.tensor([], dtype=tensor_info.dtype).element_size()
    tensor_size *= dtype_size
    return tensor_size