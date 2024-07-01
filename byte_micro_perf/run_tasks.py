import subprocess
import argparse
from typing import List


OP_LIST = [
    "add",
    "cos",
    "sin",
    "exp",
    "gelu",
    "gemm",
    "layernorm",
    "softmax",
    "unique",
    "exponential",
    "indexadd",
    "sort",
    "device2host",
    "host2device",
]

CCL_LIST = ["allgather", "allreduce", "alltoall", "broadcast", "reducescatter"]


def run_from_list(op_list: List[str]) -> None:
    for op in op_list:
        cmd = f"python launch.py --task {op} --hardware MTGPU"
        subprocess.call(cmd, shell=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        help="normal, mccl, all",
    )

    return parser.parse_args()


def _main() -> None:
    args = parse_args()
    task = args.task
    if task == "all":
        run_from_list(OP_LIST+CCL_LIST)
    elif task == "normal":
        run_from_list(OP_LIST)
    elif task == "mccl":
        run_from_list(CCL_LIST)


if __name__ == '__main__':
    _main()
