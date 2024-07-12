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

import argparse
import importlib
import json
import logging
import math
import os
import subprocess
import sys
import pathlib
import traceback
import random
from typing import Any, Dict, List
import itertools


import torch.multiprocessing as mp
import virtualenv

BYTE_MLPERF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BYTE_MLPERF_ROOT)

from backends.backend import Backend

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="gemm",
        help="The task going to be evaluted, refs to workloads/",
    )
    parser.add_argument(
        "--hardware_type",
        default="GPU",
        help="The backend going to be evaluted, refs to backends/",
    )
    parser.add_argument(
        "--vendor_path",
        required=False,
        help="The hardware configs need to be loaded, refs to vendor_zoo/NVIDIA/A100-PCIe.json",
    )
    parser.add_argument(
        "--compile_only", action="store_true", help="Run compilation only"
    )

    args = parser.parse_args()
    return args


def load_workload(task: str) -> Dict[str, Any]:
    """
    Return a list of dictionary with model Configuration
    Args: List[str]
    Returns: List[dic]
    """
    modules_dir = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/workloads"
    )

    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".json") or os.path.isdir(path))
            and file[: file.find(".json")] == task
        ):
            module_name = file
            with open("workloads/" + module_name, "r") as f:
                workload_dict = json.load(f)
            return workload_dict
    else:
        log.error(
            "Task name: [ {} ] was not found, please check your task name".format(task)
        )

def parse_workload(workload):
    shape_list = []
    if "input_shape_list" in workload:
        shape_list.extend(workload["input_shape_list"])
    # gemm or batch_gemm
    elif "M/K/N" in workload:
        if "batch_size" in workload:
            for batch_size in workload["batch_size"]:
                for M, K, N in workload["M/K/N"]:
                    shape_list.append([
                        [batch_size, M, K],
                        [batch_size, K, N]
                    ])
        else:
            for M, K, N in workload["M/K/N"]:
                shape_list.append([[M, K], [K, N]])
    # group_gemm
    elif "MKN_choices" in workload:
        seed = workload["seed"]
        MKN_list = workload["MKN_choices"]
        problems_list = workload["problems"]

        random.seed(seed)
        for problems in problems_list:
            cur_inputs = []
            for _ in range(problems):
                M, K, N = [random.choice(MKN_list) for _ in range(3)]
                cur_shapes = [[M, K], [K, N]]
                cur_inputs.append(cur_shapes)
        shape_list.append(cur_inputs)


    if "input_shape_groups" in workload:
        input_shape_groups = workload["input_shape_groups"] if isinstance(workload["input_shape_groups"], list) else [workload["input_shape_groups"]]

        for input_shape_group in input_shape_groups:
            if "inputs" in input_shape_group:
                input_shape_list = []
                for input_shapes in input_shape_group["inputs"]:
                    input_shape_list.append([list(shape) for shape in itertools.product(*input_shapes)])
                if len(input_shape_list) == 1:
                    shape_list.extend(input_shape_list[0])
                else:
                    shape_list.extend([list(input_shape) for input_shape in zip(*input_shape_list)])

            else:
                gemm_keys = ["M", "K", "N", "MN", "MK", "KN"]
                gemm_values = [input_shape_group.get(k, []) for k in gemm_keys]
                if any(gemm_values):
                    m ,k, n, mn, mk, kn = gemm_values
                    # batch gemm
                    if "batch_size" in input_shape_group:
                        bs = input_shape_group.get("batch_size", [])
                        if m and n and k:
                            for p in itertools.product(bs, m, k, n):
                                shape_list.append([[p[0], p[1], p[2]], [p[0], p[2], p[3]]])
                        if mn and k:
                            for p in itertools.product(bs, mn, k):
                                shape_list.append([[p[0], p[1][0], p[2]], [p[0], p[2], p[1][1]]])
                        if mk and n:
                            for p in itertools.product(bs, mk, n):
                                shape_list.append([[p[0], p[1][0], p[1][1]], [p[0], p[1][1], p[2]]])
                        if m and kn:
                            for p in itertools.product(bs, m, kn):
                                shape_list.append([[p[0], p[1], p[2][0]], [p[0], p[2][0], p[2][1]]])
                    # group gemm
                    elif "gemm_group" in input_shape_group:
                        groups = input_shape_group.get("gemm_group", [])
                        kn = input_shape_group.get("KN", [])
                        if k and n:
                            kn.append([list(shape) for shape in itertools.product(k, n)])
                        for group in groups:
                            for _kn in kn:
                                group_input_shape_list = []
                                for m in group:
                                    group_input_shape_list.append([[m, _kn[0]], [_kn[0], _kn[1]]])
                                shape_list.append(group_input_shape_list)
                    # gemm
                    else:
                        if m and n and k:
                            for p in itertools.product(m, k, n):
                                shape_list.append([[p[0], p[1]], [p[1], p[2]]])
                        if mn and k:
                            for p in itertools.product(mn, k):
                                shape_list.append([[p[0][0], p[1]], [p[1], p[0][1]]])
                        if mk and n:
                            for p in itertools.product(mk, n):
                                shape_list.append([[p[0][0], p[0][1]], [p[0][1], p[1]]])
                        if m and kn:
                            for p in itertools.product(m, kn):
                                shape_list.append([[p[0], p[1][0]], [p[1][0], p[1][1]]])
    return shape_list


class PerfEngine:
    def __init__(self) -> None:
        super().__init__()
        self.args = get_args()
        self.workload = load_workload(self.args.task)
        self.backend_type = self.args.hardware_type
        self.old_os_path = os.environ["PATH"]
        self.prev_sys_path = list(sys.path)
        self.real_prefix = sys.prefix

    def init_process(self, rank: int, world_size: int):
        """
        Initialize the distributed environment.

        """
        initialize_func = getattr(self.backend, "initialize_ccl")

        # world_size may excced available device count
        ret = initialize_func(rank, world_size)
        if ret is not None and not ret:
            return

        status = self.start_perf(self.workload)
        return status

    def init_backend(self, hardware_type: str) -> Backend:
        """
        Load related compile backend with input hardware type

        Arguments: str

        Returns: Heterogeneous Backend()
        """
        log.info("Loading Heterogeneous Backend: {}".format(hardware_type))

        backend = importlib.import_module(
            "backends." + hardware_type + ".backend_" + hardware_type.lower()
        )
        backend = getattr(backend, "Backend" + hardware_type)
        return backend(self.workload, self.args.vendor_path)

    def start_engine(self) -> None:
        status = self.activate_venv(self.backend_type)
        if not status:
            log.warning("Activate virtualenv Failed, Please Check...")

        self.backend = self.init_backend(self.backend_type)
        output_dir = os.path.abspath("reports/" + self.backend_type)
        os.makedirs(output_dir, exist_ok=True)

        if self.args.task in ["allreduce", "allgather", "reducescatter", "alltoall", "broadcast", "p2p"]:
            for group in self.workload["group"]:
                try:
                    mp.spawn(fn=self.init_process, args=(group,), nprocs=group)
                except Exception as e:
                    traceback.print_exc()
                    log.error(f"Execute task: {self.args.task} failed, group: {group}, error msg: {e}")
        else:
            status = self.start_perf(self.workload)

        self.deactivate_venv()

    def start_perf(self, workload: Dict[str, Any]) -> bool:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            log.info(
                "******************************************* Start to test op: [{}]. *******************************************".format(
                    workload["operator"]
                )
            )

        # Initalize Output Dir and Reports
        output_dir = pathlib.Path("reports").joinpath(self.backend_type).joinpath(workload["operator"])
        os.makedirs(output_dir, exist_ok=True)

        op_name = workload["operator"]
        base_report = {
            "Operator": op_name.upper(),
            "Backend": self.backend_type,
            "Host Info": self.get_cpu_name(),
            "Device Info": getattr(self.backend, "get_device_name")(),
        }

        op = getattr(self.backend, op_name.lower(), None)
        if op is not None and callable(op):
            op()
        else:
            raise ValueError(f"Unknown operation: {op_name.lower()}")

        # get input shape info
        shape_list = parse_workload(self.workload)

        # dtype list
        dtype_list = self.workload["dtype"]

        for dtype in dtype_list:
            perf_reports = []
            base_report["Performance"] = {}

            for input_shape in shape_list:
                """
                input_shape could be:
                  List[int]: single shape. cos
                  List[List[int]]: multiple inputs. add
                  List[List[List[in]]]: multiple inputs with multiple problems. group_gemm
                """
                if local_rank == 0:
                    log.info(f"Execute op: [{op_name.lower()}], input_shape: {input_shape}, dtype: {dtype}")
                if isinstance(input_shape[0], int):
                    input_shape = [input_shape]
                try:
                    reports = self.backend.perf(input_shape, dtype)
                except Exception as e:
                    traceback.print_exc()
                    log.error(f"Execute op: {op_name.lower()} failed, input_shape: {input_shape}, dtype: {dtype}, error msg: {e}")
                    reports = {}
                perf_reports.append(reports)
            base_report["Performance"] = perf_reports

            # write output to json file
            has_group = "Group" in base_report["Performance"][0]
            output_report_path = (
                f"result-{str(dtype)}"
                + (
                    f"-group{base_report['Performance'][0]['Group']}"
                    if has_group
                    else ""
                )
                + ".json"
            )
            output_report_path = os.path.join(output_dir, output_report_path)
            if local_rank == 0:
                # logging.info(base_report["Performance"])
                with open(output_report_path, "w") as file:
                    json.dump(base_report, file, indent=4)
        if local_rank == 0:
            log.info(
                "******************************************* Test op: [{}] SUCCESS. *******************************************".format(
                    workload["operator"]
                )
            )
        return True

    def get_cpu_name(self):
        command = "lscpu | grep 'Model name' | awk -F: '{print $2}'"
        cpu_name = subprocess.check_output(command, shell=True)
        return cpu_name.decode().strip()

    def activate_venv(self, hardware_type: str) -> bool:
        if os.path.exists("backends/" + hardware_type + "/requirements.txt"):
            log.info("Activating Virtual Env for " + hardware_type)

            venv_dir = os.path.join("backends", hardware_type + "/venv")
            activate_file = os.path.join(venv_dir, "bin", "activate_this.py")
            if not os.path.exists(venv_dir):
                log.info("venv not exist, Creating Virtual Env for " + hardware_type)

                virtualenv.create_environment(venv_dir, True)

                exec(open(activate_file).read(), {"__file__": activate_file})
                python_path = os.path.join(venv_dir, "bin", "python3")
                subprocess.call(
                    [python_path, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]
                )
                subprocess.call(
                    [
                        python_path,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        "backends/" + hardware_type + "/requirements.txt",
                        "-q",
                    ]
                )
            else:
                exec(open(activate_file).read(), {"__file__": activate_file})
                """
                just in case install failed in pre-run.
                """
                python_path = os.path.join(venv_dir, "bin", "python3")
                subprocess.call(
                    [python_path, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]
                )
                subprocess.call(
                    [
                        python_path,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        "backends/" + hardware_type + "/requirements.txt",
                        "-q",
                    ]
                )

                if not hasattr(sys, "real_prefix"):
                    return False
                return True
        return True

    def deactivate_venv(self):
        sys.path[:0] = self.prev_sys_path  # will also revert the added site-packages
        sys.prefix = self.real_prefix
        os.environ["PATH"] = self.old_os_path


if __name__ == "__main__":
    engine = PerfEngine()
    engine.start_engine()
