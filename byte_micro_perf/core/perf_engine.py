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

import os
import sys
import json
import time
import argparse
import importlib
import logging
import subprocess
import pathlib
import traceback
import random
from typing import Any, Dict, List
import itertools
from collections import namedtuple

import torch.multiprocessing as mp
import virtualenv

CUR_DIR = pathlib.Path.cwd()
FILE_DIR = pathlib.Path(sys.path[0])
BYTE_MLPERF_ROOT = FILE_DIR.parent
sys.path.insert(0, BYTE_MLPERF_ROOT.__str__())

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
        "--task_dir",
        default=str(BYTE_MLPERF_ROOT.joinpath("workloads")),
        help="The direcotry of tasks going to be evaluted, e.g., set to workloads"
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
        "--activate_venv", 
        action="store_true",
        help="Enable virtual environment to run the task",
    )
    args = parser.parse_args()
    return args


def load_workload(task: str, task_dir: str) -> Dict[str, Any]:
    """
    Return a list of dictionary with model Configuration
    Args: List[str]
    Returns: List[dic]
    """
    modules_dir = pathlib.Path(task_dir).absolute()
    # create empty workload json data
    workload_dict = {}
    for file in modules_dir.iterdir():
        if (
            file.stem.startswith('_')
            or file.stem.startswith('.')
            or file.is_dir()
            or file.suffix != '.json'
            or file.stem != task
        ):
            continue
        workload_dict = json.loads(file.read_text())

    if not workload_dict:
        log.error(f"could not find {task}.json in {modules_dir}.")
        exit(1)

    return workload_dict



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
                        batches = input_shape_group.get("batch", [])
                        kn = input_shape_group.get("KN", [])
                        if k and n:
                            kn.append([list(shape) for shape in itertools.product(k, n)])
                        for batch in batches:
                            for _kn in kn:
                                group_input_shape_list = []
                                for group in groups:
                                    group_input_shape_list.append([[group * batch, _kn[0]], [_kn[0], _kn[1]]])
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




ConfigInstance = namedtuple("ConfigInstance", ["dtype", "tensor_shapes", "index"])
ResultItem = namedtuple("ResultItem", ["config", "report"])


class PerfEngine:
    def __init__(self) -> None:
        super().__init__()

        self.args = get_args()
        self.workload = load_workload(self.args.task, self.args.task_dir)
        self.backend_type = self.args.hardware_type
        self.old_os_path = os.environ["PATH"]
        self.prev_sys_path = list(sys.path)
        self.real_prefix = sys.prefix

    def get_cpu_name(self):
        command = "lscpu | grep 'Model name' | awk -F: '{print $2}'"
        cpu_name = subprocess.check_output(command, shell=True)
        return cpu_name.decode().strip()



    def start_engine(self) -> None:

        if self.args.activate_venv:
            self.activate_venv(self.backend_type)

        # init backend
        hardware_type = self.backend_type
        log.info("Loading Heterogeneous Backend: {}".format(hardware_type))
        
        backend_module = importlib.import_module(
            "backends." + hardware_type + ".backend_" + hardware_type.lower())
        self.backend_class = getattr(backend_module, "Backend" + hardware_type)
        self.backend = self.backend_class(self.workload, self.args.vendor_path)

        # start process task
        log.info(
            "******************************************* Start to test op: [{}]. *******************************************".format(
                self.workload["operator"]
            )
        )

        # create output dir based on task
        output_dir = BYTE_MLPERF_ROOT.joinpath(
            "reports", self.backend_type, self.workload["operator"])
        output_dir.mkdir(parents=True, exist_ok=True)


        # get input shape info
        target_group_list = self.workload.get("group", [1])
        target_group_list.sort()
        device_count = getattr(self.backend, "get_device_count")()
        group_list = []
        for group in target_group_list:
            if group <= device_count:
                group_list.append(group)
            else:
                break
        dtype_list = self.workload.get("dtype", ["float32"])
        shape_list = parse_workload(self.workload)

        if not group_list or not dtype_list or not shape_list:
            log.error("empty group/dtype/shape")
            exit(1)

        test_list = []
        case_index = 0
        for dtype in dtype_list:
            for shape in shape_list:
                test_list.append(ConfigInstance(dtype, shape, case_index))
                case_index = case_index + 1

        
        try:
            mp.set_start_method("spawn", force=True)
        except Exception as e:
            traceback.print_exc()
            log.error(f"Set start method failed, error msg: {e}")


        # all operations will enter subprocess to test in parallel
        for group in group_list:
            log.info(f"Start to test group size: {group}")
            # TODO: test allreduce/allgather in parallel
            instance_num = device_count if group == 1 else group
            if self.workload["operator"] in ["device2host", "host2device"]:
                instance_num = 1





            input_queues = mp.Queue()
            output_queues = mp.Queue(maxsize=1)

            try:
                _subprocesses = mp.spawn(
                    fn=self.perf_func, 
                    args=(instance_num, group, output_dir, test_list, input_queues, output_queues), 
                    nprocs=instance_num, 
                    join=False, 
                    daemon=False
                )

                log.info("waiting for ranks to be ready")
                for _ in range(instance_num):
                    assert "ready" == output_queues.get()
                log.info("all ranks are ready and listening, init done")



                if group == 1:
                    for test_instance in test_list:
                        input_queues.put(test_instance, True)

                    for _ in range(instance_num):
                        input_queues.put("end", True)


                for process in _subprocesses.processes:
                    process.join()

            except Exception as e:
                traceback.print_exc()
                log.error(f"Execute task: {self.args.task} failed, group: {group}, error msg: {e}")

            time.sleep(1)

        if self.args.activate_venv:
            self.deactivate_venv()




    def perf_func(self, rank: int, *args):
        backend_instance = self.backend_class(self.workload, self.args.vendor_path)
        op_name = self.workload["operator"]
        
        world_size, group_size, output_dir, test_list, input_queues, output_queues = args

        # set device accroding to local_rank
        set_device_func = getattr(backend_instance, "set_device")
        set_device_func(rank)
        

        if world_size > 1:
            init_ccl_func = getattr(backend_instance, "initialize_ccl")
            init_ccl_func(rank, world_size)

        op = getattr(backend_instance, op_name.lower(), None)
        if op is not None and callable(op):
            op()
        else:
            raise ValueError(f"Unknown operation: {op_name.lower()}")


        output_queues.put("ready")

        result_list = []
        if group_size == 1:
            while True:
                test_instance = input_queues.get()
                if test_instance == "end":
                    break           
                
                print(f"rank {rank}: {test_instance.index + 1} / {len(test_list)}")

                test_dtype = test_instance.dtype
                test_shape = test_instance.tensor_shapes
                """
                input_shape could be:
                    List[int]: single shape. cos
                    List[List[int]]: multiple inputs. add
                    List[List[List[in]]]: multiple inputs with multiple problems. group_gemm
                """
                if isinstance(test_shape[0], int):
                    test_shape = [test_shape]
                try:
                    reports = backend_instance.perf(test_shape, test_dtype)
                except Exception as e:
                    traceback.print_exc()
                    log.error(f"Execute op: {op_name.lower()} failed, input_shape: {test_shape}, dtype: {test_dtype}, error msg: {e}")
                    reports = {}

                result_list.append(ResultItem(test_instance, reports))

            output_result_list = []
            if world_size > 1:
                all_gather_object_func = getattr(backend_instance, "all_gather_object")
                all_result_list = all_gather_object_func(result_list)
                for data in all_result_list:
                    output_result_list.extend(data)
            else:
                output_result_list = result_list

            result_list = sorted(output_result_list, key=lambda x: x.config.index)


        elif group_size > 1:
            for i, test_instance in enumerate(test_list):
                if rank == 0:
                    print(f"rank {rank}: {test_instance.index + 1} / {len(test_list)}")

                test_dtype = test_instance.dtype
                test_shape = test_instance.tensor_shapes
                """
                input_shape could be:
                    List[int]: single shape. cos
                    List[List[int]]: multiple inputs. add
                    List[List[List[in]]]: multiple inputs with multiple problems. group_gemm
                """
                if isinstance(test_shape[0], int):
                    test_shape = [test_shape]
                try:
                    reports = backend_instance.perf(test_shape, test_dtype)
                except Exception as e:
                    traceback.print_exc()
                    log.error(f"Execute op: {op_name.lower()} failed, input_shape: {test_shape}, dtype: {test_dtype}, error msg: {e}")
                    reports = {}

                result_list.append(ResultItem(test_instance, reports))


        if rank == 0:
            print(f"{len(result_list)} tasks finished.")

            dtype_results_mapping = {}
            for result in result_list:
                if result.config.dtype not in dtype_results_mapping:
                    dtype_results_mapping[result.config.dtype] = []
                dtype_results_mapping[result.config.dtype].append(result)

            for dtype, results in dtype_results_mapping.items():
                dtype_results_mapping[dtype] = sorted(results, key=lambda x: x.config.index)
                
                base_report = {
                    "Operator": op_name.upper(),
                    "Backend": self.backend_type,
                    "Host Info": self.get_cpu_name(),
                    "Device Info": getattr(self.backend, "get_device_name")(),
                    "Performance": [result.report for result in dtype_results_mapping[dtype]]
                }
                
                filename = (
                    f"result-{str(dtype)}"
                    + (
                        f"-group{group_size}"
                        if group_size > 1
                        else ""
                    )
                    + ".json"
                )
                filepath = output_dir.joinpath(filename)
                with open(filepath, "w") as f:
                    json.dump(base_report, f, indent=4)
        
        if world_size > 1:
            destroy_group_func = getattr(backend_instance, "destroy_process_group")
            destroy_group_func()

        return True
                


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
