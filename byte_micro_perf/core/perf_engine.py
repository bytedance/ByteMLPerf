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
import random
import datetime
import signal
import argparse
import importlib
import logging
import subprocess
import pathlib
import traceback

import prettytable

from typing import Any, Dict, List
import itertools
from collections import namedtuple

import torch.distributed
import torch.multiprocessing as mp
import virtualenv

import torch

# directory config
CUR_DIR = pathlib.Path.cwd().absolute()
FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR.parent
sys.path.insert(0, str(BYTE_MLPERF_ROOT))

# logger config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PerfEngine")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware_type", type=str)
    parser.add_argument("--task_dir", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--report_dir", type=str)
    parser.add_argument("--numa_world_size", type=int, default=1)
    parser.add_argument("--numa_rank", type=int, default=0)
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
        logger.error(f"could not find {task}.json in {modules_dir}.")
        exit(1)

    return workload_dict



def parse_workload(workload):
    shape_list = []

    if "input_shape_groups" not in workload:
        return shape_list
    
    input_shape_groups = workload["input_shape_groups"] if isinstance(workload["input_shape_groups"], list) else [workload["input_shape_groups"]]

    for input_shape_group in input_shape_groups:
        if "inputs" in input_shape_group:
            input_shape_list = []
            for input_shapes in input_shape_group["inputs"]:
                input_shape_list.append([list(shape) for shape in itertools.product(*input_shapes)])

            if len(input_shape_list) == 1:
                shape_list.extend(input_shape_list[0])
            else:
                max_num_cases = max([len(input_shape) for input_shape in input_shape_list])
                for i in range(max_num_cases):
                    test_cases = []
                    for input_shape in input_shape_list:
                        test_cases.append(input_shape[i%(len(input_shape))])
                    shape_list.append(test_cases)

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




ConfigInstance = namedtuple("ConfigInstance", ["dtype", "tensor_shapes", "index", "total"])
ResultItem = namedtuple("ResultItem", ["config", "report"])


class PerfEngine:
    def __init__(self) -> None:
        super().__init__()

        self.args = get_args()

        # get workload
        self.workload = load_workload(self.args.task, self.args.task_dir)
        self.op_name = self.workload["operator"]

        # init backend
        self.backend_type = self.args.hardware_type
        logger.info("Loading Heterogeneous Backend: {}".format(self.backend_type))
        backend_module = importlib.import_module(
            "backends." + self.backend_type + ".backend_" + self.backend_type.lower())
        self.backend_class = getattr(backend_module, "Backend" + self.backend_type)
        self.backend = self.backend_class(self.workload)

        # device related
        self.device_count = self.backend.get_device_count()
        self.device_name = self.backend.get_device_name()
        self.avail_devices = list(range(self.device_count))
        self.parallel = self.args.parallel
        self.device_config = self.args.device

        self.numa_world_size = self.args.numa_world_size
        self.numa_rank = self.args.numa_rank

        # target devices
        self.target_devices = []        
        if self.device_config == "all":
            self.target_devices = self.avail_devices
        else:
            for d in self.device_config.split(","):
                if d.isdigit() and int(d) < self.device_count:
                    self.target_devices.append(int(d))
            self.target_devices.sort()
        if not self.target_devices:
            logger.error("no valid device")
            exit(1)

        
        self.device_per_numa = len(self.target_devices) // self.numa_world_size
        self.device_offset = self.numa_rank * self.device_per_numa
        self.ori_target_devices = self.target_devices.copy()
        self.target_devices = self.target_devices[self.device_offset:self.device_offset + self.device_per_numa]
        self.target_devices_num = len(self.target_devices)

        self.target_group_list = []
        for group_size in self.workload.get("group", [0]):
            if group_size <= self.target_devices_num * self.numa_world_size:
                    self.target_group_list.append(group_size)
        self.target_group_list.sort()


        pt = prettytable.PrettyTable()
        pt.field_names = ["Key", "Value"]
        pt.add_row(["op_name", self.op_name])
        pt.add_row(["hardware_type", self.backend_type])
        pt.add_row(["device_name", self.device_name])
        pt.add_row(["device_count", self.device_count])
        if self.numa_world_size > 1:
            pt.add_row(["numa_world_size", self.numa_world_size])
            pt.add_row(["numa_rank", self.numa_rank])
            pt.add_row(["device_per_numa", self.device_per_numa])
        pt.add_row(["available_devices", self.avail_devices])
        pt.add_row(["ori_target_devices", self.ori_target_devices])
        pt.add_row(["target_devices", self.target_devices])
        pt.add_row(["parallel", self.parallel])
        pt.add_row(["group_size_list", self.target_group_list])
        print(pt)



        self.dtype_list = self.workload.get("dtype", ["float32"])
        self.shape_list = parse_workload(self.workload)
        if not self.target_group_list or not self.dtype_list or not self.shape_list:
            logger.error("empty group/dtype/shape")
            exit(1)
    
        # get report dir
        self.report_dir = pathlib.Path(self.args.report_dir)
        self.hardware_report_dir = self.report_dir.joinpath(self.backend_type)
        self.operator_report_dir = self.hardware_report_dir.joinpath(self.op_name)
        self.operator_report_dir.mkdir(parents=True, exist_ok=True)

        # version info
        self.version = self.get_version()


    def get_version(self):
        version = ""
        try:
            version_file = os.path.join(str(BYTE_MLPERF_ROOT), "../VERSION")
            with open(version_file) as f:
                _version = f.read().splitlines()
            version = '.'.join(v.split('=')[1] for v in _version)
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"get bytemlperf version failed, error msg: {e}")
        return version

    def get_cpu_name(self):
        command = "lscpu | grep 'Model name' | awk -F: '{print $2}'"
        cpu_name = subprocess.check_output(command, shell=True)
        return cpu_name.decode().strip()

    def get_numa_config(self):
        command = "lscpu | grep 'NUMA node' | awk -F: '{print $2}'"
        numa_config = subprocess.check_output(command, shell=True)

        subitems = []
        for item in numa_config.decode().splitlines():
            subitems.append(item.strip())
    
        return ";".join(subitems)


    def process_results(self, result_list, group_size=0):
        dtype_results_mapping = {}
        for result in result_list:
            if result.config.dtype not in dtype_results_mapping:
                dtype_results_mapping[result.config.dtype] = []
            dtype_results_mapping[result.config.dtype].append(result)

        for dtype, results in dtype_results_mapping.items():
            dtype_results_mapping[dtype] = sorted(results, key=lambda x: x.config.index)

            base_report = {
                "Operator": self.workload["operator"].upper(),
                "Backend": self.backend_type,
                "Host Info": self.get_cpu_name(),
                "Numa Info": self.get_numa_config(),
                "Device Info": self.device_name,
                "Version": self.version,
                "Execution Date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Performance": [result.report for result in dtype_results_mapping[dtype]]
            }
            filename = (
                f"result-{str(dtype)}"
                + (f"-group{group_size}" if group_size > 0 else "")
                + ".json"
            )
            filepath = self.operator_report_dir.joinpath(filename)
            with open(filepath, "w") as f:
                json.dump(base_report, f, indent=4)


    def start_engine(self) -> None:
        # subprocesses terminate when main process is terminated
        subprocess_pids = []
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, exiting...")
            if subprocess_pids:
                for pid in subprocess_pids:
                    logger.info(f"terminate subprocess: {pid}")
                    os.kill(pid, signal.SIGTERM)
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


        try:
            mp.set_start_method("spawn", force=True)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Set start method failed, error msg: {e}")
            sys.exit(1)


        test_list = []
        case_index = 0
        for dtype in self.dtype_list:
            for shape in self.shape_list:
                test_list.append(ConfigInstance(
                    dtype, shape, 
                    case_index + 1, 
                    len(self.dtype_list) * len(self.shape_list)
                ))
                case_index = case_index + 1


        # computation ops
        if self.target_group_list == [0]:
            # only numa node 0
            if self.numa_rank != 0:
                return

            instance_num = len(self.target_devices) if self.parallel else 1
            print(f"numa_node: 0, instance_num: {instance_num}")

            input_queues = mp.Queue()
            output_queues = mp.Queue(maxsize=1)
            try:
                _subprocesses = mp.spawn(
                    fn=self.compute_op_perf, 
                    args=(
                        self.target_devices, instance_num, 
                        input_queues, output_queues
                    ), 
                    nprocs=instance_num,
                    join=False,
                    daemon=False
                )
                subprocess_pids = _subprocesses.pids()
                for _ in range(instance_num):
                    assert "ready" == output_queues.get()
                logger.info("all ranks are ready and listening, init done")


                for test_instance in test_list:
                    input_queues.put(test_instance, False)
                for _ in range(instance_num):
                    input_queues.put(None, False)

                result_list = []
                for _ in range(instance_num):
                    result_list.extend(output_queues.get())
                result_list = sorted(result_list, key=lambda x: x.config.index)

                self.process_results(result_list)

                for process in _subprocesses.processes:
                    process.join()
            except Exception as e:
                logger.error(f"Create subprocesses failed, error msg: {e}")
                sys.exit(1)



        # communication ops
        else:
            instance_num = self.device_per_numa
            print(f"numa_node: {self.numa_rank}, instance_num: {instance_num}")

            input_queues = mp.Queue()
            output_queues = mp.Queue(maxsize=1)

            try:
                _subprocesses = mp.spawn(
                    fn=self.communication_op_perf,
                    args=(
                        self.ori_target_devices, 
                        self.target_devices, 
                        instance_num, 
                        self.numa_world_size, self.numa_rank, 
                        self.target_group_list, test_list, 
                        input_queues, output_queues
                    ), 
                    nprocs=instance_num, 
                    join=False,
                    daemon=False
                )
                subprocess_pids = _subprocesses.pids()
                for _ in range(instance_num):
                    assert "ready" == output_queues.get()
                logger.info("all ranks are ready and listening, init done")

                all_group_result_list = output_queues.get()
                for group_size in all_group_result_list:
                    self.process_results(all_group_result_list[group_size], group_size)

                for process in _subprocesses.processes:
                    process.join()

            except Exception as e:
                logger.error(f"Create subprocesses failed, error msg: {e}")
                sys.exit(1)


    def compute_op_perf(self, rank: int, *args):
        target_devices, world_size, input_queues, output_queues = args
        backend_instance = self.backend_class(
            self.workload, 
            device_index=target_devices[rank]
        )
        output_queues.put("ready")
        op_name = self.workload["operator"]
        result_list = []
        while True:
            test_instance = input_queues.get()
            if test_instance is None:
                break           

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
                logger.error(f"Execute op: {op_name.lower()} failed, input_shape: {test_shape}, dtype: {test_dtype}, error msg: {e}")
                reports = {}

            if reports and "Error" not in reports:
                result_list.append(ResultItem(test_instance, reports))

                latency = reports.get("Avg latency(us)", 0)
                kernel_bw = reports.get("Kernel bandwidth(GB/s)", 0)
                bus_bw = reports.get("Bus bandwidth(GB/s)", 0)

                print(f"{op_name}, rank {rank}, device {target_devices[rank]}, {test_instance}, latency: {latency}\nkernel_bw: {kernel_bw}, bus_bw: {bus_bw}")
            else:
                print(f"{op_name}, rank {rank}, device {target_devices[rank]}, {test_instance}, error")

        output_queues.put(result_list)




    def communication_op_perf(self, rank: int, *args):
        ori_target_devices, target_devices, \
        world_size, \
        numa_world_size, numa_rank, \
        group_size_list, test_list, \
        input_queues, output_queues = args

        
        current_device = target_devices[rank]
        world_size = len(target_devices) * numa_world_size

        backend_instance = self.backend_class(
            self.workload, 
            device_index=current_device, 
            world_size=world_size, 
            rank=current_device
        )
        output_queues.put("ready")
        op_name = self.workload["operator"]


        dist_module = backend_instance.get_dist_module()
        all_group_result_list = {}
        for group_size in group_size_list:
            result_list = []

            if group_size > 1:
                new_group = backend_instance.new_group(range(group_size))
                if current_device < group_size:
                    backend_instance.op_group = new_group
                    backend_instance.op_group_size = group_size
                    
                    test_tensor = torch.ones([1], device="cuda")
                    dist_module.all_reduce(test_tensor, group=new_group)
                    print(f"allreduce in group size {group_size}, {test_tensor}\n", end="", flush=True)


            if current_device < group_size:
                for test_instance in test_list:
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
                        logger.error(f"Execute op: {op_name.lower()} failed, input_shape: {test_shape}, dtype: {test_dtype}, error msg: {e}")
                        reports = {}

                    reports_list = [None for i in range(group_size)]
                    if group_size == 1:
                        reports_list[0] = reports
                    else:
                        dist_module.all_gather_object(reports_list, reports, group=new_group)
                    
                    if current_device == 0 and reports_list[0] and "Error" not in reports_list[0]:
                        latency_list = [reports.get("Avg latency(us)", 0) for reports in reports_list]
                        kernel_bw_list = [reports.get("Kernel bandwidth(GB/s)", 0) for reports in reports_list]
                        bus_bw_list = [reports.get("Bus bandwidth(GB/s)", 0) for reports in reports_list]
                        print(f"{op_name}, device {ori_target_devices[0:group_size]}, {test_instance}")
                        print(f"latency: {latency_list}")
                        print(f"kernel_bw: {kernel_bw_list}")
                        print(f"bus_bw: {bus_bw_list}")
                        print("")

                        reports["device_list"] = ori_target_devices[0:group_size]
                        reports["latency_list"] = latency_list
                        reports["kernel_bw_list"] = kernel_bw_list
                        reports["bus_bw_list"] = bus_bw_list
                        if op_name in ["host2device", "device2host"]:
                            reports["Kernel bandwidth(GB/s)"] = sum(kernel_bw_list)
                            reports["Bus bandwidth(GB/s)"] = sum(bus_bw_list)
                        result_list.append(ResultItem(test_instance, reports))

            if group_size > 1:
                dist_module.destroy_process_group(new_group)
                if current_device < group_size:
                    backend_instance.op_group = None
                    backend_instance.op_group = 1

            dist_module.barrier()
            all_group_result_list[group_size] = result_list

        if rank == 0:
            output_queues.put(all_group_result_list)


if __name__ == "__main__":
    engine = PerfEngine()
    engine.start_engine()
