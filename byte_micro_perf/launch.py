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
import signal
import pathlib
import logging
import argparse
import subprocess


# directory config
CUR_DIR = pathlib.Path.cwd().absolute()
FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR
sys.path.insert(0, str(BYTE_MLPERF_ROOT))

# logger config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lanuch")


def parse_task(task_dir):
    tasks = []
    if os.path.isdir(task_dir):
        for root, _, files in os.walk(task_dir, topdown=False):
            for name in files:
                if name.endswith(".json"):
                    tasks.append(name.rsplit('.', 1)[0])
    return tasks


def get_numa_configs():
    numa_configs = []
    numa_node_num = int(subprocess.check_output("lscpu | grep 'NUMA node(s)' | awk -F: '{print $2}'", shell=True).decode().strip())
    for i in range(numa_node_num):
        numa_cores = subprocess.check_output(f"lscpu | grep 'NUMA node{i}' | awk -F: '{{print $2}}'", shell=True).decode().strip()
        numa_configs.append(numa_cores)
    return numa_configs
    


if __name__ == "__main__":

    # get numa config, for example: 
    # 0: 0-31,64-95
    # 1: 32-63,96-127
    numa_configs = get_numa_configs()




    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware_type",
        default="GPU", 
        help="The backend going to be evaluted, refs to backends/",
    )
    parser.add_argument(
        "--task_dir", type=str, 
        default=str(BYTE_MLPERF_ROOT.joinpath("workloads").absolute()), 
        help="The direcotry of tasks going to be evaluted, e.g., set to workloads"
    )
    parser.add_argument(
        "--task", 
        default="all", 
        help="The task going to be evaluted, refs to workloads/, default use all tasks in workloads/"
    )
    parser.add_argument(
        "--show_task_list", 
        action="store_true", 
        help="Print all available task names"
    )
    parser.add_argument(
        "--show_hardware_list",
        action="store_true",
        help="Print all hardware bytemlperf supported",
    )
    parser.add_argument("--numa_balance", action="store_true", help=f"if enabled, process will launch num_numa_nodes={len(numa_configs)} processes")
    parser.add_argument("--device", type=str, default="all")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--report_dir", type=str, default=str(BYTE_MLPERF_ROOT.joinpath("reports").absolute()))
    args = parser.parse_args()


    args.task_dir = pathlib.Path(args.task_dir).absolute()
    args.report_dir = pathlib.Path(args.report_dir).absolute()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # report_log_path = args.report_dir.joinpath("_run_report.log")
    # with open(report_log_path, "w") as file:
    #     pass


    os.chdir(str(BYTE_MLPERF_ROOT))

    task_list = [task_json.stem for task_json in args.task_dir.glob("*.json")]
    task_list.sort()
    task_mapping = {
        "all": task_list, 
        "gemm_ops": [], 
        "unary_ops": [], 
        "binary_ops": [], 
        "reduction_ops": [], 
        "index_ops": [], 
        "h2d_ops": [], 
        "ccl_ops": []
    }
    for task in task_list:
        if task in ["gemm", "gemv", "batch_gemm", "group_gemm"]:
            task_mapping["gemm_ops"].append(task)

        if task in ["sin", "cos", "exp", "exponential", "log", "sqrt", "cast", "silu", "gelu", "swiglu"]:
            task_mapping["unary_ops"].append(task)

        if task in ["add", "mul", "sub", "div"]:
            task_mapping["binary_ops"].append(task)

        if task in ["layernorm", "softmax", "reduce_sum", "reduce_max", "reduce_min"]:
            task_mapping["reduction_ops"].append(task)
        
        if task in ["index_add", "sort", "unique", "gather", "scatter", "hash_table", "topk"]:
            task_mapping["index_ops"].append(task)

        if task in ["host2device", "device2host", "device2device"]:
            task_mapping["h2d_ops"].append(task)

        if task in ["allgather", "allreduce", "alltoall", "broadcast", "p2p", "reducescatter"]:
            task_mapping["ccl_ops"].append(task)

    hardware_list = []
    for file in BYTE_MLPERF_ROOT.joinpath("backends").iterdir():
        if file.is_dir() and file.stem.startswith("_") is False:
            hardware_list.append(file.stem)


    # show task and hardware list
    if args.show_task_list:
        logger.info("******************* Supported Task *******************")
        print(task_list)        
        exit(0)

    if args.show_hardware_list:
        logger.info("***************** Supported Hardware Backend *****************")
        print(hardware_list)
        exit(0)



    # check task
    test_cases = []
    if args.task in task_mapping.keys():
        test_cases = task_mapping[args.task]
    else:
        specified_tasks = args.task.split(",")
        for task in specified_tasks:
            if task not in task_list:
                logger.error(f"Task {task} not found in {args.task_dir}")
                exit(1)
            test_cases.append(task)

    logger.info(f"******************* Tasks: *****************")
    logger.info(f"{test_cases}\n")


    # check hardware
    hardware = args.hardware_type
    if hardware not in hardware_list:
        logger.error(f"Hardware {hardware} not found in {BYTE_MLPERF_ROOT.joinpath('backends')}")
        exit(1)

    logger.info(f"******************* hardware: *****************")
    logger.info(f"{hardware}\n")







    # terminate core task perf process
    subprocess_pids = []
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, exiting...")
        if len(subprocess_pids) > 0:
            logger.info(f"terminate subprocess: {subprocess_pids}")
            for pid in subprocess_pids:
                os.kill(pid, signal.SIGTERM)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)



        
    numa_world_size = len(numa_configs) if args.numa_balance else 1


    for task in test_cases:
        perf_cmd =  f"python3 ./core/perf_engine.py"\
                    f" --hardware_type {args.hardware_type}"\
                    f" --task_dir {args.task_dir}"\
                    f" --task {task}"\
                    f" --device {args.device}"\
                    f" --report_dir {args.report_dir}"
        if args.parallel:
            perf_cmd += " --parallel"

        print(f"******************************************* Start to test op: [{task}]. *******************************************")
        subprocess_cmds = []
        subprocess_instances = []
        subprocess_pids = []
        for i in range(numa_world_size):
            cmd = f"taskset -c {numa_configs[i]} {perf_cmd} --numa_world_size {numa_world_size} --numa_rank {i}"
            subprocess_cmds.append(cmd)
            print(f"{cmd}")
        for i in range(numa_world_size):            
            process_instance = subprocess.Popen(subprocess_cmds[i], shell=True)
            subprocess_instances.append(process_instance)
            process_pid = process_instance.pid
            subprocess_pids.append(process_pid)
        print(f"perf_subprocess_pids: {subprocess_pids}")
        print("")

        for process in subprocess_instances:
            process.wait()
        print("")