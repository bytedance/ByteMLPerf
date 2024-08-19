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
import pathlib
import json
import logging
import os
import random
import socket
import subprocess
import sys

BYTE_MLPERF_ROOT = pathlib.Path(__file__).parent.absolute()
CUR_DIR = pathlib.Path.cwd().absolute()
os.chdir(str(BYTE_MLPERF_ROOT))
sys.path.insert(0, BYTE_MLPERF_ROOT)

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hardware config
    parser.add_argument(
        "--hardware_type",
        default="GPU", 
        help="The backend going to be evaluted, refs to backends/",
    )
    parser.add_argument(
        "--vendor_path",
        required=False, 
        help="The hardware configs need to be loaded, refs to vendor_zoo/",
    )

    # task config
    parser.add_argument(
        "--task_dir", 
        default=str(BYTE_MLPERF_ROOT.joinpath("workloads").absolute()), 
        help="The direcotry of tasks going to be evaluted, e.g., set to workloads"
    )
    parser.add_argument(
        "--task", 
        default="all", 
        help="The task going to be evaluted, refs to workloads/, default use all tasks in workloads/"
    )

    # list all supported task and hardware
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
    args = parser.parse_args()


    # show tasks
    task_list = [file.stem for file in pathlib.Path(args.task_dir).iterdir()]
    task_list.sort()
    if args.show_task_list:
        logger.info("******************* Supported Task *******************")
        print(task_list)        
        exit(0)

    # show hardwares
    hardware_list = []
    for file in BYTE_MLPERF_ROOT.joinpath("backends").iterdir():
        if file.is_dir() and file.stem.startswith("_") is False:
            hardware_list.append(file.stem)
    if args.show_hardware_list:
        logger.info("***************** Supported Hardware Backend *****************")
        print(hardware_list)
        exit(0)

    # check task
    tasks = task_list if args.task == "all" else args.task.split(",")
    for task in tasks:
        if task not in task_list:
            logger.error(f"Task {task} not found in {args.task_dir}")
            exit(1)

    logger.info(f"******************* Tasks: *****************")
    logger.info(f"{tasks}\n")

    # check hardware
    hardware = args.hardware_type
    if hardware not in hardware_list:
        logger.error(f"Hardware {hardware} not found in {BYTE_MLPERF_ROOT.joinpath('backends')}")
        exit(1)

    logger.info(f"******************* hardware: *****************")
    logger.info(f"{hardware}\n")


    logger.info("******************* Pip Package Installing *******************")
    subprocess.run(
        ["python3", "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
        capture_output=True
    )
    subprocess.run(
        ["python3", "-m", "pip", "install", "-r", "requirements.txt", "--quiet"], 
        capture_output=True
    )

    for task in tasks:
        cmd = "python3 core/perf_engine.py --hardware_type {} --task {} --vendor_path {} --task_dir {}".format(
            args.hardware_type, task, args.vendor_path, args.task_dir
        )
        exit_code = subprocess.call(cmd, shell=True)

    sys.exit(exit_code)
