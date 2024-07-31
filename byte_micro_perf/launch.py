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
import json
import logging
import os
import random
import socket
import subprocess
import sys

BYTE_MLPERF_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lanuch")


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
    parser.add_argument(
        "--task", default="", help="The task going to be evaluted, refs to workloads/"
    )
    parser.add_argument(
        "--task_dir", default="", help="The direcotry of tasks going to be evaluted, e.g., set to workloads"
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
        "--compile_only",
        action="store_true",
        help="Task will stoped after compilation finished",
    )
    parser.add_argument(
        "--show_task_list", action="store_true", help="Print all task names"
    )
    parser.add_argument(
        "--show_hardware_list",
        action="store_true",
        help="Print all hardware bytemlperf supported",
    )
    args = parser.parse_args()

    if args.show_task_list:
        logging.info("******************* Supported Task *******************")
        for file in os.listdir("workloads"):
            print(file[:-5])
    if args.show_hardware_list:
        log.info("***************** Supported Hardware Backend *****************")
        for file in os.listdir("backends"):
            if not file.endswith(".py") and not file.startswith("_"):
                print(file)
    if args.task or args.task_dir:
        log.info("******************* Pip Package Installing *******************")
        subprocess.call(
            ["python3", "-m", "pip", "install", "pip", "--upgrade", "--quiet"]
        )

        subprocess.call(
            ["python3", "-m", "pip", "install", "-r", "requirements.txt", "--quiet"]
        )

        if args.task:
            if args.task_dir:
                log.warning("task and task_dir are both set, task_dir will be ignored")
            tasks = args.task.split(',')
        elif args.task_dir:
            tasks = parse_task(args.task_dir)
        logging.info(f"******************* Tasks: {tasks}")
        exit_code = 0
        for task in tasks:
            cmd = "python3 core/perf_engine.py --hardware_type {} --task {} --vendor_path {}".format(
                args.hardware_type, task, args.vendor_path
            )
            exit_code = subprocess.call(cmd, shell=True)

        sys.exit(exit_code)
