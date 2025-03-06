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
import csv
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
import itertools
import prettytable
import jsonlines

from typing import Any, Dict, List
import itertools
from collections import namedtuple


FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger, setup_logger
from core.creators import create_backend
from core.scheduler import Scheduler

                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware_type", type=str)
    parser.add_argument("--task_dir", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--numa_world_size", type=int, default=1)
    parser.add_argument("--numa_rank", type=int, default=0)
    parser.add_argument("--disable_parallel", action="store_true")
    parser.add_argument("--report_dir", type=str)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logger(args.log_level)
    return args


def parse_task(task_dir, task):
    task_dir = pathlib.Path(task_dir).absolute()
    task_cases = []
    # 如果存在 task.json
    json_path = task_dir.joinpath(task + ".json")
    if json_path.exists():
        with open(json_path, "r") as f:
            json_data = json.load(f)
        json_data_list = json_data.get("cases", [])
        if json_data_list:
            for argument_case in json_data_list:
                keys = list(argument_case.keys())
                values = list(argument_case.values())
                for i, value in enumerate(values):
                    if isinstance(value, str):
                        values[i] = [value]                
                total_cases = list(itertools.product(*values))
                for case in total_cases:
                    case_dict = dict(zip(keys, case))

                    added_key_value = {}
                    removed_key = []
                    for key in case_dict:
                        if "." in key:
                            split_keys = key.split(".")
                            for i, split_key in enumerate(split_keys):
                                added_key_value[split_key] = case_dict[key][i]
                            removed_key.append(key)
                    for key in removed_key:
                        del case_dict[key]
                    case_dict.update(added_key_value)
                    task_cases.append(case_dict)
    return task_cases



if __name__ == "__main__":
    args = parse_args()
    task_cases = parse_task(args.task_dir, args.task)
    scheduler = Scheduler(args)
    result_list = scheduler.run(task_cases)


    if args.numa_rank == 0:
        if len(result_list) == 0:
            logger.error("No result found")
            sys.exit(1)

        sku_name = result_list[0]["sku_name"]

        report_dir = pathlib.Path(args.report_dir).absolute()
        backend_dir = report_dir.joinpath(args.hardware_type)
        sku_dir = backend_dir.joinpath(sku_name)
        op_dir = sku_dir.joinpath(args.task)
        op_dir.mkdir(parents=True, exist_ok=True)


        result_mapping = {}
        for result in result_list:
            arguments = result["arguments"]
            targets = result["targets"]

            args_type = arguments.get("args_type", "default")
            world_size = arguments.get("world_size", 1)
            dtype = arguments["dtype"]

            key = (args_type, world_size, dtype)
            if key not in result_mapping:
                result_mapping[key] = []
            result_mapping[key].append(result)

        for key in result_mapping:
            file_name = f"{key[0]}-group{key[1]}-{key[2]}"
            result_list = result_mapping[key]

            jsonl_file_path = op_dir.joinpath(file_name + ".jsonl")
            with jsonlines.open(jsonl_file_path, "w") as writer:
                for result in result_list:
                    writer.write(result)

            keys = ["task_name"]
            keys.extend(list(result_list[0]["arguments"].keys()))
            keys.extend(list(result_list[0]["targets"].keys()))

            csv_file_path = op_dir.joinpath(file_name + ".csv")
            with open(csv_file_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for result in result_list:
                    row = [args.task]
                    row.extend(list(result["arguments"].values()))
                    row.extend(list(result["targets"].values()))
                    writer.writerow(row)

        