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
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List

import pandas as pd

BYTE_MLPERF_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)


from llm_perf.benchmark.bench import bench_accuracy, bench_performance
from llm_perf.server.serve import serve
from llm_perf.utils.logger import logger, setup_logger
from llm_perf.utils.reporter import Reporter, ReportType


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="chatglm-torch-fp32",
        help="The task going to be evaluted, refs to workloads/",
    )
    parser.add_argument(
        "--hardware_type",
        default="GPU",
        help="The backend going to be evaluted, refs to backends/",
    )
    parser.add_argument("--port", default="50052", help="port of the server")
    parser.add_argument(
        "--compile_only", action="store_true", help="Run compilation only"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k for sampling")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for the gRPC server"
    )

    args = parser.parse_args()
    return args


def get_model_info(model_name: str) -> Dict[str, Any]:
    with open("llm_perf/model_zoo/" + model_name + ".json", "r") as file:
        model_info = json.load(file)
    return model_info


def load_workload(task: str) -> Dict[str, Any]:
    """
    Return a list of dictionary with model Configuration

    Args: List[str]

    Returns: List[dic]
    """
    # modules_dir = os.path.dirname(os.path.dirname(__file__)) + '/workloads'
    modules_dir = "llm_perf/workloads"

    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".json") or os.path.isdir(path))
            and file[: file.find(".json")] == task
        ):
            module_name = file
            with open("llm_perf/workloads/" + module_name, "r") as f:
                workload_dict = json.load(f)
            return workload_dict
    else:
        logger.error(f"Task name: {task} was not found, please check your task name")


class PerfEngine:
    def __init__(self) -> None:
        super().__init__()
        self.args = get_args()
        self.backend_type = self.args.hardware_type
        self.port = self.args.port
        self.task = self.args.task
        self.result_queue = mp.Queue()
        self.jobs: List[mp.Process] = []
        self.server_process = None

    def __del__(self):
        self.stop_server()

    def start_server(self, tp_size: int, batch_size: int):
        fifo_name = "./server_fifo"
        try:
            os.mkfifo(fifo_name)
        except FileExistsError:
            logger.debug(f"{fifo_name} already exist")
        command = [
            "torchrun",
            "--master_port",
            "19999",
            "--nproc-per-node",
            str(tp_size),
            "llm_perf/launch.py",
            "--task",
            self.task,
            "--hardware_type",
            self.backend_type,
            "--port",
            self.port,
            "--max_batch_size",
            str(batch_size),
        ]
        logger.info(f"Start Server: {' '.join(command)}")

        # Use preexec_fn=os.setsid to make sure all subprocess in same process group (easy to kill process)
        self.server_process = subprocess.Popen(command, preexec_fn=os.setsid)

        with open(fifo_name, "r") as fifo_fd:
            while True:
                data = fifo_fd.readline().strip()
                if data == "Server Ready":
                    break
        os.remove(fifo_name)
        logger.info("Server Ready")

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            logger.info("stopping server process")
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            try:
                self.server_process.wait(timeout=5)
                logger.info("server process has stopped")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                logger.info("server process force killing")
        else:
            # logger already exit
            print(f"server process already exit with {self.server_process.poll()}")

    def start_benchmark(self, workload, model_config, report_type: ReportType):
        if report_type == ReportType.ACCURACY:
            clients = 1
            bench = bench_accuracy
        elif report_type == ReportType.PERFORMANCE:
            clients = workload["clients"]
            bench = bench_performance
        for i in range(clients):
            p = mp.Process(
                target=bench,
                args=(
                    self.backend_type,
                    workload,
                    model_config,
                    self.result_queue,
                    self.args,
                ),
            )
            self.jobs.append(p)
            p.start()

    def run_perf(
        self,
        workload: Dict[str, Any],
        model_config: Dict[str, Any],
        tp_size: int,
        batch_size: int,
        report_type: ReportType,
    ) -> None:
        self.reporter.update_meta(tp_size=tp_size, batch_size=batch_size)
        # 1. Start Server
        self.start_server(tp_size, batch_size)

        # 2. Benchmark
        self.start_benchmark(workload, model_config, report_type)

        # 3. Get Result
        alive_clients = (
            workload["clients"] if report_type == ReportType.PERFORMANCE else 1
        )
        while alive_clients:
            result = self.result_queue.get()
            if result is None:
                alive_clients = alive_clients - 1
                continue
            self.reporter.submit(result, report_type)

        # 4. Join benchmark client process
        for p in self.jobs:
            p.join()

        # 5. Kill server process
        self.stop_server()

    def start_engine(self) -> None:
        """
        Byte MlPerf will create an virtual env for each backend to avoid dependance conflict
        """
        loglevel = os.environ.get("LOG_LEVEL", "debug")
        setup_logger(loglevel)

        workload = load_workload(self.task)
        model_config = get_model_info(workload["model"])

        test_perf = bool(workload["test_perf"])
        test_accuracy = bool(workload["test_accuracy"])

        if (not os.path.exists("llm_perf/model_zoo/sota/" + workload["model"])) or (not os.path.exists("llm_perf/reports/GPU/" + workload["model"])):
            subprocess.call(
                [
                    "bash",
                    "llm_perf/prepare_model.sh",
                    workload["model"],
                    str(test_accuracy),
                ]
            )

        if not any([test_perf, test_accuracy]):
            logger.info(f"End of the llm_perf, enable at least one test item")
            return

        default_batch_size = workload["batch_sizes"][0] if test_perf else 1
        default_tp_size = workload["tp_sizes"][0] if test_perf else 1
        # 0. Start Reporter
        self.reporter = Reporter(
            task=self.task,
            backend=self.backend_type,
            tp_size=default_tp_size,
            batch_size=default_batch_size,
            min_new_tokens=workload["min_new_tokens"],
            max_new_tokens=workload["max_new_tokens"],
            test_perf=test_perf,
            test_accuracy=test_accuracy,
        )
        self.reporter.start()

        # 1. Accuracy Test: default batch_size & tp_size are both 1
        if test_accuracy:
            self.run_perf(workload, model_config, 1, 1, ReportType.ACCURACY)

        # 2. Performance Test
        if test_perf:
            for tp_size in workload["tp_sizes"]:
                for batch_size in workload["batch_sizes"]:
                    self.run_perf(
                        workload,
                        model_config,
                        tp_size,
                        batch_size,
                        ReportType.PERFORMANCE,
                    )

        self.reporter.stop()
        self.reporter.summary()


if __name__ == "__main__":
    instance = PerfEngine()
    instance.start_engine()
