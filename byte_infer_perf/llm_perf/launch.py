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
import argparse
import subprocess
import json
import multiprocessing as mp
import signal
from typing import Any, Dict, Iterable, List


# ${prj_root}/
BYTE_MLPERF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)

from llm_perf.benchmark.bench import benchmark
from llm_perf.utils.logger import logger, setup_logger
from llm_perf.utils.reporter import Reporter, ReportType


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="chatglm2-torch-fp16-6b",
        help="The task going to be evaluted, refs to workloads/",
    )
    parser.add_argument(
        "--hardware_type",
        default="GPU",
        help="The backend going to be evaluted, refs to backends/",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for the gRPC server"
    )
    parser.add_argument("--port", default="50052", help="port of the server")

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


    def start_engine(self) -> None:
        """
        Byte MlPerf will create an virtual env for each backend to avoid dependance conflict
        """
        loglevel = os.environ.get("LOG_LEVEL", "info")
        setup_logger(loglevel)

        workload = load_workload(self.task)

        test_perf = bool(workload["test_perf"])
        test_accuracy = bool(workload["test_accuracy"])

        if (not os.path.exists("llm_perf/model_zoo/sota/" + workload["model"])) or (
            not os.path.exists("llm_perf/reports/GPU/" + workload["model"])
        ):
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

        tp_size = workload["tp_sizes"][0] if test_perf else 1
        batch_size = workload["batch_sizes"][0] if test_perf else 1
        input_tokens = workload["input_tokens"][0] if test_perf else 1

        # 0. Start Reporter
        self.reporter = Reporter(
            task=self.task,
            backend=self.backend_type,
            tp_size=tp_size,
            batch_size=batch_size,
            input_tokens=input_tokens,
            min_new_tokens=workload["min_new_tokens"],
            max_new_tokens=workload["max_new_tokens"],
            test_perf=test_perf,
            test_accuracy=test_accuracy,
        )
        self.reporter.start()

        # 1. Accuracy Test: default batch_size & tp_size are both 1
        if test_accuracy:
            self.run_perf(
                workload, 
                1, 
                1, 
                1, 
                ReportType.ACCURACY
            )

        # 2. Performance Test
        if test_perf:
            for tp_size in workload["tp_sizes"]:
                for batch_size in workload["batch_sizes"]:
                    for input_tokens in workload["input_tokens"]:
                        self.run_perf(
                            workload,
                            tp_size,
                            batch_size,
                            input_tokens,
                            ReportType.PERFORMANCE,
                        )

        self.reporter.stop()
        self.reporter.summary()



    def run_perf(
        self,
        workload: Dict[str, Any],
        tp_size: int,
        batch_size: int,
        input_tokens: int,
        report_type: ReportType,
    ) -> None:
        # 1. Start server
        self.start_server(tp_size, batch_size)

        # 2. Benchmark clients
        self.start_benchmark(workload, batch_size, input_tokens, report_type)

        # 3. Get result
        alive_clients = batch_size if report_type == ReportType.PERFORMANCE else 1
        started: bool = False
        while alive_clients:
            result = self.result_queue.get()
            if isinstance(result, str) and result == "@start":
                if not started:
                    # Reset reporter mate information
                    self.reporter.update_meta(tp_size, batch_size, input_tokens)
                started = True
                continue
            elif result is None:
                alive_clients = alive_clients - 1
                continue
            self.reporter.submit(result, report_type)

        # 4. Join benchmark client process
        for p in self.jobs:
            p.join()

        # 5. Kill server process
        self.stop_server()




    def start_server(self, tp_size: int, batch_size: int):
        fifo_name = "./server_fifo"
        try:
            os.mkfifo(fifo_name)
        except FileExistsError:
            logger.debug(f"{fifo_name} already exist")
        command = [
            "torchrun",
            "--master_port", "19999",
            "--nproc-per-node", str(tp_size),
            "llm_perf/server/launch_server.py",
            "--task", self.task,
            "--hardware_type", self.backend_type,
            "--port", self.port,
            "--max_batch_size", str(batch_size),
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


    # launch clients threads
    def start_benchmark(
        self,
        workload: Dict[str, Any],
        batch_size: int,
        input_tokens: int,
        report_type: ReportType,
    ): 
        clients = 1 if report_type == ReportType.ACCURACY else batch_size
        for i in range(clients):
            p = mp.Process(
                target=benchmark,
                args=(
                    i,
                    workload,
                    report_type,
                    input_tokens,
                    self.result_queue,
                    self.args,
                ),
            )
            self.jobs.append(p)
            p.start()

if __name__ == "__main__":
    instance = PerfEngine()
    instance.start_engine()
