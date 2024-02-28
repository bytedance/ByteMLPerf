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
import os
import logging
import importlib
import subprocess
import sys
import virtualenv
from typing import Any, List, Dict

import torch
import torch.multiprocessing as mp

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
        help="The task going to be evaluted, refs to workloads/")
    parser.add_argument(
        "--hardware_type",
        default="GPU",
        help="The backend going to be evaluted, refs to backends/")
    parser.add_argument(
        "--vendor_path",
        default="",
        help="The hardware configs need to be loaded, refs to vendor_zoo/NVIDIA")      
    parser.add_argument("--compile_only",
                        action='store_true',
                        help="Run compilation only")

    args = parser.parse_args()
    return args

def load_workload(task: str) -> Dict[str, Any]:
    """
    Return a list of dictionary with model Configuration
    Args: List[str]
    Returns: List[dic]
    """
    modules_dir = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + '/workloads'

    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)
        if (not file.startswith('_') and not file.startswith('.')
                and (file.endswith('.json') or os.path.isdir(path))
                and file[:file.find('.json')] == task):
            module_name = file
            with open("workloads/" + module_name, 'r') as f:
                workload_dict = json.load(f)
            return workload_dict
    else:
        log.error(
            "Task name: [ {} ] was not found, please check your task name".
            format(task))
        
class PerfEngine:
    def __init__(self) -> None:
        super().__init__()
        self.args = get_args()
        self.workload = load_workload(self.args.task)
        self.backend_type = self.args.hardware_type
        self.old_os_path = os.environ['PATH']
        self.prev_sys_path = list(sys.path)
        self.real_prefix = sys.prefix

    def init_process(self, rank: int, world_size: int):
        """ 
        Initialize the distributed environment. 

        """
        initialize_func = getattr(self.backend, "initialize_ccl")
        initialize_func(rank, world_size)
        status = self.start_perf(self.workload)   

    def init_backend(self, hardware_type: str) -> Backend:
        """
        Load related compile backend with input hardware type

        Arguments: str

        Returns: CompileBackend()
        """
        log.info("Loading Heterogeneous Backend: {}".format(hardware_type))

        backend = importlib.import_module('backends.' +
                                                hardware_type +
                                                ".backend_" +
                                                hardware_type.lower())
        backend = getattr(backend,
                                "Backend" + hardware_type)
        return backend(self.workload, self.args.vendor_path)

    def start_engine(self) -> None:
        status = self.activate_venv(self.backend_type)
        if not status:
            log.warning("Activate virtualenv Failed, Please Check...")

        self.backend = self.init_backend(self.backend_type)
        output_dir = os.path.abspath('reports/' +
                                     self.backend_type)
        os.makedirs(output_dir, exist_ok=True)

        if self.args.task in ["allreduce", "allgather", "reducescatter", "alltoall"]:
            for group in self.workload['group']:
                mp.spawn(
                    fn = self.init_process,
                    args=(group,),
                    nprocs=group
                )
        else:
            status = self.start_perf(self.workload)

        self.deactivate_venv()

    def start_perf(
            self, workload: Dict[str, Any]) -> bool:
        log.info("******************************************* Start to test op: {}. *******************************************".format(workload['operator']))

        # Initalize Output Dir and Reports
        output_dir = os.path.abspath('reports/' +
                                     self.backend_type + '/' +
                                     workload['operator'])
        os.makedirs(output_dir, exist_ok=True)

        op_name = workload['operator']
        base_report = {
            "Operator": op_name.upper(),
            "Backend": self.backend_type,
            "Host Info": self.get_cpu_name(),
            "Device Info": getattr(self.backend, "get_device_name")()
        }

        op = getattr(self.backend, op_name.lower(), None)
        if op is not None and callable(op):
            op()
        else:
            raise ValueError(f"Unknown operation: {op_name.lower()}")

        perf_reports = []
        if "input_shape_list" in self.workload:
            shape_list = self.workload['input_shape_list'] 
        else:
            shape_list = []
            for M, N, K in self.workload['M/N/K']:
                shape_list.append([[M,N], [N,K]])
        
        for dtype in self.workload['dtype']:
            perf_reports = []
            base_report['Performance'] = {}
            for input_shape in shape_list:
                if isinstance(input_shape[0], int):
                    input_shape = [input_shape]        
                reports = self.backend.perf(input_shape, dtype)
                perf_reports.append(reports)
            base_report['Performance'] = perf_reports
            print(base_report)
            # write output to json file
            if 'Group' in base_report['Performance'][0]:
                output_report_path = output_dir + "/result-" + str(dtype) + \
                                "-{}".format(base_report['Performance'][0]['Group']) + ".json"
            else:
                output_report_path = output_dir + "/result-" + str(dtype) + ".json"                    
            with open(output_report_path, 'w') as file:
                json.dump(base_report, file, indent=4)
    
        return True

    def get_cpu_name(self):
        command = "lscpu | grep 'Model name' | awk -F: '{print $2}'"
        cpu_name = subprocess.check_output(command, shell=True)
        return cpu_name.decode().strip()

    def activate_venv(self, hardware_type: str) -> bool:
        if os.path.exists('backends/' + hardware_type +
                          '/requirements.txt'):
            log.info("Activating Virtual Env for " + hardware_type)

            venv_dir = os.path.join("backends",
                                    hardware_type + "/venv")
            activate_file = os.path.join(venv_dir, 'bin', 'activate_this.py')
            if not os.path.exists(venv_dir):
                log.info("venv not exist, Creating Virtual Env for " +
                         hardware_type)

                virtualenv.create_environment(venv_dir, True)

                exec(open(activate_file).read(), {'__file__': activate_file})
                python_path = os.path.join(venv_dir, 'bin', 'python3')
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'
                ])
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '-r', 'backends/' +
                    hardware_type + '/requirements.txt', '-q'
                ])
            else:
                exec(open(activate_file).read(), {'__file__': activate_file})
                '''
                just in case install failed in pre-run.
                '''
                python_path = os.path.join(venv_dir, 'bin', 'python3')
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'
                ])      
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '-r', 'backends/' +
                    hardware_type + '/requirements.txt', '-q'
                ])

                if not hasattr(sys, 'real_prefix'):
                    return False
                return True
        return True

    def deactivate_venv(self):
        sys.path[:0] = self.prev_sys_path  #will also revert the added site-packages
        sys.prefix = self.real_prefix
        os.environ['PATH'] = self.old_os_path

if __name__ == "__main__":
    engine = PerfEngine()
    engine.start_engine()
