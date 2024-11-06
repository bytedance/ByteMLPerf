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

import logging
import os
import time

import numpy as np
import sophon.sail as sail
from general_perf.backends import runtime_backend

log = logging.getLogger("RuntimeBackendTPU")

class RuntimeBackendTPU(runtime_backend.RuntimeBackend):
    def __init__(self):
        super().__init__()
        self.hardware_type = "TPU"
        self.need_reload = False
        self.model_runtimes = []
        self.configs = None
        self.pack_config = None
        self.batch_size = -1
        self.pack_bs = -1
        self.packrunner = False
        self.engine = None
        self.runner_name = "SAIL"
        self.compiled_dir = (
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/compiled_models/"
        )
        self.precision = "fp32"
        
    def version(self) -> str:
        return sail.__version__
    
    def load(self, batch_size) -> None:
        log.warning("TPU Backend only support static batch_size now.")
        self.bmodel_path = self.compiled_dir + self.configs["model"] + ".bmodel"
        # self.input_key = self.configs["input_shape"][self.configs["inputs"]]
        self.net = sail.nn.Engine(self.bmodel_path, 1)
        self.net_name = self.net.get_net_names()[0]
        self.input_name = self.net.get_input_names(self.net_name)[0]
        self.output_names = self.net.get_output_names(self.net_name)
        self.input_shape = self.net.get_input_shapes(self.net_name, 0)[0]
        self.output_shapes = self.net.get_output_shapes(self.net_name, 0)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
    
    def get_loaded_batch_size(self) -> int:
        return self.batch_size
    
    def predict(self, data):
        if isinstance(data, dict):
            input_data = {0: next(iter(data.values()))}
        else:
            input_data = {0: data}

        output_arrays = [np.ndarray(shape=(self.output_shapes[i]), dtype=np.float32) for i in range(len(self.output_shapes))]
        outputs = {i:array for i, array in enumerate(output_arrays)}
        ret = self.net.process(input_data, outputs, self.net_name)
        return outputs
        
    def _run_benchmark(self, bs, iter):
        input = np.random.rand(*self.input_shape).astype(np.float32)
        durations = []
        
        for i in range(iter):
            log.info(f'Running Predict times {i}')
            start_time = time.time()
            _ = self.predict(input)
            end_time = time.time()
            durations.append(end_time - start_time)
        
        total_duration = np.sum(durations)
        avg_latency = total_duration / iter
        tail_latency = np.percentile(durations, 95)
    
        qps = iter / total_duration
        
        return qps, avg_latency, tail_latency
    
    def benchmark(self, dataloader):
        report = {}
        report["BS"] = self.batch_size
        interact_info = self.configs.get("interact_info", {})
        iterations = self.workload["iterations"]

        qps, avg_latency, tail_latency = self._run_benchmark(
            self.batch_size, iterations
        )

        report["QPS"] = int(qps)
        report["AVG Latency"] = avg_latency
        report["P99 Latency"] = tail_latency

        return report