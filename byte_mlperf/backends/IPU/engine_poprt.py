# Copyright 2023 ByteDance and/or its affiliates.
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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
import threading as th
import time
from queue import Queue

import numpy as np
import torch
from poprt import runtime

from . import engine

log = logging.getLogger("engine_poprt")


class PopRT(engine.Engine):
    def __init__(self, popef_path):
        self.model_runner = runtime.ModelRunner(popef_path)

    def predict(self, feeds):
        input_descriptions = self.model_runner.get_model_inputs()
        for desc in input_descriptions:
            if isinstance(feeds[desc.name], list):
                feeds[desc.name] = np.array(
                    feeds[desc.name], dtype=desc.numpy_data_type()
                )
            elif isinstance(feeds[desc.name], np.ndarray):
                feeds[desc.name] = feeds[desc.name].astype(desc.numpy_data_type())
            elif isinstance(feeds[desc.name], torch.Tensor):
                feeds[desc.name] = (
                    feeds[desc.name].numpy().astype(desc.numpy_data_type())
                )
            else:
                raise TypeError("The feeds[value] must be list or np.ndarray")

        # create the output numpy arrays
        output_descriptions = self.model_runner.get_model_outputs()
        results = {}
        for output_desc in output_descriptions:
            results[output_desc.name] = np.zeros(
                output_desc.shape, dtype=output_desc.numpy_data_type()
            )

        self.model_runner.execute(feeds, results)
        return results

    def benchmark(self, clients, batch_size, iterations):
        input_view = runtime.InputMemoryView()
        input_descriptions = self.model_runner.get_model_inputs()
        output_descriptions = self.model_runner.get_model_outputs()
        inputs = {}
        outputs = {}
        for input_desc in input_descriptions:
            inputs[input_desc.name] = np.random.randn(*input_desc.shape).astype(
                input_desc.numpy_data_type()
            )
        for output_desc in output_descriptions:
            outputs[output_desc.name] = np.zeros(
                output_desc.shape, dtype=output_desc.numpy_data_type()
            )

        log.info("Warm up")
        for _ in range(5):
            self.model_runner.execute(inputs, outputs)
        log.info("Warm up completed, start the time counting")

        q = Queue()

        def client_fun(model_runner, iteration, input_view):
            durations = []
            for _ in range(iteration):
                start_time = time.time()
                self.model_runner.execute(inputs, outputs)
                end_time = time.time()
                durations.append((start_time, end_time))
            # remove the first and last 20
            if iteration > 40:
                durations = durations[20:-20]
            q.put(durations, timeout=10)

        thp = [
            th.Thread(
                target=client_fun, args=(self.model_runner, iterations, input_view)
            )
            for _ in range(clients)
        ]
        for t in thp:
            t.start()
        for t in thp:
            t.join()

        durations_from_th = []
        while not q.empty():
            durations_from_th += q.get()
        max_timestamp = max(y for _, y in durations_from_th)
        min_timestamp = min(x for x, _ in durations_from_th)
        if iterations > 40:
            iterations -= 40  # iterations -40 as line 260
        qps = clients * batch_size * iterations / (max_timestamp - min_timestamp)
        times_range = [y - x for x, y in durations_from_th]

        times_range.sort()
        tail_latency = round(times_range[int(len(times_range) * 0.99)] * 1000, 2)
        avg_latency = round(sum(times_range) / len(times_range) * 1000, 2)

        log.info(
            "Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}".format(
                batch_size, int(qps), avg_latency, tail_latency
            )
        )

        np_latency = np.array(times_range) * 1000.0
        log.info(
            f"====== Latency P50: {np.percentile(np_latency, 50)}, P90: {np.percentile(np_latency, 90)}, P99: {np.percentile(np_latency, 99)} ======"
        )

        return qps, avg_latency, tail_latency
