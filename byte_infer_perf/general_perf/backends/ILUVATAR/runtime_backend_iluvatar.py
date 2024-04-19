# Copyright 2023 Graphcore Ltd.
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
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
import threading

from general_perf.backends import runtime_backend
from general_perf.backends.ILUVATAR.common import init_by_tensorrt, setup_io_bindings
from general_perf.backends.ILUVATAR.common import Task, TaskThread, _cudaGetErrorEnum, checkCudaErrors
from tensorrt import Dims
from cuda import cuda, cudart

from general_perf.backends.ILUVATAR.common import load_ixrt_plugin
load_ixrt_plugin()

log = logging.getLogger("RuntimeBackendILUVATAR")

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "LONG": torch.long,
    "INT64": torch.int64,
    "BOOL": torch.bool
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool
}

class RuntimeBackendILUVATAR(runtime_backend.RuntimeBackend):
    def __init__(self):
        super(RuntimeBackendILUVATAR, self).__init__()
        self.hardware_type = "ILUVATAR"
        self.need_reload = False
        self.model_runtimes = []
        self.configs = None
        self.engine = None
        self.context = None
        self.batch_size = -1
        self.workload = None

    # Dual-core inference of Tian SoC BI-150 graphics card
    def benchmark(self, dataloader):
        performance_reports = []
        merged_dict = {}
        
        workers = []
        lock = threading.Lock()
        for i in range(2):
            device_id = i
            task = Task(self.batch_size, dataloader, device_id, self.load, self.benchmark_interact, performance_reports, lock)

            work = TaskThread(task.run, [])
            workers.append(work)
            work.start()
            work.join()
            
        del self.engine
        del self.context

        if len(performance_reports[0]) == len(performance_reports[1]):
            if performance_reports[0].keys() == performance_reports[1].keys():

                qps = performance_reports[0]['QPS'] + performance_reports[1]['QPS']
                avg_latency = round(((performance_reports[0]['AVG Latency'] + performance_reports[1]['AVG Latency']) / 2.0), 2)
                p99_latency = round(((performance_reports[0]['P99 Latency'] + performance_reports[1]['P99 Latency']) / 2.0), 2)

                merged_dict['BS'] = performance_reports[0]['BS']
                merged_dict['QPS'] = qps
                merged_dict['AVG Latency'] = avg_latency
                merged_dict["P99 Latency"] = p99_latency

        return merged_dict  

    def predict(self, feeds):
        # The deberta model is currently unable to undergo accuracy testing temporarily
        input_tensors = []
        i = 0

        model_name = self.configs["model"].split("-")[0]
        if model_name == 'deberta':
            keys = list(feeds.keys())
            input_ids = torch.tensor(feeds[keys[0]], dtype=pt_dtype_map[self.input_type[0]])
            attention_mask = torch.tensor(feeds[keys[1]], dtype=pt_dtype_map[self.input_type[1]])
            input_tensors = [input_ids, attention_mask]

        else:
            for key, _ in feeds.items():
                tmp_tensor = torch.tensor(feeds[key],
                                    dtype=pt_dtype_map[self.input_type[i]])
                input_tensors.append(tmp_tensor)
                i += 1

        # ixrt inference
        engine = self.engine
        assert engine
        context = self.context
        assert context

        # set dynamic shape
        input_tensor_map = self.configs["segments"][0]["input_tensor_map"]
        input_shape = input_tensor_map.values()

        i = 0
        for input_name, _ in input_tensor_map.items():
            if model_name == 'widedeep':
                input_tensors.append(np.zeros((self.batch_size, 1), dtype=np.float32))
                input_names = [
                    "new_categorical_placeholder:0",
                    "new_numeric_placeholder:0",
                    "import/head/predictions/zeros_like:0"
                ]
                for input_name in input_names:
                    if input_name == 'new_categorical_placeholder:0':
                        input_shape = input_tensors[0].shape
                    if input_name == 'new_numeric_placeholder:0':
                        input_shape = input_tensors[1].shape
                    if input_name == 'import/head/predictions/zeros_like:0':
                        input_shape = input_tensors[2].shape
                
                    input_idx = engine.get_binding_index(input_name)
                    context.set_binding_shape(input_idx, Dims(input_shape))
            else:
                input_shape = input_tensors[i].shape
                input_idx = engine.get_binding_index(input_name)
                context.set_binding_shape(input_idx, Dims(input_shape))
                i += 1
        
        # Setup I/O bindings
        inputs, outputs, allocations = setup_io_bindings(engine, context)

        # Prepare the output data
        outputs_list = []
        for i in range(len(outputs)):
            output = np.zeros(outputs[i]["shape"], outputs[i]["dtype"])
            outputs_list.append(output)

        data_batch_list = []
        for i in range(len(input_tensors)):
            data_batch = np.ascontiguousarray(input_tensors[i])
            data_batch_list.append(data_batch)

        # H2D: host to device
        for i in range(len(inputs)):
            (err, ) = cudart.cudaMemcpy(
                        inputs[i]["allocation"],
                        data_batch_list[i],
                        inputs[i]["nbytes"],
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )

        context.execute_v2(allocations)
        
        # D2H: device to host
        for i in range(len(outputs)):
            (err, )= cudart.cudaMemcpy(outputs_list[i], 
                        outputs[i]["allocation"], 
                        outputs[i]["nbytes"], 
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            )
           
        # Free Gpu Memory
        # cuda-python
        for i in range(len(inputs)):
            err, = cudart.cudaFree(inputs[i]["allocation"])
            assert err == cudart.cudaError_t.cudaSuccess

        for i in range(len(outputs)):
            err, = cudart.cudaFree(outputs[i]["allocation"])
            assert err == cudart.cudaError_t.cudaSuccess
        
        result = {}

        output_tensor_map = self.configs["segments"][0]["output_tensor_map"]
        output_name = output_tensor_map.split(",")

        for i in range(len(output_name)):
            result[output_name[i]] = outputs_list[i]
        
        if model_name == 'videobert':
            return outputs_list
        else:
            return result
    
    def benchmark_interact(self, dataloader):
        batch_size = self.get_loaded_batch_size()
        iterations = self.workload['iterations']
        times_range = []
        report = {}
        report["BS"] = batch_size

        test_data = self._get_fake_samples(batch_size=batch_size,
                        shape=self.configs['segments'][0]['input_tensor_map'],
                        input_type=self.configs['input_type'])

        for _ in range(30):
            self.predict(test_data)

        for _ in range(iterations):
            start_time = time.time()
            self.predict(test_data)
            end_time = time.time()
            times_range.append(end_time - start_time)

        times_range.sort()
        tail_latency = round(
            times_range[int(len(times_range) * 0.99)] * 1000, 2)
        avg_latency = round(sum(times_range) / iterations * 1000, 2)
        qps = int(1000.0 * self.batch_size / avg_latency)

        log.info(
            'Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}'.
            format(self.batch_size, qps, avg_latency, tail_latency))

        report['QPS'] = qps
        report['AVG Latency'] = avg_latency
        report['P99 Latency'] = tail_latency

        return report

    def get_loaded_batch_size(self):
        # return self.workload['batch_sizes'][0]
        return self.batch_size

    def load(self, batch_size) -> None:
        # load engine
        model = self.configs['model']
        model_name = self.configs['model'].split("-")[0]
        model_path = self.configs['model_path']

        if model_name == 'videobert' or model_name == 'conformer':
            engine_path = model_path.split(".")[0] + "_end.engine"

        elif model_name == 'yolov5':
            engine_path = model_path.split(".")[0] + "_sim.engine"

        elif model_name == 'widedeep':
            engine_path = model_path + "/" + model + "_end.engine"
        
        elif model_name == 'roformer':
            engine_path = model_path + "/" + model + ".engine"
        
        elif model_name == 'bert' or model_name == 'albert' or model_name == 'roberta' or model_name == 'deberta' or model_name == 'swin':
            engine_path = os.path.dirname(model_path) + "/" + model + "_end.engine" 

        else:
            engine_path = os.path.dirname(model_path) + "/" + model + ".engine"
        
        # **************to do*************
        if model_name == 'widedeep':      
            engine_path = "general_perf/model_zoo/regular/open_wide_deep_saved_model/widedeep_dynamicshape_sim_" + str(batch_size) + ".engine"

        if model_name == 'conformer':
            engine_path = "general_perf/model_zoo/popular/open_conformer/conformer_encoder_optimizer_end" + ".engine"    
        
        # if model_name == 'roformer':
        #     engine_path = "general_perf/model_zoo/popular/open_roformer/roformer-frozen-sim-modified-" + str(batch_size) + ".engine" 
        
        if model_name == 'deberta':
            engine_path = "general_perf/model_zoo/popular/open_conformer/deberta-base-squad-sim_end" + ".engine"   

        engine, context = init_by_tensorrt(engine_path)

        self.input_type = self.configs['input_type']
        
        self.batch_size = batch_size
        self.model_runtimes = []
        self.engine = engine
        self.context = context

    def _get_fake_samples(self, batch_size, shape, input_type):
        data = {}
        if input_type:
            i = 0
            for key, val in shape.items():
                if key != "text":
                    val = [val[0] * batch_size] + val[1:]
                    data[key] = np.random.random(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                else:
                    data[key] = np.random.random(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                i += 1
            return data
        else:
            raise ValueError("Please provide input type")