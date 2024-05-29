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
import importlib

from general_perf.backends import runtime_backend
from general_perf.backends.ILUVATAR.common import init_by_tensorrt, setup_io_bindings
from general_perf.backends.ILUVATAR.common import Task, TaskThread
from cuda import cuda, cudart
import numa

from general_perf.backends.ILUVATAR.common import load_ixrt_plugin

log = logging.getLogger("RuntimeBackendILUVATAR")

Dims = None

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "INT32":torch.int32,
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
        self.predict_fps = None
        self.predict_time = None
        self.task = None
        self.inputs = None
        self.outputs = None
        self.allocations = None
        numa.memory.set_local_alloc()
        numa.schedule.run_on_nodes(0)

    def isSDmodel(self, model_name):
        result = False
        if model_name == 'vae-decoder-onnx-fp32' or model_name == 'vae-encoder-onnx-fp32' or model_name == 'clip-onnx-fp32':
            result = True
        return result

    # Dual-core inference of Tian SoC BI-150 graphics card
    def benchmark(self, dataloader):
        performance_reports = []
        merged_dict = {}
        model_name = self.configs["model"].split("-")[0]
        
        workers = []
        lock = threading.Lock()
        for i in range(2):
            device_id = i
            self.task = Task(self.batch_size, dataloader, device_id, self.load, self.benchmark_interact, performance_reports, lock, framework=model_name)

            work = TaskThread(self.task.run, [])
            workers.append(work)
            work.start()
            work.join()
        
        if model_name != 'gpt2':
            if not self.isSDmodel(self.configs["model"]):
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

                if model_name != 'gpt2':
                    predict_qps = performance_reports[0]['predict QPS'] + performance_reports[1]['predict QPS']
                    predict_avg_latency = round(((performance_reports[0]['predict AVG Latency'] + performance_reports[1]['predict AVG Latency']) / 2.0), 2)
                    predict_p99_latency = round(((performance_reports[0]['predict P99 Latency'] + performance_reports[1]['predict P99 Latency']) / 2.0), 2)

                    merged_dict['predict QPS'] = predict_qps
                    merged_dict['predict AVG Latency'] = predict_avg_latency
                    merged_dict["predict P99 Latency"] = predict_p99_latency
                
        return merged_dict
    
    def init_allocs(self):
        if self.inputs is not None:
            for i in range(len(self.inputs)):
                err, = cudart.cudaFree(self.inputs[i]["allocation"])
                assert err == cudart.cudaError_t.cudaSuccess

            for i in range(len(self.outputs)):
                err, = cudart.cudaFree(self.outputs[i]["allocation"])
                assert err == cudart.cudaError_t.cudaSuccess
            self.inputs = None

    def get_allocs(self):
        if self.inputs is None:
            self.inputs, self.outputs, self.allocations = setup_io_bindings(self.engine, self.context)
        return self.inputs, self.outputs, self.allocations

    def predict_dump(self, feeds):
        input_tensors = []
        i = 0

        model_name = self.configs["model"].split("-")[0]
    
        if model_name != 'gpt2':
            if model_name == 'deberta':
                keys = list(feeds.keys())
                input_ids = feeds[keys[0]]
                attention_mask = feeds[keys[1]]
                input_tensors = [input_ids, attention_mask]

            else:
                for key, _ in feeds.items():
                    input_tensors.append(feeds[key])
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

                elif model_name == 'deberta':
                    input_names = [
                        "input_ids.1",
                        "attention_mask.1",
                    ]
                    for input_name in input_names:
                        if input_name == 'input_ids.1':
                            input_shape = input_tensors[0].shape
                        if input_name == 'attention_mask.1':
                            input_shape = input_tensors[1].shape
                    
                        input_idx = engine.get_binding_index(input_name)
                        context.set_binding_shape(input_idx, Dims(input_shape))

                else:
                    input_shape = input_tensors[i].shape
                    input_idx = engine.get_binding_index(input_name)
                    context.set_binding_shape(input_idx, Dims(input_shape))
                    i += 1
            
            # Setup I/O bindings
            inputs, outputs, allocations = self.get_allocs()

            # Prepare the output data
            outputs_list = []
            for i in range(len(outputs)):
                output = np.zeros(outputs[i]["shape"], outputs[i]["dtype"])
                outputs_list.append(output)

            data_batch_list = []
            for i in range(len(input_tensors)):
                data_batch = np.ascontiguousarray(input_tensors[i])
                data_batch_list.append(data_batch)

        return input_tensors, inputs, outputs, data_batch_list, allocations, context, outputs_list

    def predict_timing(self, input_tensors, inputs, outputs, data_batch_list, allocations, context, outputs_list):
        model_name = self.configs["model"].split("-")[0]
        
        # H2D: host to device
        for i in range(len(inputs)):
            (err, ) = cudart.cudaHostRegister(data_batch_list[i], inputs[i]["nbytes"], 2)

        for i in range(len(inputs)):
            (err, ) = cudart.cudaMemcpy(
                        inputs[i]["allocation"],
                        data_batch_list[i],
                        inputs[i]["nbytes"],
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )

        for i in range(len(inputs)):
            (err, ) = cudart.cudaHostUnregister(data_batch_list[i])

        starttime = time.time()
        context.execute_v2(allocations)
        endtime = time.time()

        self.predict_time = endtime - starttime
        
        # D2H: device to host
        for i in range(len(outputs)):
            (err, )= cudart.cudaMemcpy(outputs_list[i], 
                        outputs[i]["allocation"], 
                        outputs[i]["nbytes"], 
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            )
        
        result = {}

        output_tensor_map = self.configs["segments"][0]["output_tensor_map"]
        output_name = output_tensor_map.split(",")

        for i in range(len(output_name)):
            if model_name == 'yolov5':
                result[output_name[0]] = outputs_list[0]
                break

            result[output_name[i]] = outputs_list[i]
        
        if model_name == 'videobert':
            return outputs_list
        
        elif model_name == 'gpt2':
            return None
        
        else:
            return result

    def predict(self, feeds):
        # The deberta model is currently unable to undergo accuracy testing temporarily
        input_tensors = []
        i = 0

        model_name = self.configs["model"].split("-")[0]
        if self.isSDmodel(self.configs["model"]):
            for key, _ in feeds.items():
                tmp_tensor = torch.tensor(feeds[key],
                                        dtype=pt_dtype_map[self.input_type[i]])
                input_tensors.append(tmp_tensor)
                i += 1

            self.predict_sd(input_tensors)
            return
        
        elif model_name != 'gpt2':
            if model_name == 'deberta':
                keys = list(feeds.keys())
                input_ids = np.array(feeds[keys[0]], dtype=INPUT_TYPE[self.input_type[i]])
                attention_mask = np.array(feeds[keys[1]], dtype=INPUT_TYPE[self.input_type[i]])
                input_tensors = [input_ids, attention_mask]

            else:
                trans_index = [0, 1, 2]
                if model_name == 'bert' and self.configs['compile_precision'] == 'INT8':
                    trans_index = [0, 2, 1]

                for key, _ in feeds.items():
                    tmp_tensor = np.array(feeds[key], dtype=INPUT_TYPE[self.input_type[trans_index[i]]])
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

                elif model_name == 'deberta':
                    input_names = [
                        "input_ids.1",
                        "attention_mask.1",
                    ]
                    for input_name in input_names:
                        if input_name == 'input_ids.1':
                            input_shape = input_tensors[0].shape
                        if input_name == 'attention_mask.1':
                            input_shape = input_tensors[1].shape
                    
                        input_idx = engine.get_binding_index(input_name)
                        context.set_binding_shape(input_idx, Dims(input_shape))

                else:
                    input_shape = input_tensors[i].shape
                    input_idx = engine.get_binding_index(input_name)
                    context.set_binding_shape(input_idx, Dims(input_shape))
                    i += 1
            
            # Setup I/O bindings
            inputs, outputs, allocations = self.get_allocs()

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
                (err, ) = cudart.cudaHostRegister(data_batch_list[i], inputs[i]["nbytes"], 2)

            for i in range(len(inputs)):
                (err, ) = cudart.cudaMemcpy(
                            inputs[i]["allocation"],
                            data_batch_list[i],
                            inputs[i]["nbytes"],
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                )
            
            for i in range(len(inputs)):
                (err, ) = cudart.cudaHostUnregister(data_batch_list[i])

            starttime = time.time()
            context.execute_v2(allocations)
            endtime = time.time()

            self.predict_time = endtime - starttime
            
            # D2H: device to host
            for i in range(len(outputs)):
                (err, )= cudart.cudaMemcpy(outputs_list[i], 
                            outputs[i]["allocation"], 
                            outputs[i]["nbytes"], 
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
                )
            
            # Free Gpu Memory
            # cuda-python
            self.init_allocs()
            
            result = {}

            output_tensor_map = self.configs["segments"][0]["output_tensor_map"]
            output_name = output_tensor_map.split(",")

            for i in range(len(output_name)):
                if model_name == 'yolov5':
                    result[output_name[0]] = outputs_list[0]
                    break

                result[output_name[i]] = outputs_list[i]
        else:
            self.predict_igie(feeds)
            
        if model_name == 'videobert':
            return outputs_list
        
        elif model_name == 'gpt2':
            return None
        
        else:
            return result
    
    def predict_igie(self, dataloader):
        tvm = importlib.import_module("tvm")
        self.task.module.set_input("input_ids", tvm.nd.array(dataloader["input_ids"].astype('int64'), self.device))
        self.task.module.run()
        output = self.task.module.get_output(0)

        return output
    
    def benchmark_interact(self, dataloader):
        batch_size = self.get_loaded_batch_size()
        iterations = self.workload['iterations']
        model_name = self.configs["model"].split("-")[0]
        times_range = []
        predict_range = []
        report = {}
        report["BS"] = batch_size

        if model_name == 'gpt2':
            self.load_igie(batch_size)
        elif self.isSDmodel(self.configs["model"]):
            self.load_sd(batch_size)   
    
        test_data = self._get_fake_samples(batch_size=batch_size,
                        shape=self.configs['segments'][0]['input_tensor_map'],
                        input_type=self.configs['input_type'])
        
        # Free Gpu Memory
        # cuda-python
        self.init_allocs()

        for _ in range(30):
            self.predict(test_data)

        for _ in range(iterations):
            if model_name != 'gpt2' and model_name != 'vae' and model_name != 'clip':
                input_tensors, inputs, outputs, data_batch_list, allocations, context, outputs_list = self.predict_dump(test_data)

                start_time = time.time()
                self.predict_timing(input_tensors, inputs, outputs, data_batch_list, allocations, context, outputs_list)
                end_time = time.time()
            
            else:
                start_time = time.time()
                self.predict(test_data)
                end_time = time.time()

            times_range.append(end_time - start_time)
            predict_range.append(self.predict_time)           

        times_range.sort()
        tail_latency = round(
            times_range[int(len(times_range) * 0.99)] * 1000, 2)
        avg_latency = round(sum(times_range) / iterations * 1000, 2)
        qps = int(1000.0 * self.batch_size / avg_latency)

        if model_name != 'gpt2':
            predict_range.sort()
            predict_tail_latency = round(
                predict_range[int(len(predict_range) * 0.99)] * 1000, 2)
            predict_avg_latency = round(sum(predict_range) / iterations * 1000, 2)
            fps = int(1000.0 * batch_size / predict_avg_latency)

        log.info(
            'Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}'.
            format(self.batch_size, qps, avg_latency, tail_latency))
        
        # log.info(
        #     'Batch size is {}, fps: {}, predict_avg_latency:{}, predict_tail_latency:{}'.
        #     format(self.batch_size, fps, predict_avg_latency, tail_latency))


        report['QPS'] = qps
        report['AVG Latency'] = avg_latency
        report['P99 Latency'] = tail_latency

        if model_name != 'gpt2':
            report['predict QPS'] = fps
            report['predict AVG Latency'] = predict_avg_latency
            report['predict P99 Latency'] = predict_tail_latency

        return report

    def get_loaded_batch_size(self):
        # return self.workload['batch_sizes'][0]
        return self.batch_size

    def load(self, batch_size) -> None:
        global Dims

        # load engine
        model = self.configs['model']
        model_name = self.configs['model'].split("-")[0]
        model_path = self.configs['model_path']

        precision = self.configs['compile_precision'].replace('FP32', 'FP16')

        if precision == 'FP16':
            if model_name == 'resnet50' or model_name == 'bert' or model_name == 'albert' or model == 'deberta' or model_name == 'yolov5':
                mod = importlib.import_module("tensorrt_legacy")
                Dims = getattr(mod, "Dims")
            else:
                mod = importlib.import_module("tensorrt")
                Dims = getattr(mod, "Dims")

        if precision == 'INT8':
            mod = importlib.import_module("tensorrt")
            Dims = getattr(mod, "Dims")     

        load_ixrt_plugin(model=model_name, precision=precision)

        if model_name == 'gpt2':
            self.batch_size = batch_size
            return
        
        elif self.isSDmodel(model):
            self.batch_size = batch_size
            #self.load_sd(batch_size)
            return
        
        if self.configs['compile_precision'] == 'FP16':
            if model_name == 'videobert' or model_name == 'conformer' or model_name == 'yolov5':
                engine_path = model_path.split(".")[0] + "_end.engine"

            elif model_name == 'widedeep' or model_name == 'roformer':
                engine_path = model_path + "/" + model + "_end.engine"
                    
            elif model_name == 'bert' or model_name == 'albert' or model_name == 'roberta' or model_name == 'deberta' or model_name == 'swin' \
                or model_name == 'resnet50':
                engine_path = os.path.dirname(model_path) + "/" + model + "_end.engine" 

            else:
                engine_path = os.path.dirname(model_path) + "/" + model + ".engine"
            
            if model_name == 'widedeep':      
                engine_path = "general_perf/model_zoo/regular/open_wide_deep_saved_model/widedeep_dynamicshape" + ".engine"

            if model_name == 'roformer':
                engine_path = "general_perf/model_zoo/popular/open_roformer/roformer-frozen_end" + ".engine"     
            
            if model_name == 'deberta':
                engine_path = "general_perf/model_zoo/popular/open_deberta/deberta-sim-drop-clip-drop-invaild-cast_end" + ".engine"

        if self.configs['compile_precision'] == 'INT8':
            if model_name == 'widedeep':
                engine_path = "general_perf/model_zoo/regular/open_wide_deep_saved_model/quantized_widedeep_staticshape" + ".engine"    
            
            if model_name == 'resnet50':
                engine_path = "general_perf/model_zoo/regular/open_resnet50/quantized_Resnet50" + ".engine"

            if model_name == 'yolov5':
                engine_path = "general_perf/model_zoo/popular/open_yolov5/quantized_yolov5s" + ".engine"    

            if model_name == 'bert':
                engine_path = "general_perf/model_zoo/regular/open_bert/bert_zijie_int8_b196.engine"

        engine, context = init_by_tensorrt(engine_path)

        self.model_runtimes.append(engine)

        self.input_type = self.configs['input_type']
        
        self.batch_size = batch_size
        self.engine = engine
        self.context = context


    def load_sd(self, batch_size):
        model_path = self.configs['model_path']

        import onnx
        from onnx2torch import convert
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self.model_sd = convert(model_path).to(device)

        self.input_type = self.configs['input_type']
        self.batch_size = batch_size
        pass

    def predict_sd(self, dataloader):
        self.model_sd = self.model_sd.eval()
        dataloader = dataloader[0].to('cuda')
        torch.cuda.synchronize()
        starttime = time.time()
        out = self.model_sd(dataloader)
        torch.cuda.synchronize()
        endtime = time.time()

        self.predict_time = endtime - starttime

        return out

    def load_igie(self, batch_size):
        model = self.configs['model']
        model_path = self.configs['model_path']

        tvm = importlib.import_module("tvm")
        from general_perf.backends.ILUVATAR.utils import get_target

        target, _ = get_target('iluvatar_with_all_libs')
        device = tvm.device(target.kind.name, self.task.device_id)
        engine_path = os.path.dirname(model_path) + "/" + model + "_bs" + str(batch_size) + ".so"
        lib = tvm.runtime.load_module(engine_path)
        self.task.module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

        self.device = device
        self.batch_size = batch_size

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