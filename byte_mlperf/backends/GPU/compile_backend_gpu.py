import os
import json
import logging
import copy

import torch
import torch_tensorrt
# import lego

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import onnx
# import onnx_tensorrt.backend as onnx_convert

tf.get_logger().setLevel('ERROR')

import numpy as np
import time
from backends import compile_backend

log = logging.getLogger("CompileBackendGPU")

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "INT32": torch.int32,
    "LONG": torch.long,
    "BOOL": torch.bool
}

tf_dtype_map = {
    "FLOAT32": tf.float32,
    "FLOAT16": tf.float16,
    "INT32": tf.int32,
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool,
}


class CompileBackendGPU(backend.CompileBackend):
    def __init__(self):
        super(CompileBackendGPU, self).__init__()
        self.hardware_type = 'GPU'
        self.need_reload = False
        self.model_runtimes = []

    def compile(self, config, dataloader=None):
        self.model_name = config['model_info']['model']
        self.framework = config['model_info']['framework']
        self.input_type = config['model_info']['input_type'].split(",")

        model_paths = config['model_info']['model_path'].split('.')
        model_path = config['model_info']['model_path']
        test_data = dataloader.get_fake_samples(
            1, config['model_info']['input_shape'], self.input_type)
        input_tensors = []
        if self.framework == "Pytorch":
            lego.torch_load_lego_library()
            model_path = model_paths[0] + "_optimized." + model_paths[1]
            for i, (key, value) in enumerate(test_data.items()):
                input_tensors.append(
                    torch.tensor(
                        value, dtype=pt_dtype_map[self.input_type[i]]).cuda())
            if not os.path.exists(model_path):
                original_model = torch.jit.load(
                    config['model_info']['model_path']).cuda().half()
                if 'bert' in config['model_info']['model'] or 'vit' in config[
                        'model_info']['model']:
                    lego_model = lego.optimize(original_model, input_tensors)
                    lego_model.save(model_path)
                    lego.perf_model(original_model, lego_model, input_tensors)
                else:
                    tensort_input = []
                    for i, (key, value) in enumerate(
                            config['model_info']['input_shape'].items()):
                        tensort_input.append(
                            torch_tensorrt.Input(value, dtype=torch.half))
                    model_trt = torch_tensorrt.compile(
                        original_model,
                        inputs=tensort_input,
                        enabled_precisions={torch.half})
                    model_trt.save(model_path)
        elif self.framework == "Tensorflow":
            model_path = model_path + '_optimized'
            if not os.path.exists(model_path):
                params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
                params = params._replace(
                    precision_mode=trt.TrtPrecisionMode.FP16,
                    max_workspace_size_bytes=2 << 32,  # 8,589,934,592 bytes
                    maximum_cached_engines=100,
                    minimum_segment_size=3,
                    allow_build_at_runtime=True)
                converter = trt.TrtGraphConverterV2(
                    input_saved_model_dir=config['model_info']['model_path'],
                    conversion_params=params,
                )
                converter.convert()
                converter.save(model_path)
        # else:
        # model_path = model_paths[0] + "_optimized." + model_paths[1]
        # original_model = onnx.load(config['model_info']['model_path'])

        result = {
            "model":
            self.model_name,
            "framework":
            config['model_info']['framework'],
            "compile_precision":
            config['model_info']['model_precision'],
            "input_type":
            self.input_type,
            "max_batch_size":
            config['model_info']['max_batch_size'],
            "compile_status":
            "success",
            "sg_percent":
            100,
            "segments": [
                {
                    "sg_idx":
                    0,
                    "is_fallback":
                    False,
                    "input_tensor_map":
                    config['model_info']['input_shape'],
                    "output_tensor_map":
                    config['model_info']['outputs'],
                    "compiled_model": [
                        {
                            "compiled_bs": 1,
                            "compiled_obj": model_path,
                        },
                    ],
                },
            ]
        }
        self.configs = result
        self.workload = config['workload']
        self.model_info = config['model_info']
        return result

    def get_interact_profile(self, config):
        model_profile = []
        file_path = "backends/GPU/" + self.hardware_type + '.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                model_profile = json.load(f)
        else:
            log.info(
                'File path: {} does not exist, please check'.format(file_path))

        return model_profile

    def predict(self, feeds):
        if not self.model_runtimes:
            self._load()

        results = {}
        if self.framework == "Tensorflow":
            entry_rt = self.model_runtimes[0].signatures['serving_default']
            all_sn_inputs = entry_rt.structured_input_signature

            def get_real_feeds(feeds, sn_inputs):
                sn_inputs = tf.nest.flatten(sn_inputs, True)
                real_feeds = {}
                itr = 0
                for _, val in feeds.items():
                    real_feeds[sn_inputs[itr].name] = tf.constant(val)
                    itr += 1
                return real_feeds

            real_feeds = get_real_feeds(feeds, all_sn_inputs)

            for model_runtime in self.model_runtimes:
                with tf.device('/GPU:0'):
                    _results = model_runtime.signatures['serving_default'](
                        **real_feeds)

            results = {}
            for key, val in _results.items():
                results[key] = val.numpy()

            assert len(results) != 0

        elif self.framework == "Pytorch":
            input_tensors = []
            i = 0
            for key, _ in feeds.items():
                input_tensors.append(
                    torch.tensor(
                        feeds[key],
                        dtype=pt_dtype_map[self.input_type[i]]).cuda())
                i += 1
            with torch.no_grad():
                for model_runtime in self.model_runtimes:
                    results = model_runtime(*input_tensors)

            if isinstance(results, dict):
                for key, val in results.items():
                    results[key] = val.cpu().detach().numpy()
            elif isinstance(results, tuple):
                dic = {}
                for i, key in enumerate(self.outputs):
                    dic[key] = list(results)[i]
            else:
                results = {self.outputs[0]: results.cpu().numpy()}
        else:
            for model_runtime in self.model_runtimes:
                results = model_runtime.run(feeds)
        return results

    def benchmark(self, dataloader):
        batch_sizes = self.workload['batch_sizes']
        iterations = self.workload['iterations']
        reports = []
        for batch_size in batch_sizes:
            print(batch_sizes)
            times_range = []
            report = {}
            report['BS'] = batch_size
            test_data = dataloader.get_fake_samples(
                batch_size, self.configs['segments'][0]['input_tensor_map'],
                self.configs['input_type'])

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
            qps = int(1000.0 * batch_size / avg_latency)

            report['QPS'] = qps
            report['AVG Latency'] = avg_latency
            report['P99 Latency'] = tail_latency
            reports.append(report)

        return reports

    def get_loaded_batch_size(self):
        return 4

    def _load(self):
        for i, segment in enumerate(self.configs['segments']):
            # there is no input/output meta data i the graph so it need to come from config.
            if not segment['input_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs inputs")
            if not segment['output_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs outputs")

            self.input_shapes = segment['input_tensor_map']
            self.outputs = segment['output_tensor_map'].split(",")

            if self.framework == "Tensorflow":
                with tf.device('/GPU:1'):
                    model = tf.saved_model.load(
                        segment['compiled_model'][0]['compiled_obj'])
            elif self.framework == "Pytorch":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                model = torch.jit.load(
                    segment['compiled_model'][0]['compiled_obj'],
                    torch.device(self.device))
                if torch.cuda.is_available():
                    model.cuda()
                inputs = list(model.graph.inputs())
                names = [i.debugName() for i in inputs]
                model.eval()
            else:
                # original_model = onnx.load(segment['compiled_model'][0]['compiled_obj'])
                # model =  onnx_convert.prepare(original_model, device='CUDA:1')

                import onnxruntime
                model = onnxruntime.InferenceSession(
                    segment['compiled_model'][0]['compiled_obj'],
                    providers=['CUDAExecutionProvider'])

            self.model_runtimes.append(model)
