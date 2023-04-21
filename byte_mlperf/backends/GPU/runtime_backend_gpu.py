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
from backends import backend

log = logging.getLogger("RuntimeBackendGPU")

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


class RuntimeBackendGPU(backend.RuntimeBackend):
    def __init__(self):
        super(BackendGPU, self).__init__()
        self.hardware_type = 'GPU'
        self.need_reload = False
        self.model_runtimes = []

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
