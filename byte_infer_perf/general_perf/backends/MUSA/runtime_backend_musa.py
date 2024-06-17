import os
import json
import logging
import torch
import torch_musa
import time
import numpy as np

from general_perf.backends import runtime_backend

torch.backends.mudnn.allow_tf32 = True
log = logging.getLogger("BackendMUSA")

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "LONG": torch.long
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool
}


class RuntimeBackendMUSA(runtime_backend.RuntimeBackend):
    def __init__(self):
        super(RuntimeBackendMUSA, self).__init__()
        self.hardware_type = 'MUSA'
        self.need_reload = False
        self.model_runtimes = []
        self.configs = None
        self.batch_size = -1

    def predict(self, feeds):
        results = {}
        input_tensors = []
        i = 0
        for key, _ in feeds.items():
            input_tensors.append(
                torch.tensor(feeds[key],
                                dtype=pt_dtype_map[self.input_type[i]]).to(
                                    self.device))
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
        return results

    def benchmark(self, dataloader):
        iterations = self.workload['iterations']
        batch_size = self.get_loaded_batch_size()
        times_range = []
        report = {}
        report['BS'] = batch_size
        test_data = self._get_fake_samples(
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

        log.info(
            'Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}'.
            format(batch_size, qps, avg_latency, tail_latency))

        report['QPS'] = qps
        report['AVG Latency'] = avg_latency
        report['P99 Latency'] = tail_latency

        return report

    def get_loaded_batch_size(self):
        return self.batch_size

    def load(self, batch_size) -> None:
        self.batch_size = batch_size
        self.model_runtimes = []
        self.input_type = self.configs['input_type']
        self.framework = self.configs['framework']

        self.model_name = self.configs['model']

        for i, segment in enumerate(self.configs['segments']):
            # there is no input/output meta data i the graph so it need to come from config.
            if not segment['input_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs inputs")
            if not segment['output_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs outputs")

            self.input_shapes = segment['input_tensor_map']
            self.outputs = segment['output_tensor_map'].split(",")

            self.device = "musa"
            model = torch.jit.load(
                segment['compiled_model'][0]['compiled_obj'],
                torch.device('musa'))
            if os.getenv("NCHW"):
                model = model.to(memory_format=torch.channels_last)
            if os.getenv("FP16"):
                model = model.half()
            model.eval()
            self.model_runtimes.append(model)

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
