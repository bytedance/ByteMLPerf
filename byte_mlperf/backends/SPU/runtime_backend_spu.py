import logging
import tensorflow as tf
import torch
import numpy as np
import time
import yaml
import multiprocessing
from multiprocessing import Manager
from byte_mlperf.backends import runtime_backend
import spu_backend
from inference import SQUADTester, ImageNetTester, ConformerTester, get_yaml_path, squad_preprocess, squad_postprocess

hardware_type = "spu".upper()

tf.get_logger().setLevel('ERROR')
log = logging.getLogger(f"Backend-{hardware_type}")

bfloat16 = tf.bfloat16.as_numpy_dtype
pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "LONG": torch.long
}

tf_dtype_map = {
    "FLOAT32": tf.float32,
    "FLOAT16": tf.float16,
    "INT32": tf.int32,
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "INT8": np.int8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64
}


class RuntimeBackendSPU(runtime_backend.RuntimeBackend):
    def __init__(self):
        super(RuntimeBackendSPU, self).__init__()
        self.hardware_type = hardware_type
        self.need_reload = False
        self.batch_size = None
        self.model_name = ""
        self.configs = None
        self.workload = None
        self.model_info = None
        self.model = None
        self.yaml_config = None
        self.need_reload = True

    def predict(self, feeds):
        if not self.model:
            log.info("no model_runtime...")
            self.load(self.get_loaded_batch_size())
        if self.model_name == "resnet50-torch-fp32":
            request = [feeds['actual_input']]
            response = self.model.inference(request)
            return response
        elif self.model_name == "conformer-encoder-onnx-fp32":
            request = [feeds["src"], feeds["src_pad_mask"]]
            response = self.model.inference(request)
            return response
        elif self.model_name in ["bert-torch-fp32", "albert-torch-fp32", "roberta-torch-fp32"]:
            request, model_info = squad_preprocess(feeds, self.yaml_config)
            response = self.model.inference(request)
            response = squad_postprocess(response, model_info)
            return response
        else:
            raise NotImplementedError(f"task: {self.model_name} not supported")

    def benchmark(self, dataloader):
        batch_sizes = self.workload['batch_sizes']
        reports = []
        iterations = self.workload['iterations']
        for idx, batch_size in enumerate(batch_sizes):
            if batch_size != self.batch_size:
                continue
            self.yaml_config.update(
                {"min_batch_size": self.yaml_config['chunk_size']})
            report = {}
            dataloader.rebatch(batch_size)
            if self.model_name == "resnet50-torch-fp32":
                test_data, _ = dataloader.get_samples(0)
                all_resnet50_start_time_list = []
                all_resnet50_end_time_list = []
                model = ImageNetTester(self.yaml_config)
                model.load_model()
                request = [test_data['actual_input']]
                start = time.time()
                for _ in range(iterations):
                    resnet50_start_time = time.time()
                    all_resnet50_start_time_list.append(resnet50_start_time * 1000)
                    output_data = model.inference(request)
                    resnet50_end_time = time.time()
                    all_resnet50_end_time_list.append(resnet50_end_time * 1000)
                start_time_list = all_resnet50_start_time_list
                end_time_list = all_resnet50_end_time_list

            elif self.model_name == "conformer-encoder-onnx-fp32":
                test_data = dataloader.get_samples(0)
                all_conformer_start_time_list = []
                all_conformer_end_time_list = []
                model = ConformerTester(self.yaml_config)
                model.load_model()
                request = [test_data["src"], test_data["src_pad_mask"]]
                start = time.time()
                for _ in range(iterations):
                    conformer_start_time = time.time()
                    all_conformer_start_time_list.append(conformer_start_time * 1000)
                    output_data = model.inference(request)
                    conformer_end_time = time.time()
                    all_conformer_end_time_list.append(conformer_end_time * 1000)
                start_time_list = all_conformer_start_time_list
                end_time_list = all_conformer_end_time_list

            elif self.model_name in ["bert-torch-fp32", "albert-torch-fp32", "roberta-torch-fp32"]:
                test_data, _ = dataloader.get_samples(0)

                def input_worker(_input_queue, data, iteration, shared_list):
                    for i in range(iteration):
                        batch_start_time = time.time()
                        shared_list.append(batch_start_time)
                        _input_queue.put(data)
                    _input_queue.put(None)
                    return

                def preprocessing_worker(_input_queue, _preprocess_queue, _info_queue, model_info):
                    while True:
                        data = _input_queue.get()
                        if data is None:
                            _info_queue.put(None)
                            _preprocess_queue.put(None)
                            return

                        input_data_list, info = squad_preprocess(data, model_info)
                        _preprocess_queue.put(input_data_list)
                        _info_queue.put(info)

                def inference_worker(_preprocess_queue, _inference_queue, config):
                    _model = SQUADTester(config)
                    model.load_model()
                    while True:
                        data = _preprocess_queue.get()
                        if data is None:
                            _inference_queue.put(None)
                            _model.destroy()
                            return
                        _output_data = model.inference(data)
                        _inference_queue.put(_output_data)

                def postprocessing_worker(_inference_queue, _postprocess_queue, _info_queue):
                    while True:
                        data = _inference_queue.get()
                        info = _info_queue.get()
                        if data is None:
                            _postprocess_queue.put(None)
                            _info_queue.put(None)
                            return
                        _postprocess_queue.put(squad_postprocess(data, info))

                def consumer(_postprocess_queue, _shared_end_list):
                    ans = []
                    while True:
                        i = _postprocess_queue.get()
                        batch_end_time = time.time()
                        _shared_end_list.append(batch_end_time)
                        if i is None:
                            return ans
                        ans.append(i)

                manager = Manager()
                shared_start_list = manager.list()
                shared_end_list = manager.list()
                # Inference Pipeline
                input_queue = multiprocessing.JoinableQueue()
                preprocess_queue = multiprocessing.JoinableQueue()
                info_queue = multiprocessing.JoinableQueue()
                inference_queue = multiprocessing.JoinableQueue()
                postprocess_queue = multiprocessing.JoinableQueue()

                # [0] 获取数据的进程
                input_process = multiprocessing.Process(
                    target=input_worker, args=(input_queue, test_data, iterations, shared_start_list))
                # [1] 模型前处理进程
                preprocessing_process = multiprocessing.Process(
                    target=preprocessing_worker, args=(input_queue, preprocess_queue, info_queue, self.yaml_config))
                # [2] 模型推理进程
                inference_process = multiprocessing.Process(
                    target=inference_worker, args=(preprocess_queue, inference_queue, self.yaml_config))

                # [3] 模型后处理进程
                postprocessing_process = multiprocessing.Process(
                    target=postprocessing_worker, args=(inference_queue, postprocess_queue, info_queue))

                # 开始计时
                start = time.time()
                input_process.start()
                preprocessing_process.start()
                inference_process.start()
                postprocessing_process.start()

                processes = [input_process, preprocessing_process, inference_process, postprocessing_process]
                responses = consumer(postprocess_queue, shared_end_list)
                for p in processes:
                    p.join()

                start_time_list = list(shared_start_list)
                end_time_list = list(shared_end_list)
            else:
                raise NotImplementedError(f"task: {self.model_name} not supported")

            # 结束计时
            duration = (time.time() - start) * 1000
            all_latency = [(x - y) * 1000 for x, y in zip(end_time_list, start_time_list)]
            all_latency.sort()
            index = int(len(all_latency) * 0.99)
            tail_latency = all_latency[index] / 1000
            avg_latency = duration / iterations
            qps = round(1000.0 * batch_size / avg_latency, 2)
            tail_latency = round(tail_latency, 2)
            avg_latency = round(avg_latency, 2)
            qps = round(qps, 2)
            report['BS'] = batch_size
            report['QPS'] = qps
            report['AVG Latency'] = avg_latency
            report['P99 Latency'] = tail_latency
            print(f"AVG Latency:{avg_latency}, P99 Latency:{tail_latency}")
            reports.append(report)
        return reports

    def get_loaded_batch_size(self):
        # only used in accuracy mode, not in benchmark.
        name = self.configs['model']
        self.yaml_config.update({"min_batch_size": self.yaml_config['chunk_size']})
        if "bert-torch-fp32" in name or "albert-torch-fp32" in name:
            return 12
        elif "roberta-torch-fp32" in name:
            return 4
        elif "resnet50-torch-fp32" in name:
            return 16
        elif "conformer-encoder-onnx-fp32" in name:
            return 16
        else:
            raise NotImplementedError(f"task : {name} not supported")

    def load(self, batch_size):
        self.batch_size = batch_size
        self.model_name = self.configs['model']
        self.model_info.update({"input_name": self.model_info['inputs'].split(",")})
        self.yaml_config = yaml.safe_load(open(get_yaml_path(self.model_info["model"]), "r"))
        self.yaml_config.update({
            "model": self.configs["model"],
            "input_name": self.model_info["input_name"]
        })
        if 'input_order' in self.yaml_config["model_input"][0]:
            self.yaml_config.update(
                {"input_order": [inp['input_order'] for inp in self.yaml_config["model_input"]]})

        if self.need_reload:
            dataset_name = self.model_info['dataset_name']
            if dataset_name == "open_squad":
                self.model = SQUADTester(self.model_info)
            elif dataset_name == "open_imagenet":
                self.model = ImageNetTester(self.model_info)
            elif dataset_name == "fake_dataset":
                self.model = ConformerTester(self.model_info)
            else:
                raise NotImplementedError(f"dataset_name : {dataset_name} not supported")
            self.model.load_model()
            self.need_reload = False
        else:
            log.info("model has been loaded, skip load process")
