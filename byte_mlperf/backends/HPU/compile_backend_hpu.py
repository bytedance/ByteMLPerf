import os
import json
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import time
import numpy as np

from byte_mlperf.backends import compile_backend

log = logging.getLogger("CompileBackendHPU")

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "LONG": torch.long
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64
}


class CompileBackendHPU(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendHPU, self).__init__()
        self.hardware_type = 'HPU'
        self.need_reload = False
        self.model_runtimes = []

    def _update_model_env(self):
        if self.model_info["model"] in ("bert-torch-fp32", "albert-torch-fp32"):
            os.environ['LOWER_LIST'] ='byte_mlperf/backends/HPU/bert/bf16.txt'
            os.environ['FP32_LIST'] ='byte_mlperf/backends/HPU/bert/fp32.txt'

    def compile(self, config, dataloader=None):
        result = {
            "model": config['model_info']['model'],
            "framework": config['model_info']['framework'],
            "compile_precision": “BF16”,
            "optimizations":{},
            "instance_count": 1,
            "device_count": 1,
            "input_type": config['model_info']['input_type'].split(","),
            "max_batch_size": config['model_info']['max_batch_size'],
            "compile_status": "success",
            "sg_percent": 100,
            "segments": [
                {
                    "sg_idx":
                    0,
                    "is_fallback": False,
                    "input_tensor_map": config['model_info']['input_shape'],
                    "output_tensor_map": config['model_info']['outputs'],
                    "compiled_model": [
                        {
                            "compiled_bs": 1,
                            "compiled_obj": config['model_info']['model_path'],
                        },
                    ],
                },
            ]
        }
        self.configs = result
        self.workload = config['workload']
        self.model_info = config['model_info']
        self._update_model_env()
        return result

    def get_interact_profile(self, config):
        model_profile = []
        file_path = "byte_mlperf/backends/HPU/" + self.hardware_type + '.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                model_profile = json.load(f)
        else:
            log.info(
                'File path: {} does not exist, please check'.format(file_path))

        return model_profile

    def get_best_batch_size(self):
        """
        Get Best Batch Size for the model
        """
        return None
