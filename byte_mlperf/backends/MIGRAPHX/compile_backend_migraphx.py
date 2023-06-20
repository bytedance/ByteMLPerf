import os
import json
import logging
import copy

import torch
# import lego

import tensorflow as tf

import onnx

import migraphx

tf.get_logger().setLevel('ERROR')

import numpy as np
import time
from byte_mlperf.backends import compile_backend
import subprocess

log = logging.getLogger("CompileBackendMIGRAPHX")
from byte_mlperf.tools import saved_to_onnx, torch_to_onnx

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


class CompileBackendMIGRAPHX(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendMIGRAPHX, self).__init__()
        self.target = 'rocm'
        self.hardware_type= 'MIGRAPHX'

    def compile(self, config, dataloader=None):
        self.model_name = config['model_info']['model']
        self.framework = config['model_info']['framework']
        self.input_type = config['model_info']['input_type'].split(",")

        model_paths = config['model_info']['model_path'].split('.')
        model_path = config['model_info']['model_path']
        test_data = dataloader.get_fake_samples(
            1, config['model_info']['input_shape'], self.input_type)
        input_tensors = []
        #convert tf/pytorch model to onnx model, tf works, not verify pytorch now.
        model_onnx_path=model_path
        if self.framework == "Tensorflow":
            model_onnx_path = os.path.join(model_path,self.model_name+ '.onnx')
            if not os.path.exists(model_onnx_path):
                saved_to_onnx.savedmodel_to_onnx(model_path, model_onnx_path)
        elif self.framework == "Pytorch":
            raise NotImplementedError("MIGraphX backend for models of PyTorch framework has not been implemented yet.")
        batch_sizes = config['workload']['batch_sizes']
        for batch_size in batch_sizes:
            print(batch_sizes)
            model_path_for_batch_size = model_onnx_path.rsplit("/",1)
            model_onnx_path_set_batch_size = os.path.join(model_path_for_batch_size[0],str(batch_size),model_path_for_batch_size[1])
            model_dir = os.path.join(model_path_for_batch_size[0],str(batch_size))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_paths = model_onnx_path_set_batch_size.split('.')
            model_path = model_paths[0] + "_optimized." + model_paths[1]
            if not os.path.exists(model_path):
                new_input = {}
                for key, value in config['model_info']['input_shape'].items():
                    if(('videobert' in config['model_info']['model']) and (key=='text')):
                        new_input[key]=value
                    else:
                        new_input[key]=[value[0]*batch_size]+value[1:]
                model = migraphx.parse_onnx(model_onnx_path,map_input_dims=new_input,default_dim_value=batch_size)
                model.compile(migraphx.get_target("gpu"))
                migraphx.save(model, model_path, format='msgpack')

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
            config['model_info']['max_batch_size'][-1],
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
                            "compiled_bs": config["workload"]["batch_sizes"][-1],
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
        file_path = "backends/MIGRAPHX/" + self.hardware_type + '.json'
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
