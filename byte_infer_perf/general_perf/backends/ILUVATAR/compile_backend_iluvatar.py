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
import logging
import subprocess

import tensorrt

from general_perf.backends.ILUVATAR.common import load_ixrt_plugin
load_ixrt_plugin()

from general_perf.backends.ILUVATAR.common import build_engine, build_igie_engine
from general_perf.backends.ILUVATAR.optimizer.passes import *
from general_perf.tools.torch_to_onnx import torch_to_onnx
from general_perf.tools.saved_to_onnx import savedmodel_to_onnx
from general_perf.model_zoo import *
from general_perf.backends import compile_backend

log = logging.getLogger("CompileBackendILUVATAR")


class CompileBackendILUVATAR(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendILUVATAR, self).__init__()
        self.hardware_type = "ILUVATAR"
        self.need_reload = False
        self.model_runtimes = []
        self.model_config = None

    def version(self) -> str:
        """Return compile backend version details."""
        return tensorrt.__version__
    
    def compile(self, configs, dataloader=None):
        model = configs['model_info']['model']
        model_name = configs['model_info']['model'].split("-")[0]
        model_path = configs['model_info']['model_path']
        MaxBatchSize = configs['model_info']['max_batch_size']

        # call the ONNX model and the compiled engine file
        if model_name == 'videobert' or model_name == 'conformer' or model_name == 'yolov5':
            onnx_model_path = model_path.split(".")[0] + "_end.onnx"
            engine_path = model_path.split(".")[0] + "_end.engine"

        elif model_name == 'widedeep' or model_name == 'roformer':
            onnx_model_path = model_path + "/" + model + "_end.onnx"
            engine_path = model_path + "/" + model + "_end.engine"

        elif model_name == 'bert' or model_name == 'albert' or model_name == 'roberta' or model_name == 'deberta' or model_name == 'swin' \
             or model_name == 'resnet50':
            onnx_model_path = os.path.dirname(model_path) + "/" + model + "_end.onnx"
            engine_path = os.path.dirname(model_path) + "/" + model + "_end.engine"
        
        else:
            onnx_model_path = os.path.dirname(model_path) + "/" + model + ".onnx"
            engine_path = os.path.dirname(model_path) + "/" + model + ".engine"

        # model preprocessing
        self.get_onnx(configs)

        # build engine
        if configs['model_info']['model_precision'].replace('FP32', 'FP16') == 'FP16':
            precision_flag = "FP16"
            if model_name == 'widedeep':
                onnx_model_path = "general_perf/model_zoo/regular/open_wide_deep_saved_model/widedeep_dynamicshape.onnx"
                engine_path = "general_perf/model_zoo/regular/open_wide_deep_saved_model/widedeep_dynamicshape" + ".engine"    
                build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize, BuildFlag='FP16')

            elif model_name == 'deberta':
                onnx_model_path = "general_perf/model_zoo/popular/open_deberta/deberta-sim-drop-clip-drop-invaild-cast_end.onnx"
                engine_path = "general_perf/model_zoo/popular/open_deberta/deberta-sim-drop-clip-drop-invaild-cast_end" + ".engine"    
                build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize, BuildFlag='FP16')

            elif model_name == 'roformer':
                onnx_model_path = "general_perf/model_zoo/popular/open_roformer/roformer-frozen_end.onnx"
                engine_path = "general_perf/model_zoo/popular/open_roformer/roformer-frozen_end" + ".engine"    
                build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize, BuildFlag='FP16')

            elif model_name == 'gpt2':
                for bs in configs['workload']['batch_sizes']:
                    onnx_model_path = os.path.dirname(model_path) + "/" + model + ".onnx"
                    engine_path = os.path.dirname(model_path) + "/" + model + "_bs" + str(bs) + ".so" 

                    for key, val in configs['model_info']['input_shape'].items():
                        input_dict = {}
                        val = val = [val[0] * bs] + val[1:] 
                        input_dict[key] = val
                        
                    build_igie_engine(model_name=model_name, model_path=onnx_model_path, input_dict=input_dict, model_framework='onnx', precision='fp16', engine_path=engine_path)
            
            elif model == 'vae-decoder-onnx-fp32' or model == 'vae-encoder-onnx-fp32' or model == 'clip-onnx-fp32':
                pass
            
            else:
                build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize, BuildFlag='FP16')
            
        if configs['model_info']['model_precision'] == 'INT8':
            precision_flag = "INT8"
            if model_name == 'widedeep':
                onnx_model_path = "general_perf/model_zoo/regular/open_wide_deep_saved_model/quantized_widedeep_staticshape.onnx"
                engine_path = "general_perf/model_zoo/regular/open_wide_deep_saved_model/quantized_widedeep_staticshape" + ".engine"    
                build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize, BuildFlag='INT8')
            
            if model_name == 'resnet50':
                onnx_model_path = "general_perf/model_zoo/regular/open_resnet50/quantized_Resnet50.onnx"
                engine_path = "general_perf/model_zoo/regular/open_resnet50/quantized_Resnet50" + ".engine"    
                build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize, BuildFlag='INT8')

            if model_name == 'yolov5':
                onnx_model_path = "general_perf/model_zoo/popular/open_yolov5/quantized_yolov5s.onnx"
                engine_path = "general_perf/model_zoo/popular/open_yolov5/quantized_yolov5s" + ".engine"    
                build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize, BuildFlag='INT8')

            if model_name == 'bert':
                print(f"\n==========****bert模型的int8精度推理采用直接加载engine文件, 因此不需要build engine! ****===========")

        result = {
            "model": 
                configs['model_info']['model'],
            "model_name": 
                configs['model_info']['model'].split("-")[0],
            "model_path":
                configs['model_info']['model_path'],
            "framework": 
                configs['model_info']['framework'],
            "compile_precision": 
                precision_flag,
            "input_type": 
                configs['model_info']['input_type'].split(","),
            "max_batch_size": 
                configs['model_info']['max_batch_size'],
            "compile_status":
                "success",
            "sg_percent": 100,
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": 
                        configs['model_info']['input_shape'],
                    "output_tensor_map": 
                        configs['model_info']['outputs'],
                    "compiled_model": [
                        {
                            "compiled_bs": 1,
                            "compiled_obj": configs['model_info']['model_path'],
                        },
                    ],
                },
            ],
        }

        self.configs = result
        self.workload = configs['workload']
        self.model_info = configs['model_info']

        for key, value in result.items():
            print('{key}: {value}'.format(key=key, value=value))

        return result


    def get_interact_profile(self, configs):
        """
            Collect information for core engine to let user interactively fill in configurations.
        """
        return []

    def get_best_batch_size(self):
        """Get Best Batch Size for the model.
        Usually take the max batch size can be loaded to IPU as the best batch size to
        get highest throughput.
        """
        return None
    
    def get_onnx(self, configs):
        model = configs['model_info']['model']
        model_name = configs['model_info']['model'].split("-")[0]
        model_path = configs['model_info']['model_path']

        # model save location
        if model_name == 'videobert' or model_name == 'conformer' or model_name == 'yolov5':
            onnx_model_path = model_path 

        elif model_name == 'widedeep' or model_name == 'roformer':
            onnx_model_path = model_path + "/" + model + ".onnx"

        else:
            onnx_model_path = os.path.dirname(model_path) + "/" + model + ".onnx"

        framework = configs['model_info']['framework']

        if framework == 'Pytorch':
            torch_to_onnx(model_path=model_path, output_path=onnx_model_path)
            print("***Convert pt model to onnx model success!***")

        if framework == 'Tensorflow':
            savedmodel_to_onnx(model_path=model_path, output_path=onnx_model_path)
            print("***Convert pb model to onnx model success!***")

        # Convert ONNX model to plugin operator model: Support fusion of dynamic and static graphs
        if model_name == 'bert' or model_name == 'albert' or model_name == 'roberta' or \
            model_name == 'videobert' or model_name == 'resnet50' or model_name == 'widedeep':
            
            cmd = f'python3 general_perf/backends/ILUVATAR/optimizer/optimizer.py --onnx {onnx_model_path}'
            subprocess.call(cmd, shell=True)
            print("***Convert onnx model to plugin operator model success!***")

        elif model_name == 'deberta':
            onnx_model_path = "general_perf/model_zoo/popular/open_deberta/deberta-sim-drop-clip-drop-invaild-cast.onnx"
            cmd = f'python3 general_perf/backends/ILUVATAR/optimizer/optimizer.py --onnx {onnx_model_path}'
            subprocess.call(cmd, shell=True)
            print("***Convert onnx model to plugin operator model success!***")

        elif model_name == 'swin':
            cmd = f'python3 general_perf/backends/ILUVATAR/optimizer/optimizer.py --onnx {onnx_model_path} --model_type swint'
            subprocess.call(cmd, shell=True)
            print("***Convert onnx model to plugin operator model success!***")

        elif model_name == 'yolov5':
            cmd = f'python3 general_perf/backends/ILUVATAR/optimizer/optimizer.py --onnx {onnx_model_path} --model_type yolo'
            subprocess.call(cmd, shell=True)
            print("***Convert onnx model to plugin operator model success!***")

        elif model_name == 'roformer':
            onnx_model_path = "general_perf/model_zoo/popular/open_roformer/roformer-frozen.onnx"
            cmd = f'python3 general_perf/backends/ILUVATAR/optimizer/optimizer.py --onnx {onnx_model_path} --model_type roformer --input_shapes input_segment0:bsx1024,input_token0:bsx1024'
            subprocess.call(cmd, shell=True)
            print("***Convert onnx model to plugin operator model success!***")

        elif model_name == 'conformer':
            cmd = f'python3 general_perf/backends/ILUVATAR/optimizer/optimizer.py --onnx {onnx_model_path} --model_type conformer --hidden_size 512 --num_heads 8'
            subprocess.call(cmd, shell=True)
            print("***Convert onnx model to plugin operator model success!***")

        else:
            pass