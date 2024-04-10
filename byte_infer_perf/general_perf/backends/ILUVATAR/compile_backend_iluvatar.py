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

import logging

import tensorrt
from tensorrt import Dims
from general_perf.backends.ILUVATAR.common import load_ixrt_plugin
load_ixrt_plugin()

from general_perf.backends.ILUVATAR.common import build_engine
from general_perf.tools import torch_to_onnx

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
        model_name = configs["model_info"]["model"].split("-")[0]
        MaxBatchSize = configs['model_info']['max_batch_size']
        onnx_model_path = configs['model_info']['onnx_model_path']
        engine_path = configs['model_info']['engine_path']

        # build engine
        if model_name == 'widedeep':
            for bs in configs['workload']['batch_sizes']:
                engine_paths = "general_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model/widedeep_dynamicshape_sim_" + str(bs) + ".engine"    
                build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_paths, MaxBatchSize=bs)
        else:   
            build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize)

        result = {
            "model": configs["model_info"]["model"],
            "engine_path": engine_path,
            "model_name": configs['model_info']["model"].split("-")[0],
            "framework": configs["model_info"]["framework"],
            "framework_iluvatar": configs["model_info"]["framework_iluvatar"],
            "compile_precision": configs['model_info']['model_precision'],
            "input_type": configs["model_info"]["input_type"].split(","),
            "max_batch_size": configs["model_info"]["max_batch_size"],
            "compile_status":"success",
            "sg_percent": 100,
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": configs["model_info"]["input_shape"],
                    "output_tensor_map": configs["model_info"]["outputs"],
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
        self.model_info = configs["model_info"]

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
    
    # 下面的两个函数待优化, 目前得到的onnx模型都是事先转换好的
    # to do
    def get_onnx(self, model_path, onnx_path):
        torch_to_onnx(model_path, onnx_path)
        
    def pre_optimize(self, configs):
        # todo: pt转onnx模型
        model_name = configs["model_info"]["model"].split("-")[0]

        if model_name == "resnet50":
            configs["model_info"]["model_path"] = "general_perf/general_perf/model_zoo/regular/open_resnet50/resnet50.onnx"

        elif model_name == "yolov5":
            configs["model_info"]["model_path"] = 'general_perf/general_perf/model_zoo/popular/open_yolov5/yolov5s_sim.onnx'

        return configs