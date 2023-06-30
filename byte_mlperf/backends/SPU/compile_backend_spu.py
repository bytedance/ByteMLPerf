# Copyright 2023 ByteDance and/or its affiliates.
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
import json
import logging
import time
from typing import Any, Dict, List, Optional
from byte_mlperf.backends import compile_backend
from base_compile import Resnet50Builder, BertBaseBuilder, AlbertBuilder, RobertaBuilder, ConformerBuilder

log = logging.getLogger("CompileBackendSPU")


class CompileBackendSPU(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendSPU, self).__init__()
        self.hardware_type = "SPU"
        self.batch_size = None
        self.model_name = ""
        self.configs = None
        self.workload = None
        self.model_info = None
        self.model = None
        self.interact_info = None

    def compile(self,
                configs: Dict[str, Any],
                dataloader=None) -> Dict[str, Any]:
        """
        Model compilation interface. Model conversion and compilation 
        can be performed here. The model format can be changed here.
        """
        name = configs['model']
        builder_dict = {
            "resnet50-torch-fp32": Resnet50Builder,
            "bert-torch-fp32": BertBaseBuilder,
            "albert-torch-fp32": AlbertBuilder,
            "roberta-torch-fp32": RobertaBuilder,
            "conformer-encoder-onnx-fp32": ConformerBuilder
        }

        if name in builder_dict:
            sparser_builder = builder_dict[name]
        else:
            raise NotImplementedError(f"task: {name} not supported")

        self.interact_info = self.get_interact_profile(configs)
        onnx_path = self.interact_info["onnx_path"]
        dump_dir = self.interact_info["dump_dir"]
        dataset_dir = self.interact_info["dataset_dir"]
        dataset_cfg = self.interact_info["dataset_cfg"]
        dtype = self.interact_info["dtype"]
        batch_size = self.interact_info["batch_size"]
        verify = self.interact_info["verify"]

        start_time = time.time()
        builder = sparser_builder(onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify)
        result = builder()
        end_time = time.time()
        compilation_time = round(end_time - start_time, 2)
        assert isinstance(result, dict), "return is not dict"
        result.update({"compilation_time": compilation_time})
        return result

    def get_interact_profile(self, configs: Dict[str, Any]):
        model_profile = []
        file_path = f"backends/{configs['model']}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                model_profile = json.load(f)
            return model_profile
        else:
            log.info('File path: {} does not exist, please check'.format(file_path))
            raise NotImplementedError("CompileBackend:get_interact_profile")

    def get_best_batch_size(self) -> Optional[List[int]]:
        """
        Get Best Batch Size for the model
        """
        return [1]
