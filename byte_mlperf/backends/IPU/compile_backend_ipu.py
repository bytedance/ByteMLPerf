# Copyright 2023 ByteDance and/or its affiliates.
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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

import json
import logging
import os
from typing import Any, Dict

import onnx
import poprt
from poprt import runtime
from poprt.compiler import Compiler, CompilerOptions
from poprt.converter import Converter
from tools import saved_to_onnx, torch_to_onnx

from byte_mlperf.backends import compile_backend

log = logging.getLogger("CompileBackendIPU")


class CompileBackendIPU(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendIPU, self).__init__()
        self.hardware_type = "IPU"
        self.need_reload = False
        self.model_runtimes = []
        self.current_dir = os.path.split(os.path.abspath(__file__))[0]
        self.interact_info = None

    def version(self) -> str:
        """Return compile backend version details."""
        return poprt.__version__

    def pre_optimize(self, configs: Dict[str, Any]):
        """Model pre-optimization interface.

        Requirements: Model pre-optimization
        cannot change the model format. Torch model export to ONNX is allowed.
        """
        # convert model to onnx if it's not
        # configs['workload'] is the content of workloads/<task_name>.json and
        # configs['model_info'] is content of model_zoo/<task_name>.json
        model_type = configs["model_info"]["model_format"]
        model_name = configs["model_info"]["model"]
        model_path = os.path.abspath(configs["model_info"]["model_path"])
        onnx_path = self.current_dir + "/pre_optimized_models/" + model_name + ".onnx"
        if model_type != "onnx":
            if os.path.exists(onnx_path):
                configs["model_info"]["model_path"] = onnx_path
                log.info("{} file exists, skip ONNX conversion".format(onnx_path))
            else:
                # conver the model not onnx
                log.info(
                    "Convert the model: {} from format: {} to onnx".format(
                        model_name, model_type
                    )
                )
                if model_type == "saved_model":
                    saved_to_onnx.savedmodel_to_onnx(model_path, onnx_path)
                elif model_type == "pt":
                    torch_to_onnx.torch_to_onnx(model_path, onnx_path)

                if os.path.exists(onnx_path):
                    configs["model_info"]["model_path"] = onnx_path
                    log.info(
                        "Converted the model: {} from format: {} to onnx".format(
                            model_name, model_type
                        )
                    )
                else:
                    log.error(
                        "{} not exists, failed to convert the model: {} to onnx".format(
                            onnx_path, model_name
                        )
                    )
                    raise RuntimeError("Failed to convert model to onnx")
        else:
            log.info("{} is onnx model, skip ONNX conversion".format(model_name))

        return configs

    def compile(self, config, dataloader=None):
        self.model_info = config["model_info"]
        if not self.interact_info:
            self.interact_info = config["interact_info"]
        log.info("The interaction info is:\n {}".format(self.interact_info))

        result = {
            "model": config["model_info"]["model"],
            "framework": config["model_info"]["framework"],
            "compile_precision": config["model_info"]["model_precision"],
            "input_type": config["model_info"]["input_type"].split(","),
            "max_batch_size": config["model_info"]["max_batch_size"],
            "compile_status": "success",
            "sg_percent": 100,
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": config["model_info"]["input_shape"],
                    "output_tensor_map": config["model_info"]["outputs"],
                    "compiled_model": [
                        {
                            "compiled_bs": 1,
                            "compiled_obj": config["model_info"]["model_path"],
                        },
                    ],
                },
            ],
            "interact_info": self.interact_info,
        }

        for batch_size in config["workload"]["batch_sizes"]:
            self._compile(batch_size)

        return result

    def get_interact_profile(self, config):
        model_profile = []
        # load the interact_info by model name
        interact_info_file = (
            self.current_dir
            + "/interact_infos/"
            + config["model_info"]["model"]
            + ".json"
        )
        if os.path.exists(interact_info_file):
            with open(interact_info_file, "r") as f:
                self.interact_info = json.load(f)
                log.info("interact_info set by file: {}".format(interact_info_file))
        else:
            file_path = self.current_dir + "/" + self.hardware_type + ".json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    model_profile = json.load(f)
            else:
                log.info("File path: {} does not exist, please check".format(file_path))

        return model_profile

    def get_best_batch_size(self):
        """Get Best Batch Size for the model."""
        return self.interact_info.get("batch_sizes", None)

    def _compile(self, batch_size):
        self.batch_size = batch_size
        self.popef_path = (
            self.current_dir
            + "/compiled_models/"
            + self.model_info["model"]
            + "/"
            + str(batch_size)
            + "/executable.popef"
        )
        self.popef_path = os.path.abspath(self.popef_path)
        if os.path.exists(self.popef_path):
            log.info(
                "The PopEF file {} already exist, skip compile".format(
                    os.path.abspath(self.popef_path)
                )
            )
            return self.popef_path

        log.info("Create the directory {}".format(os.path.split(self.popef_path)[0]))
        os.makedirs(os.path.split(self.popef_path)[0], exist_ok=True)

        converted_model = self._convert()
        compiler_options = self.interact_info.get("compiler_options", {})
        self._poprt_compile(converted_model, compiler_options, self.popef_path)

        return self.popef_path

    def _convert(self) -> onnx.ModelProto:
        model_proto = onnx.load(self.model_info["model_path"])
        args = {"convert_version": 11}
        if self.interact_info.get("is_shape_inference", True):
            input_shape = {}
            not_extended_with_batch = self.interact_info.get(
                "not_extended_with_batch", []
            )
            for name, shape in self.model_info["input_shape"].items():
                if name in not_extended_with_batch:
                    batched_shape = [shape[0]] + shape[1:]
                else:
                    batched_shape = [shape[0] * self.batch_size] + shape[1:]
                log.info(
                    "The model input {} with shape {} in the model information, and shape with batch size is {}.".format(
                        name, shape, batched_shape
                    )
                )
                input_shape[name] = batched_shape
            args["input_shape"] = input_shape

        if self.interact_info.get("is_forced_fp16", True):
            args["precision"] = "fp16"

        if self.interact_info.get("customized_model_passes", None):
            passes = self.interact_info["customized_model_passes"].split(",")
            args["used_passes"] = passes

        args["disable_fast_norm"] = self.interact_info.get("disable_fast_norm", False)
        args["enable_insert_remap"] = self.interact_info.get(
            "enable_insert_remap", True
        )

        converter = Converter(**args)
        converted_model = converter.convert(model_proto)
        # Add other passes here
        if self.interact_info.get("custom_pass_config", None):
            custom_pass_name = self.interact_info["custom_pass_config"]
            assert custom_pass_name in poprt.get_registered_passes()
            converted_model = poprt.Pass.get_pass(custom_pass_name)(converted_model)
        converted_model = poprt.Pass.get_pass("int64_to_int32")(converted_model)
        converted_model = poprt.Pass.get_pass("gelu_pattern")(converted_model)

        return converted_model

    def _poprt_compile(
        self, converted_model: onnx.ModelProto, compiler_options: dict, popef_path: str
    ):
        options = CompilerOptions()
        options.ipu_version = runtime.DeviceManager().ipu_hardware_version()

        options.num_io_tiles = compiler_options.get("num_iotiles", 0)
        options.batches_per_step = compiler_options.get("batches_per_step", 1)
        options.enable_prefetch_datastreams = compiler_options.get(
            "enable_prefetch_datastreams", False
        )
        options.stream_buffering_depth = compiler_options.get(
            "stream_buffering_depth", 1
        )
        options.available_memory_proportion = compiler_options.get(
            "available_memory_proportion", 0.6
        )
        options.partials_type = compiler_options.get("partials_type", "half")
        options.use_128bit_conv_unit_load = compiler_options.get(
            "use_128bit_conv_unit_load", False
        )
        options.enable_fast_reduce = compiler_options.get("enable_fast_reduce", False)
        options.group_host_sync = compiler_options.get("group_host_sync", False)
        options.rearrange_anchors_on_host = compiler_options.get(
            "rearrange_anchors_on_host", False
        )

        outputs = [o.name for o in converted_model.graph.output]
        Compiler.compile_and_export(
            converted_model.SerializeToString(), outputs, popef_path, options
        )

        return popef_path
