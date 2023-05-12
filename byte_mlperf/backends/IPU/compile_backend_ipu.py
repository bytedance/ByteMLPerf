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

import json
import logging
import os
import subprocess
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
        onnx_path = os.path.join(
            self.current_dir, "pre_optimized_models", model_name + ".onnx"
        )
        if model_type != "onnx":
            # check and download converted onnx if existing
            subprocess.call(
                [
                    "bash",
                    os.path.join(self.current_dir, "get_converted_onnx.sh"),
                    model_name,
                ]
            )
            if os.path.exists(onnx_path):
                configs["model_info"]["model_path"] = onnx_path
                log.info("{} file exists, skip ONNX conversion".format(onnx_path))
            else:
                # convert the model to onnx
                log.info(
                    "Convert the model: {} from format: {} to onnx".format(
                        model_name, model_type
                    )
                )
                if model_type == "saved_model":
                    saved_to_onnx.savedmodel_to_onnx(model_path, onnx_path)
                elif model_type == "pt":
                    torch_to_onnx.torch_to_onnx(model_path, onnx_path)
                else:
                    log.error(
                        "Wrong model type: {}, which must be saved_model, pt, or onnx".format(
                            model_type
                        )
                    )
                    raise TypeError("Model type must be saved_model, pt, or onnx")

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

        if not self.interact_info:
            interact_info = configs.get("interact_info", {})
            self.interact_info = {}
            self.interact_info["converter_options"] = interact_info.get(
                "converter_options", {}
            )
            self.interact_info["clients"] = int(interact_info.get("clients", "1"))
            batch_sizes = interact_info.get("batch_sizes", "").split(",").remove("")
            if batch_sizes:
                self.interact_info["batch_sizes"] = [
                    int(x.strip()) for x in batch_sizes
                ]
            self.interact_info["compiler_options"] = json.loads(
                interact_info.get("compiler_options", "{}")
            )

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
        interact_info_file = os.path.join(
            self.current_dir, "interact_infos", config["model_info"]["model"] + ".json"
        )
        if os.path.exists(interact_info_file):
            with open(interact_info_file, "r") as f:
                self.interact_info = json.load(f)
                log.info("interact_info set by file: {}".format(interact_info_file))
        else:
            file_path = os.path.join(self.current_dir, self.hardware_type + ".json")
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
        self.popef_path = os.path.join(
            self.current_dir,
            "compiled_models",
            self.model_info["model"],
            str(batch_size),
            "executable.popef",
        )
        self.popef_path = os.path.abspath(self.popef_path)
        if os.path.exists(self.popef_path):
            log.info(
                "The PopEF file {} already exist, skip compile".format(
                    os.path.abspath(self.popef_path)
                )
            )
            return self.popef_path

        log.info("Create the directory {}".format(os.path.dirname(self.popef_path)))
        os.makedirs(os.path.dirname(self.popef_path), exist_ok=True)

        converter_options = self.interact_info.get("converter_options", {})
        compiler_options = self.interact_info.get("compiler_options", {})

        converted_model = self._convert(converter_options)
        self._poprt_compile(converted_model, compiler_options, self.popef_path)

        return self.popef_path

    def _convert(self, converter_options: Dict) -> onnx.ModelProto:
        model_proto = onnx.load(self.model_info["model_path"])

        input_shape = {}
        not_extended_with_batch = self.interact_info.get("not_extended_with_batch", [])
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
        converter_options["input_shape"] = input_shape

        converter = Converter(**converter_options)
        converted_model = converter.convert(model_proto)

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
