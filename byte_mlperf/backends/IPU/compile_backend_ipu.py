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

import copy
import json
import logging
import os
from pathlib import Path
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
        self.packrunner = False

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

        if self.interact_info.get("pack_config"):
            self.packrunner = True

        model_info = configs["model_info"]
        model_type = model_info["model_format"]
        model_name = model_info["model"]

        pre_optimized_root = Path(self.current_dir) / "pre_optimized_models"
        if not pre_optimized_root.exists():
            pre_optimized_root.mkdir(parents=True)

        model_path = os.path.abspath(configs["model_info"]["model_path"])
        onnx_path = pre_optimized_root / (model_name + ".onnx")

        if model_type != "onnx":
            if onnx_path.exists():
                if self.packrunner:
                    self._update_pack_model(onnx_path, model_info)
                else:
                    model_info["model_path"] = onnx_path
                log.info("{} file exists, skip ONNX conversion".format(onnx_path.name))
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
                    torch_to_onnx.torch_to_onnx(model_path, str(onnx_path))
                else:
                    log.error(
                        "Wrong model type: {}, which must be saved_model, pt, or onnx".format(
                            model_type
                        )
                    )
                    raise TypeError("Model type must be saved_model, pt, or onnx")

                if os.path.exists(onnx_path):
                    model_info["model_path"] = onnx_path
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
        if self.interact_info.get("pack_config"):
            self.packrunner = True

        log.info("The interaction info is:\n {}".format(self.interact_info))

        precision = (
            self.interact_info.get("converter_options", {})
            .get("precision", "FP32")
            .upper()
        )

        result = {
            "model": config["model_info"]["model"],
            "framework": config["model_info"]["framework"],
            "compile_precision": precision,
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

        # packrunner takes single data as input and perform batching asynchoroly
        # within itself.
        # The config option "pack_bs" is the bs for model compiling, and the regular
        # "batch_size" is used to specify bs for the dataloader/inferencing and
        # it should always be 1 since pack runner only take single sample a time
        if self.packrunner:
            pack_config = self.interact_info["pack_config"]
            assert (
                "batch_size" in pack_config
            ), "for pack mode, we acquire pack_bs as the actual compile batch sizes for the pack model"
            assert isinstance(
                pack_config["batch_size"], int
            ), "pack_bs should be a positive integers"

            compile_bs = [pack_config["batch_size"]]
        else:
            compile_bs = config["workload"]["batch_sizes"]

        for batch_size in compile_bs:
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
        """Get Best Batch Size for the model.

        Usually take the max batch size can be loaded to IPU as the best batch size to
        get highest throughput.
        """
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
            elif name == "text" and 'videobert' in self.model_info['model']:
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
        options.enable_outlining = compiler_options.get("enable_outlining", True)
        options.outline_threshold = compiler_options.get("outline_threshold", 1.0)

        outputs = [o.name for o in converted_model.graph.output]
        Compiler.compile_and_export(
            converted_model.SerializeToString(), outputs, popef_path, options
        )

        return popef_path

    def _update_pack_model(self, input_model_path, model_info):
        """bert like model conversion for pack mode, update corresponded configs as well."""
        model = onnx.load(input_model_path)

        save_path = (
            Path(self.current_dir)
            / "pre_optimized_models"
            / (Path(input_model_path).stem + "_pack.onnx")
        )
        assert "input_shape" in model_info
        assert "inputs" in model_info
        assert "dataset_name" in model_info
        assert "input_type" in model_info

        model_info["inputs"] += ",position_ids"
        model_info["input_type"] += ",LONG"
        model_info["input_shape"]["position_ids"] = [1, 384]
        model_info["model_path"] = str(save_path)

        self.model_info = model_info

        if self.model_info["model"] == "roberta-torch-fp32":
            rm_node_names = [
                "Equal_8",
                "Not_9",
                "Cast_10",
                "CumSum_12",
                "Add_14",
                "Mul_15",
                "Cast_16",
            ]
            rm_nodes = []
            for node in model.graph.node:
                if node.name in rm_node_names:
                    rm_nodes.append(node)

            assert len(rm_node_names) == len(rm_nodes)

            for node in rm_nodes:
                model.graph.node.remove(node)

            position_ids = copy.deepcopy(model.graph.input[0])
            position_ids.name = "position_ids"
            model.graph.input.append(position_ids)

            for node in model.graph.node:
                if node.op_type == "Add" and node.name == "Add_18":
                    node.input[0] = position_ids.name
        elif self.model_info["model"] in ("bert-torch-fp32", "albert-torch-fp32"):
            # for packed bert, we need to export position_ids to model's input
            # step 1: remove unneed node
            rm_node_names = [
                "Shape_7",
                "Gather_9",
                "Add_11",
                "Unsqueeze_12",
                "Slice_14",
                "Constant_8",
                "Constant_10",
                "Constant_13",
            ]
            rm_nodes = []
            for node in model.graph.node:
                if node.name in rm_node_names:
                    rm_nodes.append(node)

            assert len(rm_node_names) == len(rm_nodes)

            for node in rm_nodes:
                model.graph.node.remove(node)

            # step 2: add position_ids to model's input
            position_ids = copy.deepcopy(model.graph.input[0])
            position_ids.name = "position_ids"
            model.graph.input.append(position_ids)
            # step 3: rename input.1 to token_type_ids.1
            for i, input in enumerate(model.graph.input):
                if input.name == "input.1":
                    model.graph.input[i].name = "token_type_ids.1"
                    break

            for node in model.graph.node:
                if node.op_type == "Gather" and node.name == "Gather_18":
                    node.input[1] = position_ids.name
                if node.op_type == "Gather" and node.name == "Gather_16":
                    node.input[1] = "token_type_ids.1"

        print("Save preprocessed model to {}".format(save_path))
        onnx.save(model, save_path)

        # return model_info
