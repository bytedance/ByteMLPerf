# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import List, Tuple, Union

import numpy as np
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import FusionUtils, NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


def get_tensor_attr(attrs, attr_name):
    result = None
    for i in attrs:
        if i.name == attr_name:
            return numpy_helper.to_array(i.t)
    return result


class FusionYoloV5Decoder(Fusion):
    """
    Fuse SwinL subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, "YoloV5Decoder", ["Reshape"])

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        short_path = ["Concat", "Slice", "Sigmoid", "Transpose", "Reshape"]
        paths = [
            (["Concat", "Unsqueeze", "Gather", "Shape"], [1] + [None] * 3),
            (
                ["Concat", "Mul", "Add", "Sub", "Mul", "Slice", "Sigmoid", "Transpose"],
                [0, 0] + [None] * 6,
            ),
            (
                ["Concat", "Mul", "Pow", "Mul", "Slice", "Sigmoid", "Transpose"],
                [0, 1] + [None] * 5,
            ),
            (short_path, [None] * 5),
            (short_path + ["Concat", "Unsqueeze", "Gather", "Shape"], [None] * 9),
        ]
        paths_found = []
        nodes_names_found = set()
        nodes_found = []
        for path_i in paths:
            nodes = self.model.match_parent_path(normalize_node, path_i[0], path_i[1])
            paths_found.append(nodes)
            if nodes:
                for n in nodes:
                    if n.name not in nodes_names_found:
                        nodes_names_found.add(n.name)
                        nodes_found.append(n)
        if not all(paths_found):
            return
        shape_node = paths_found[-1][-1]
        params = self._find_yolov5_decoder_params(paths_found)
        self._fuse_node(
            inputs=shape_node.input, outputs=normalize_node.output, params=params
        )
        self.nodes_to_remove.extend(nodes_found)
        self._delete_extra_output_edges(paths_found)
        self.prune_graph = True

    def _fuse_node(self, inputs, outputs, params):
        fused_node = helper.make_node(
            "YoloV5Decoder",
            inputs=inputs,
            outputs=outputs,
            name=self.model.create_node_name("YoloV5Decoder"),
        )
        fused_node.attribute.extend(params)
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name

    def _delete_extra_output_edges(self, paths_found):
        transpose_node = paths_found[2][-1]
        assert transpose_node.op_type == "Transpose"
        out_edge = transpose_node.output[0]
        for item in self.model.graph().output:
            if item.name == out_edge:
                self.model.graph().output.remove(item)
                logger.warning(f"Output: {out_edge} is useless in graph, delete it")
                return

    def _find_yolov5_decoder_params(self, paths_found):
        # num_class
        concat_op = paths_found[0][0]
        assert concat_op.op_type == "Concat"
        num_class_arr = self.model.get_initializer(concat_op.input[2], True)
        assert num_class_arr
        num_class = (num_class_arr - 5).tolist()[0]
        num_class = helper.make_attribute("num_class", num_class)

        # stride
        mul_op = paths_found[1][1]
        assert mul_op.op_type == "Mul"
        input_arrs = self.model.get_initializer_input_edges(mul_op.name, True)
        assert len(input_arrs) == 1
        stride = input_arrs[0].tolist()
        stride = helper.make_attribute("stride", stride)

        # anchor
        mul_op = paths_found[2][1]
        assert mul_op.op_type == "Mul"
        anchor = self.model.get_initializer_input_edges(mul_op.name, True)
        assert len(anchor) == 1
        anchor = anchor[0]
        anchor = anchor[0, :, 0, 0, :] if len(anchor.shape) == 5 else anchor[:, 0, 0, :]
        anchor = helper.make_attribute("anchor", list(anchor.flatten()))

        # fast_impl
        fast_impl = helper.make_attribute("faster_impl", 1)

        return [num_class, stride, anchor, fast_impl]
