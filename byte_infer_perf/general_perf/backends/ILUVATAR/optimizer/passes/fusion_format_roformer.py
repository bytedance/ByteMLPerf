# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import Tuple, Union

import math
import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_options import AttentionMaskFormat
from .fusion_utils import FusionUtils, NumpyHelper
from .onnx_model import OnnxModel


logger = getLogger(__name__)

class FusionRemoveUselessElementwise(Fusion):
    """
    Fusion to remove useless elementwise in roformer model.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, "Sqrt", "Sqrt")

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        paths = {
            "path1" : (["Max", "Min", "Add", "GlobalAveragePool"], [None, None, None, None]),
        }

        pool_nodes, pool_path = self.match_parent_path_from_dict(node, paths)

        if pool_nodes is None:
            logger.debug("GlobalAveragePool: failed searching path after pool node.")
            return

        max_node = pool_nodes[0]
        min_node = pool_nodes[1]
        add_node = pool_nodes[2]
        pool_node = pool_nodes[3]
        if not self.model.has_constant_input(add_node, 9.999999960041972e-13):
            return

        if not self.model.has_constant_input(max_node, 0):
            return

        max_node.input[0] = pool_node.output[0]
        self.nodes_to_remove.extend([min_node, add_node])


class FusionFormatInvalidMask(Fusion):
    """
    Fusion to format invalid mask in roformer model.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, "Softmax", ["Softmax"])

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        nodes = self.model.match_parent_path(
            node,
            ["Add", "Mul"],
            [0, 1],
        )

        if nodes is None:
            logger.debug("Roformer: unable to format the mul.")
            return

        mul_node = nodes[1]

        inputs = mul_node.input
        outputs = mul_node.output

        coef0 = self.model.get_initializer(inputs[0])
        coef1 = self.model.get_initializer(inputs[1])
        if (coef0 and coef1) or (not coef0 and not coef1):
            return
        coef = coef0 if coef0 else coef1
        coef.CopyFrom(numpy_helper.from_array(np.array([-100.0]).astype(np.float32), coef.name))

        new_node = helper.make_node(
            "Mul",
            inputs = inputs,
            outputs = outputs,
            name = mul_node.name,
        )
        new_node.domain = "com.iluvatar"

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([mul_node])