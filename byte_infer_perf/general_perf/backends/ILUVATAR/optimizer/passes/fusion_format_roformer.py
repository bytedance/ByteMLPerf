# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import Tuple, Union

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
            "path1": (
                ["Max", "Min", "Add", "GlobalAveragePool"],
                [None, None, None, None],
            ),
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
        super().__init__(model, "", ["Greater"])

    def fuse(self, start_node, input_name_to_nodes, output_name_to_node):
        nodes = self.model.match_parent_path(
            start_node,
            [
                "ReduceMin",
                "Cast",
                "Concat",
                "Unsqueeze",
                "Greater",
                "ReduceMin",
                "Cast",
                "Concat",
                "Unsqueeze",
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

        if nodes is None:
            logger.debug("Roformer: unable to format the mask.")
            return

        unsqueeze_node = nodes[-1]

        for node in self.model.graph().node:
            for (id, input) in enumerate(node.input):
                if start_node.output[0] == input:
                    node.input[id] = unsqueeze_node.input[0]

        self.nodes_to_remove.extend(nodes)
        self.nodes_to_remove.extend([start_node])
