# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import List, Tuple, Union

from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionDisentangledAttention(Fusion):
    """
    Match Disentangled Attention
        -------------------------------------------
                                                  |
        GatherElements          -->   Add  -->   Add  -->
                                       |
        GatherElements --> Transpose  ->
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, "DisentangledAttention_IxRT", "Add")

    def create_disentangled_attention_node(
        self,
        inputs: List[str],
        outputs: List[str],
    ) -> Union[NodeProto, None]:
        """Create an disentangled attention node.

        Args:
            inputs List[str]: data input names
            outputs List[str]: data output names

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        disentangled_attention_node_name = self.model.create_node_name(
            "DisentangledAttention"
        )

        disentangled_attention_node = helper.make_node(
            "DisentangledAttention_IxRT",
            inputs=inputs,
            outputs=outputs,
            name=disentangled_attention_node_name,
        )
        disentangled_attention_node.domain = "com.iluvatar"
        disentangled_attention_node.attribute.extend(
            [helper.make_attribute("plugin_namespace", "")]
        )
        disentangled_attention_node.attribute.extend(
            [helper.make_attribute("plugin_version", "1")]
        )
        disentangled_attention_node.attribute.extend(
            [helper.make_attribute("factor", 0.1)]
        )
        disentangled_attention_node.attribute.extend(
            [helper.make_attribute("span", 512)]
        )

        return disentangled_attention_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):

        disentangled_attention_path1 = {
            "path": (["Add", "GatherElements", "MatMul"], [None, None, None]),
        }

        disentangled_attention_path2 = {
            "path": (
                ["Add", "Transpose", "GatherElements", "MatMul"],
                [None, None, None, None],
            ),
        }

        nodes1, _ = self.match_parent_path_from_dict(node, disentangled_attention_path1)
        nodes2, _ = self.match_parent_path_from_dict(node, disentangled_attention_path2)

        if nodes1 is not None and nodes2 is not None:
            if nodes1[0] == nodes2[0]:
                (head_add, first_gather, first_matmul) = nodes1
                (_, transpose, second_gather, second_matmul) = nodes2
                tail_add = node

                first_input = [i for i in tail_add.input if i != head_add.output[0]][0]
                second_input = first_matmul.output[0]
                third_input = second_matmul.output[0]
                output = tail_add.output[0]

                disentangled_attention_node = self.create_disentangled_attention_node(
                    [first_input, second_input, third_input], [output]
                )
                self.nodes_to_add.append(disentangled_attention_node)
                self.node_name_to_graph_name[
                    disentangled_attention_node.name
                ] = self.this_graph_name
                self.nodes_to_remove.append(tail_add)
                self.nodes_to_remove.append(head_add)
                self.nodes_to_remove.append(first_gather)
                self.nodes_to_remove.append(transpose)
                self.nodes_to_remove.append(second_gather)
