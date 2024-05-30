# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import Tuple, Union

from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSplitQKV(Fusion):
    """
    Fuse FusionSplitQKV
    """

    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int):
        super().__init__(model, "SplitQKV_IxRT", "MatMul")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def create_splitqkv_node(
        self, input: str, query_out: str, key_out: str, value_out: str
    ) -> Union[NodeProto, None]:
        """Create an XSoftmax node.

        Args:
            data_input (str): data input name
            mask_input (str): max input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        node_name = self.model.create_node_name("SplitQKV_IxRT")

        new_node = helper.make_node(
            "SplitQKV_IxRT",
            inputs=[input],
            outputs=[query_out, key_out, value_out],
            name=node_name,
        )
        new_node.domain = "com.iluvatar"
        new_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        new_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        new_node.attribute.extend(
            [helper.make_attribute("atten_scale", 1 / self.num_heads)]
        )

        return new_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):

        split_query_paths = {
            "query_path": (
                ["Div", "Transpose", "Reshape", "Slice", "CustomFCPluginDynamic_IxRT"],
                [0, 0, 0, 0, 0],
            ),
        }

        split_key_paths = {
            "key_path": (["Transpose", "Reshape", "Slice"], [1, 0, 0]),
        }

        q_nodes, q_path = self.match_parent_path_from_dict(node, split_query_paths)

        k_nodes, k_path = self.match_parent_path_from_dict(node, split_key_paths)

        if (q_nodes is not None) and (k_nodes is not None):
            (
                q_div_node,
                q_transpose_node,
                q_reshape_node,
                q_slice_node,
                coustom_fc_node,
            ) = q_nodes
            k_transpose_node, k_reshape_node, k_slice_node = k_nodes
            slice_nodes = self.model.get_children(coustom_fc_node)

            if len(slice_nodes) != 3:
                return
            slice_nodes.remove(q_slice_node)
            slice_nodes.remove(k_slice_node)
            v_slice_node = slice_nodes[0]

            node.input[0] = q_div_node.input[0]  # dele div
            new_node = self.create_splitqkv_node(
                coustom_fc_node.output[0],
                q_slice_node.output[0],
                k_slice_node.output[0],
                v_slice_node.output[0],
            )

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name
            self.nodes_to_remove.append(q_slice_node)
            self.nodes_to_remove.append(k_slice_node)
            self.nodes_to_remove.append(v_slice_node)
            self.nodes_to_remove.append(q_div_node)

        else:
            return
