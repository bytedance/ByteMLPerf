# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import Tuple, Union

import numpy as np
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionConformerXSoftmax(Fusion):
    """
    Fuse Where + Softmax + Where into one node: XSoftmax
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, "XSoftmax_IxRT", "Softmax")

    def create_xsoftmax_node(
        self, data_input: str, mask_input: str, output: str
    ) -> Union[NodeProto, None]:
        """Create an XSoftmax node.

        Args:
            data_input (str): data input name
            mask_input (str): max input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """

        unique_index = data_input
        new_edge = "edge_modified_" + unique_index
        shape_tensor = helper.make_tensor(
            name="shape_modified_tensor_" + unique_index,
            data_type=TensorProto.INT64,
            dims=[4],
            vals=np.int64(
                [-1, 8, 128, 128]  # (BSZ, HEAD_NUM, SEQ_LEN, SEQ_LEN)
            ).tobytes(),
            raw=True,
        )
        self.model.add_initializer(shape_tensor, self.this_graph_name)
        self.model.add_node(
            helper.make_node(
                "Reshape",
                [data_input, shape_tensor.name],
                [new_edge],
                "reshape_modified_" + unique_index,
            ),
            self.this_graph_name,
        )

        new_edge2 = "edge_modified2_" + unique_index
        xsoftmax_node_name = self.model.create_node_name("XSoftmax")

        xsoftmax_node = helper.make_node(
            "XSoftmax_IxRT",
            inputs=[new_edge, mask_input],
            outputs=[new_edge2],
            name=xsoftmax_node_name,
        )
        xsoftmax_node.domain = "com.iluvatar"
        xsoftmax_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        xsoftmax_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        xsoftmax_node.attribute.extend([helper.make_attribute("type_id", 2)])
        xsoftmax_node.attribute.extend([helper.make_attribute("dim", -1)])
        xsoftmax_node.attribute.extend([helper.make_attribute("is_conformer", 1)])

        shape_tensor2 = helper.make_tensor(
            name="shape_modified_tensor2_" + unique_index,
            data_type=TensorProto.INT64,
            dims=[3],
            vals=np.int64(
                [-1, 128, 128]  # (BSZ, HEAD_NUM, SEQ_LEN, SEQ_LEN)
            ).tobytes(),
            raw=True,
        )
        self.model.add_initializer(shape_tensor2, self.this_graph_name)
        self.model.add_node(
            helper.make_node(
                "Reshape",
                [new_edge2, shape_tensor2.name],
                [output],
                "reshape_modified2_" + unique_index,
            ),
            self.this_graph_name,
        )

        return xsoftmax_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):

        xsoftmax_paths = {
            "path": (["Add", "Where", "Reshape", "Expand"], [None, None, None, None]),
        }
        xsoftmax_nodes, xsoftmax_path = self.match_parent_path_from_dict(
            node, xsoftmax_paths
        )

        if xsoftmax_nodes is None:
            logger.debug("fuse_xsoftmax: failed to match xsoftmax path")
            return
        else:
            (add_node, where_node, reshape_node, expand_node) = xsoftmax_nodes

            mask_input = expand_node.input[0]

            data_output = node.output[0]

            data_input = add_node.input[0]
            if where_node.output[0] == add_node.input[0]:
                data_input = add_node.input[1]
            xsoftmax_node = self.create_xsoftmax_node(
                data_input, mask_input, data_output
            )

            self.nodes_to_remove.extend(xsoftmax_nodes)
            self.nodes_to_add.append(xsoftmax_node)
            self.node_name_to_graph_name[xsoftmax_node.name] = self.this_graph_name
