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


class FusionXSoftmax(Fusion):
    """
    Fuse Where + Softmax + Where into one node: XSoftmax
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, "XSoftmax_IxRT", "MatMul")

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
        xsoftmax_node_name = self.model.create_node_name("XSoftmax")

        xsoftmax_node = helper.make_node(
            "XSoftmax_IxRT",
            inputs=[data_input, mask_input],
            outputs=[output],
            name=xsoftmax_node_name,
        )
        xsoftmax_node.domain = "com.iluvatar"
        xsoftmax_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        xsoftmax_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        xsoftmax_node.attribute.extend([helper.make_attribute("type_id", 2)])
        xsoftmax_node.attribute.extend([helper.make_attribute("dim", -1)])

        return xsoftmax_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):

        xsoftmax_paths = {
            "path": (["Where", "Softmax", "Where", "Add"], [None, None, None, None]),
        }
        xsoftmax_nodes, xsoftmax_path = self.match_parent_path_from_dict(
            node, xsoftmax_paths
        )

        if xsoftmax_nodes is None:
            logger.debug("fuse_xsoftmax: failed to match xsoftmax path")
            return
        else:
            (tail_where, softmax, head_where, add) = xsoftmax_nodes
            where_inputs = [i for i in tail_where.input if i in head_where.input]
            assert len(where_inputs) == 1
            mask_input = where_inputs[0]
            data_input = add.output[0]
            data_output = tail_where.output[0]

            xsoftmax_node = self.create_xsoftmax_node(
                data_input, mask_input, data_output
            )

            self.nodes_to_add.append(xsoftmax_node)
            self.node_name_to_graph_name[xsoftmax_node.name] = self.this_graph_name
            self.nodes_to_remove.append(tail_where)
            self.nodes_to_remove.append(softmax)
            self.nodes_to_remove.append(head_where)
