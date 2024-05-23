# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from onnx import helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionRoPE(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "CustomRoPEPluginDynamic_IxRT", "Add")

    def fuse(self, start_node, input_name_to_nodes, output_name_to_node):
        src_paths = {"path1": (["Mul", "Concat", "Split", "Slice"], [0, 1, None, 0])}
        src_nodes, src_path = self.match_parent_path_from_dict(start_node, src_paths)
        if src_nodes is None:
            logger.debug("fuse_rope: failed to match src_node")
            return

        src_node = src_nodes[0]

        rotate_paths = {"path1": (["Mul", "Reshape", "Concat"], [1, 0, 0])}
        rotate_nodes, rotate_path = self.match_parent_path_from_dict(
            start_node, rotate_paths
        )

        if rotate_nodes is None:
            logger.debug("fuse_rope: failed to match rotate_path")
            return

        concat_node = rotate_nodes[-1]
        mul_right_node = rotate_nodes[0]

        odd_paths = {"path1": (["Unsqueeze", "Neg", "Slice", "Reshape"], [0, 0, 0, 0])}
        odd_nodes, odd_path = self.match_parent_path_from_dict(concat_node, odd_paths)

        if odd_nodes is None:
            logger.debug("fuse_rope: failed to match odd_path")
            return

        even_paths = {"path1": (["Unsqueeze", "Slice", "Reshape"], [1, 0, 0])}
        even_nodes, even_path = self.match_parent_path_from_dict(
            concat_node, even_paths
        )

        if even_nodes is None:
            logger.debug("fuse_rope: failed to match even_path")
            return
        reshape_node = even_nodes[-1]

        if reshape_node.output[0] == src_node.input[0]:
            rope_node_name = self.model.create_node_name("RoPE")
            rope_node = helper.make_node(
                "CustomRoPEPluginDynamic_IxRT",
                inputs=[
                    reshape_node.output[0],
                    src_nodes[0].input[1],
                    mul_right_node.input[1],
                ],
                outputs=[start_node.output[0]],
                name=rope_node_name,
            )
            rope_node.domain = "com.iluvatar"
            rope_node.attribute.extend([helper.make_attribute("type_id", 2)])
            rope_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
            rope_node.attribute.extend([helper.make_attribute("plugin_version", "1")])

            self.nodes_to_add.append(rope_node)
            self.node_name_to_graph_name[rope_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([start_node])
            self.nodes_to_remove.extend([src_nodes[0]])
            self.nodes_to_remove.extend(rotate_nodes)
            self.nodes_to_remove.extend(odd_nodes[:-1])
            self.nodes_to_remove.extend(even_nodes[:-1])
