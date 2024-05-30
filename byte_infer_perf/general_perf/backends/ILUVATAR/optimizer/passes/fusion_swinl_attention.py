# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import Tuple, Union, List

import numpy as np
from .fusion_base import Fusion
from .fusion_options import AttentionMaskFormat
from .fusion_utils import FusionUtils, NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from .onnx_model import OnnxModel
from .shape_infer_helper import SymbolicShapeInferenceHelper, get_shape_from_type_proto
import onnx

logger = getLogger(__name__)


def get_tensor_attr(attrs, attr_name):
    result = None
    for i in attrs:
        if i.name == attr_name:
            return numpy_helper.to_array(i.t)
    return result


class FusionSwinLAttention(Fusion):
    """
    Fuse SwinL subgraph into one Attention node.
    """

    def __init__(
            self,
            model: OnnxModel,
    ):
        super().__init__(model, "CustomQKVToContextPluginDynamic_IxRT", ["CustomFCPluginDynamic_IxRT"])

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size(self, reshape_v: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        v_shape = self.model.get_initializer(reshape_v.input[1])
        if v_shape is None:
            logger.debug(f"{reshape_v.input[1]} is not initializer.")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        v_shape_value = NumpyHelper.to_array(v_shape)
        if len(v_shape_value) != 3 or (v_shape_value[1] <= 0 or v_shape_value[2] <= 0):
            logger.debug(f"v_shape_value={v_shape_value}. Expected value are like [0, 0, num_heads, head_size].")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = 1
        for value_info in self.model.graph().value_info:
            if value_info.name == reshape_v.input[0]:
                num_heads = value_info.type.tensor_type.shape.dim[2].dim_value
                break
        hidden_size = v_shape_value[2]

        return num_heads, hidden_size

    def create_attention_node(
            self,
            num_heads: int,
            hidden_size: int,
            inputs: List[str],
            output: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        attention_node_name = self.model.create_node_name("Attention")

        attention_node = helper.make_node(
            "CustomQKVToContextPluginDynamic_IxRT",
            inputs=inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.iluvatar"
        attention_node.attribute.extend([helper.make_attribute("type_id", 2)])
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        attention_node.attribute.extend([helper.make_attribute("hidden_size", hidden_size)])
        attention_node.attribute.extend([helper.make_attribute("has_mask", 1)])
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        attention_node.attribute.extend([helper.make_attribute("has_qk_bias", 1)])
        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        self.fuse_pattern1(normalize_node, input_name_to_nodes, output_name_to_node)
        self.fuse_pattern2(normalize_node, input_name_to_nodes, output_name_to_node)

    def fuse_pattern2(self, normalize_node, input_name_to_nodes, output_name_to_node):
        """ match Swin-L pattern and fuse them to CustomFC --> Attention --> CustomFC
         """
        logger.debug("fuse swin-L attention pass")
        # 1. CustomFCPluginDynamic_IxRT node as start, go up to find a pattern for swin-L pattern
        start_node = normalize_node
        qkv_paths = {
            "path1": (["Reshape", "Transpose", "MatMul"], [0, 0, 0]),
        }
        qkv_nodes, qkv_path = self.match_parent_path_from_dict(start_node, qkv_paths)
        if qkv_nodes is None:
            logger.debug("fuse_attention: failed to match qkv path")
            return
        assert qkv_path == 'path1', 'abnormal qkv path'
        reshape_qkv, transpose_qkv, matmul_qkv = qkv_nodes

        # 2. MatMul as start, go up to find v path
        v_paths = {
            "path1": (["Transpose", "Reshape", "CustomFCPluginDynamic_IxRT"], [None, 0, 0])
        }
        v_nodes, v_path = self.match_parent_path_from_dict(matmul_qkv, v_paths)
        if not v_nodes:
            logger.debug("fuse_attention: failed to match v path")
            return
        assert v_path == 'path1', 'abnormal v path'

        # 3. MatMul as start, go up to find q,k paths
        # q path
        q_paths = {
            "path1": (["Softmax", "Add", "Div", "MatMul", "Transpose", "Reshape", "CustomFCPluginDynamic_IxRT"],
                      [None, 0, 0, 0, 0, 0, 0]),
        }
        q_nodes, q_path = self.match_parent_path_from_dict(matmul_qkv, q_paths)
        if not q_nodes:
            logger.debug("fuse_attention: failed to match q path")
            return
        assert q_path == 'path1', 'abnormal q paths found'

        # get Add(bias) input name as fused Attention inputs
        add_op, div_op = q_nodes[1], q_nodes[2]
        relative_position_bias_name = add_op.input[1] if add_op.input[0] == div_op.output[0] else add_op.input[0]

        # k path
        k_paths = {
            "path2": (["Softmax", "Add", "Div", "MatMul", "Transpose", "Reshape", "CustomFCPluginDynamic_IxRT"],
                      [None, 0, 0, 0, 1, 0, 0])
        }
        k_nodes, k_path = self.match_parent_path_from_dict(matmul_qkv, k_paths)
        if not k_nodes:
            logger.debug("fuse_attention: failed to match k path")
            return
        assert k_path == 'path2', 'abnormal k paths found'
        # 4. Fuse 3 CustomFC into one, and fuse attention
        # Fuse FCs
        fc_nodes = [q_nodes[-1], k_nodes[-1], v_nodes[-1]]
        weight = self.fuse_tensor_in_node_attrs(fc_nodes, "W", q_nodes[-1].name + "_Weight")
        bias = self.fuse_tensor_in_node_attrs(fc_nodes, "B", q_nodes[-1].name + "_Bias")
        fused_node = helper.make_node(
            "CustomFCPluginDynamic_IxRT",
            inputs=[q_nodes[-1].input[0]],
            outputs=q_nodes[-1].output,
            name=self.model.create_node_name("CustomFC", "MatMul_AddBias_"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("out_dims", numpy_helper.to_array(bias).shape[0])])
        fused_node.attribute.extend([helper.make_attribute("type_id", 2)])
        fused_node.attribute.extend([helper.make_attribute("W", weight)])
        fused_node.attribute.extend([helper.make_attribute("B", bias)])
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("act_type", -1)])
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)

        # Fuse Attention
        num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_qkv)
        attention_node = self.create_attention_node(
            num_heads,
            hidden_size,
            [fused_node.output[0], relative_position_bias_name],
            reshape_qkv.output[0],
        )
        if not attention_node:
            return
        self.nodes_to_add.append(attention_node)
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.nodes_to_remove.extend([*qkv_nodes, *q_nodes[:-2], *k_nodes[:-2], *v_nodes])
        self.prune_graph = True

    def fuse_pattern1(self, normalize_node, input_name_to_nodes, output_name_to_node):
        """ match Swin-L pattern and fuse them to CustomFC --> Attention --> CustomFC
        """
        logger.debug("fuse swin-L attention pass")
        # 1. CustomFCPluginDynamic_IxRT node as start, go up to find a pattern for swin-L pattern
        start_node = normalize_node
        qkv_paths = {
            "path1": (["Reshape", "Transpose", "MatMul"], [0, 0, 0]),
        }
        qkv_nodes, qkv_path = self.match_parent_path_from_dict(start_node, qkv_paths)
        if qkv_nodes is None:
            logger.debug("fuse_attention: failed to match qkv path")
            return
        assert qkv_path == 'path1', 'abnormal qkv path'
        reshape_qkv, transpose_qkv, matmul_qkv = qkv_nodes

        # 2. MatMul as start, go up to find v path
        v_paths = {
            "path1": (["Transpose", "Reshape", "Add", "Split", "MatMul"], [None, 0, 0, None, 0])
        }
        v_nodes, v_path = self.match_parent_path_from_dict(matmul_qkv, v_paths)
        if not v_nodes:
            logger.debug("fuse_attention: failed to match v path")
            return
        assert v_path == 'path1', 'abnormal v path'

        # 3. MatMul as start, go up to find q,k paths
        # q path
        q_paths = {
            "path1": (["Softmax", "Add", "Div", "MatMul", "Transpose", "Reshape", "Add", "Split", "MatMul"],
                      [None, 0, 0, 0, 0, 0, 0, None, 0]),
        }
        q_nodes, q_path = self.match_parent_path_from_dict(matmul_qkv, q_paths)
        if not q_nodes:
            logger.debug("fuse_attention: failed to match q path")
            return
        assert q_path == 'path1', 'abnormal q paths found'

        # get Add(bias) input name as fused Attention inputs
        add_op, div_op = q_nodes[1], q_nodes[2]
        relative_position_bias_name = add_op.input[1] if add_op.input[0] == div_op.output[0] else add_op.input[0]

        # k path
        k_paths = {
            "path2": (["Softmax", "Add", "Div", "MatMul", "Transpose", "Reshape", "Add", "Split", "MatMul"],
                      [None, 0, 0, 0, 1, 0, 0, None, 0])
        }
        k_nodes, k_path = self.match_parent_path_from_dict(matmul_qkv, k_paths)
        if not k_nodes:
            logger.debug("fuse_attention: failed to match k path")
            return
        assert k_path == 'path2', 'abnormal k paths found'
        # 4. Attention and CustomFC have been found, now transform the found nodes to two plugin nodes
        # Test 3 paths have the same origin
        is_same_origin = q_nodes[-1] is k_nodes[-1] is v_nodes[-1]
        is_same_origin &= q_nodes[-2] is k_nodes[-2] is v_nodes[-2]
        is_same_origin &= q_nodes[-3] is not k_nodes[-2] is not v_nodes[-3]
        if not is_same_origin:
            print("swin-L fuse_attention: found qkv path but not has the same origin")
            return
        origin_matmul = q_nodes[-1]
        fc_add = [q_nodes[-3], k_nodes[-3], v_nodes[-3]]
        # Now fuse
        num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_qkv)

        # Fuse FC
        weight = self.model.get_initializer(origin_matmul.input[1])
        biases = [self.model.get_initializer(i.input[0]) for i in fc_add]
        if not weight or not all(biases):
            print("swin-L: couldn't find weights")
            return
        weight_arr = onnx.numpy_helper.to_array(weight).transpose(1,0)
        weight.CopyFrom(numpy_helper.from_array(weight_arr))
        bias_arr = np.concatenate([onnx.numpy_helper.to_array(i) for i in biases], axis=0)

        fused_node = helper.make_node(
            "CustomFCPluginDynamic_IxRT",
            inputs=[origin_matmul.input[0]],
            outputs=fc_add[0].output,
            name=self.model.create_node_name("CustomFC", "MatMul_AddBias_"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("out_dims", bias_arr.shape[0])])
        fused_node.attribute.extend([helper.make_attribute("type_id", 2)])
        fused_node.attribute.extend([helper.make_attribute("W", weight)])
        fused_node.attribute.extend([helper.make_attribute("B", numpy_helper.from_array(bias_arr))])
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("act_type", -1)])
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)
        # Fuse Attention
        attention_node = self.create_attention_node(
            num_heads,
            hidden_size,
            [fused_node.output[0], relative_position_bias_name],
            reshape_qkv.output[0],

        )
        if not attention_node:
            return
        self.nodes_to_add.append(attention_node)
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.nodes_to_remove.extend([*qkv_nodes, *q_nodes[:-2], *k_nodes[:-2], *v_nodes])
        self.prune_graph = True

    def fuse_tensor_in_node_attrs(self, fc_nodes, attr_name, tensor_name):
        result = [get_tensor_attr(i.attribute, attr_name) for i in fc_nodes]
        result = np.concatenate(result, axis=0)
        result = numpy_helper.from_array(result, tensor_name)
        return result
