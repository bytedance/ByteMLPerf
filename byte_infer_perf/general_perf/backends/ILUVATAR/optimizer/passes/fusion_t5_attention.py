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
from .shape_infer_helper import SymbolicShapeInferenceHelper, get_shape_from_type_proto

logger = getLogger(__name__)


class FusionT5Attention(Fusion):
    """
    Fuse T5Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(
            model,
            "CustomQKVToContextPluginDynamic_IxRT",
            ["CustomSkipLayerNormPluginDynamic_IxRT", "RMSNormPluginDynamic_IxRT"],
        )

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape = self.model.get_initializer(reshape_q.input[1])
        if q_shape is None:
            logger.debug(f"{reshape_q.input[1]} is not initializer.")
            return [0, 0]

        q_shape_value = NumpyHelper.to_array(q_shape)
        if len(q_shape_value) != 4 or (q_shape_value[2] <= 0 or q_shape_value[3] <= 0):
            logger.debug(
                f"q_shape_value={q_shape_value}. Expected value are like [0, 0, num_heads, head_size]."
            )
            return [0, 0]

        num_heads = q_shape_value[2]
        head_size = q_shape_value[3]
        hidden_size = num_heads * head_size

        return num_heads, hidden_size

    def create_attention_node(
        self,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        matmul_qk_add: NodeProto,
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
            logger.debug(
                f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}"
            )
            return None

        attention_node_name = self.model.create_node_name("Attention")

        qk_bias = None
        has_mask = 0
        has_qk_bias = 0
        add_input_is_value = False
        if matmul_qk_add is not None:
            has_qk_bias = 1
            qk_bias = self.model.get_initializer(matmul_qk_add.input[1])
            if qk_bias:
                add_input_is_value = True
                qk_bias_arr = NumpyHelper.to_array(qk_bias)
                if len(qk_bias_arr.shape) == 3:
                    qk_bias_arr = qk_bias_arr.squeeze(0)
                has_neg_inf = np.isinf(qk_bias_arr) & (qk_bias_arr < 0)
                if np.any(has_neg_inf):
                    qk_bias_arr = np.where(qk_bias_arr == -np.inf, -100, 0.0).astype(
                        np.float32
                    )
                qk_bias.CopyFrom(numpy_helper.from_array(qk_bias_arr, qk_bias.name))

        attention_inputs = [input]

        # 如果add的输入不是值，而是一个边，那么这个边的值需要cast到fp32
        cast_node = None
        if not add_input_is_value:
            cast_out_name = attention_node_name + "_fp32_in1"
            cast_out_tensor = helper.make_tensor_value_info(
                cast_out_name, TensorProto.FLOAT, [None, None, None, None]
            )
            # self.model.add_initializer(cast_out_name)
            cast_node = helper.make_node(
                "Cast",
                inputs=[matmul_qk_add.input[1]],
                outputs=[cast_out_tensor.name],
                name=self.model.create_node_name("Cast"),
                to=1,
            )
            self.node_name_to_graph_name[cast_node.name] = self.this_graph_name
            attention_inputs.append(cast_out_name)

        if has_qk_bias:
            if add_input_is_value:
                has_mask = 1
                attention_inputs.append(qk_bias.name)
            else:
                has_mask = 1

        attention_node = helper.make_node(
            "CustomQKVToContextPluginDynamic_IxRT",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.iluvatar"
        attention_node.attribute.extend([helper.make_attribute("type_id", 2)])
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        attention_node.attribute.extend(
            [helper.make_attribute("hidden_size", hidden_size)]
        )
        attention_node.attribute.extend([helper.make_attribute("has_mask", has_mask)])
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        attention_node.attribute.extend(
            [helper.make_attribute("has_qk_bias", has_qk_bias)]
        )
        attention_node.attribute.extend([helper.make_attribute("is_t5_mode", 1)])

        return attention_node, cast_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type == "RMSNormPluginDynamic_IxRT":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_paths = {
            "path1": (["MatMul", "Reshape", "Transpose", "MatMul"], [0, 0, 0, 0]),
            "path2": (["MatMul", "Reshape", "Transpose", "MatMul"], [1, 0, 0, 0]),
        }

        qkv_nodes, qkv_path = self.match_parent_path_from_dict(start_node, qkv_paths)

        if qkv_nodes is None:
            logger.debug("fuse_attention: failed to match qkv path")
            return

        if qkv_path in ["path1", "path2"]:
            (atten_matmul, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        other_inputs = []
        for i, input in enumerate(start_node.input):
            if input not in output_name_to_node:
                continue

            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]
        """
        Match T5
        Add/Gather --> LayerNormalization --> Attention --> Add --> LayerNormalization
         |                                                  |
         |                                                  |
         +---------------------------------------------------
        """
        transpose_before_layernorm = self.model.match_parent(start_node, "Gather", 0)
        if transpose_before_layernorm is not None:
            node_children = input_name_to_nodes[transpose_before_layernorm.output[0]]
            for child in node_children:
                if child is not None and child.op_type == "RMSNormPluginDynamic_IxRT":
                    root_input = child.output[0]

        add_before_layernorm = self.model.match_parent(start_node, "Add", None)
        if add_before_layernorm is not None:
            node_children = input_name_to_nodes[add_before_layernorm.output[0]]
            for child in node_children:
                if child is not None and child.op_type == "RMSNormPluginDynamic_IxRT":
                    root_input = child.output[0]

        v_paths = {
            "path1": (
                ["Transpose", "Reshape", "Split", "MatMul"],
                [1, 0, 0, None],
            )  # T5
        }

        v_nodes, v_path = self.match_parent_path_from_dict(matmul_qkv, v_paths)
        if v_path == "path1":
            (_, _, _, matmul_in_qkv) = v_nodes

        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return

        qk_paths = {
            "path1": (["Softmax", "MatMul"], [0, 0]),
            "path2": (["Softmax", "Add", "MatMul"], [0, 0, None]),
        }

        qk_nodes, qk_path = self.match_parent_path_from_dict(matmul_qkv, qk_paths)

        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        matmul_qk_add = None
        if qk_path == "path1":
            (_, matmul_qk) = qk_nodes
        else:
            (_, matmul_qk_add, matmul_qk) = qk_nodes

        q_paths = {"path1": (["Transpose", "Reshape", "Split"], [0, 0, 0])}
        q_nodes, q_path = self.match_parent_path_from_dict(matmul_qk, q_paths)
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return

        if q_path == "path1":
            (_, reshape_q, split_q) = q_nodes
            # print("   split_q.name : ", split_q.name)

        k_paths = {
            "path1": (["Transpose", "Reshape", "Split"], [1, 0, 0]),
        }
        k_nodes, k_path = self.match_parent_path_from_dict(matmul_qk, k_paths)

        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return

        if k_path == "path1":
            (_, _, split_k) = k_nodes

        if (
            matmul_in_qkv.input[0] == root_input
            and split_q.input[0] == matmul_in_qkv.output[0]
            and split_k.input[0] == matmul_in_qkv.output[0]
        ):
            attention_last_node = reshape_qkv

            num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

            new_node, new_cast_node = self.create_attention_node(
                num_heads,
                hidden_size,
                matmul_in_qkv.output[0],
                attention_last_node.output[0],
                matmul_qk_add,
            )
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            if new_cast_node:
                self.nodes_to_add.append(new_cast_node)

            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend(
                [attention_last_node, transpose_qkv, matmul_qkv]
            )
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes[:-2])
