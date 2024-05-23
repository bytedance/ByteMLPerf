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


class FusionVITAttention(Fusion):
    """
    Fuse VITAttention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(
            model,
            "CustomQKVToContextPluginDynamic_IxRT",
            ["CustomSkipLayerNormPluginDynamic_IxRT", "LayerNormalization"],
        )

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size(
        self, custom_fc: NodeProto, mul: NodeProto
    ) -> Tuple[int, int]:
        mul_initializer = self.model.get_initializer(mul.input[1])

        # 检查float_data是否为空
        if len(mul_initializer.float_data) > 0:
            mul_value = mul_initializer.float_data[0]
        else:
            # 如果float_data为空，尝试其他方式获取数据
            # 例如，如果数据存储在raw_data中
            if len(mul_initializer.raw_data) > 0:
                dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[mul_initializer.data_type]
                mul_value = np.frombuffer(mul_initializer.raw_data, dtype=dtype)[0]
            else:
                raise ValueError("Data not found in the mul_initializer")

        for attr in custom_fc.attribute:
            if attr.name == "W":
                tensor_value = attr.t
                tensor_shape = [dim for dim in tensor_value.dims]
                break
        head_dim = math.floor(1.0 / (mul_value * mul_value)) * math.floor(
            1.0 / (mul_value * mul_value)
        )
        hidden_size = tensor_shape[0]
        num_heads = hidden_size // head_dim

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
        # print(hidden_size, num_heads)
        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(
                f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}"
            )
            return None

        attention_node_name = self.model.create_node_name("Attention")

        qk_bias = None
        has_mask = 0
        has_qk_bias = 0
        if matmul_qk_add is not None:
            has_qk_bias = 1
            qk_bias = self.model.get_initializer(matmul_qk_add.input[1])
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

        if qk_bias is not None:
            has_mask = 1
            attention_inputs.append(qk_bias.name)

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

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type == "LayerNormalization":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_paths = {
            "path1": (["CustomFCPluginDynamic_IxRT", "Transpose", "MatMul"], [0, 0, 0]),
            "path2": (["CustomFCPluginDynamic_IxRT", "Transpose", "MatMul"], [1, 0, 0]),
        }

        qkv_nodes, qkv_path = self.match_parent_path_from_dict(start_node, qkv_paths)

        if qkv_nodes is None:
            logger.debug("fuse_attention: failed to match qkv path")
            return

        if qkv_path in ["path1", "path2"]:
            (custom_fc_after_atten, transpose_qkv, matmul_qkv) = qkv_nodes

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
        Match VIT
        transpose --> LayerNormalization -->  custom_fc -> attention -> Add
         |                                                                  |
         |                                                                  |
         +-------------------------------------------------------------------
        """
        transpose_before_layernorm = self.model.match_parent(start_node, "Transpose", 0)
        if transpose_before_layernorm is not None:
            node_children = input_name_to_nodes[transpose_before_layernorm.output[0]]
            for child in node_children:
                if child is not None and child.op_type == "LayerNormalization":
                    root_input = child.output[0]

        add_before_layernorm = self.model.match_parent(start_node, "Add", None)
        if add_before_layernorm is not None:
            node_children = input_name_to_nodes[add_before_layernorm.output[0]]
            for child in node_children:
                if child is not None and child.op_type == "LayerNormalization":
                    root_input = child.output[0]

        # print("root_input: ", root_input, matmul_qkv.name)
        v_paths = {
            "path1": (
                [
                    "Reshape",
                    "Transpose",
                    "Reshape",
                    "Gather",
                    "Squeeze",
                    "Transpose",
                    "Unsqueeze",
                    "Reshape",
                    "CustomFCPluginDynamic_IxRT",
                ],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
            )  # vit
        }

        v_nodes, v_path = self.match_parent_path_from_dict(matmul_qkv, v_paths)

        squeeze_input = custom_fc = None
        if v_path == "path1":
            (_, _, _, _, squeeze_input, _, _, _, custom_fc) = v_nodes

        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return

        qk_paths = {
            "path1": (["Softmax", "MatMul"], [0, 0]),
            "path2": (["Softmax", "Add", "MatMul"], [0, 0, None]),
        }

        qk_nodes, qk_path = self.match_parent_path_from_dict(matmul_qkv, qk_paths)
        # print("qk_nodes:", qk_nodes[1].name)
        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        matmul_qk_add = None
        if qk_path == "path1":
            (_, matmul_qk) = qk_nodes
        else:
            (_, matmul_qk_add, matmul_qk) = qk_nodes

        q_paths = {
            "path1": (
                ["Mul", "Reshape", "Transpose", "Reshape", "Gather", "Squeeze"],
                [0, 0, 0, 0, 0, 0],
            ),
        }
        q_nodes, q_path = self.match_parent_path_from_dict(matmul_qk, q_paths)
        # print("q_nodes:", q_nodes[0].name)
        squeeze_q = mul_q = None
        if q_path == "path1":
            squeeze_q = q_nodes[-1]
            mul_q = q_nodes[0]

        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return

        k_paths = {
            "path1": (
                [
                    "Mul",
                    "Transpose",
                    "Reshape",
                    "Transpose",
                    "Reshape",
                    "Gather",
                    "Squeeze",
                ],
                [1, 0, 0, 0, 0, 0, 0],
            ),
        }
        k_nodes, k_path = self.match_parent_path_from_dict(matmul_qk, k_paths)
        # print("k_nodes:", k_nodes[0].name)
        squeeze_k = None
        if k_path == "path1":
            squeeze_k = k_nodes[-1]

        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return

        if (
            custom_fc.input[0] == root_input
            and squeeze_input == squeeze_q
            and squeeze_input == squeeze_k
        ):
            attention_last_node = transpose_qkv

            num_heads, hidden_size = self.get_num_heads_and_hidden_size(
                custom_fc_after_atten, mul_q
            )

            new_node = self.create_attention_node(
                num_heads,
                hidden_size,
                custom_fc.output[0],
                attention_last_node.output[0],
                matmul_qk_add,
            )
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes[:-1])
            self.nodes_to_remove.extend(k_nodes[:-1])
            self.nodes_to_remove.extend(v_nodes[:-1])

            # fuse head and tail transpose
            if transpose_before_layernorm is not None:
                node_children = input_name_to_nodes[
                    transpose_before_layernorm.output[0]
                ]
                for child in node_children:
                    for i, input in enumerate(child.input):
                        if child.input[i] == transpose_before_layernorm.output[0]:
                            child.input[i] = transpose_before_layernorm.input[0]
                self.nodes_to_remove.extend([transpose_before_layernorm])

                node = transpose_before_layernorm
                while True:
                    found = False
                    node_children = input_name_to_nodes[node.output[0]]
                    for child in node_children:
                        if child is not None and child.op_type in [
                            "SkipLayerNorm",
                            "Add",
                        ]:
                            node = child
                            found = True
                            break
                    if not found:
                        break
                node_children = input_name_to_nodes[node.output[0]]
                if len(node_children) == 1 and node_children[0].op_type == "Transpose":
                    transpose_node = node_children[0]
                    transpose_children = input_name_to_nodes[transpose_node.output[0]]
                    for i, input in enumerate(transpose_children[0].input):
                        if transpose_children[0].input[i] == transpose_node.output[0]:
                            transpose_children[0].input[i] = transpose_node.input[0]
                    self.nodes_to_remove.extend([transpose_node])
            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            # self.nodes_to_remove.extend(mask_nodes)
            # self.prune_graph = True
