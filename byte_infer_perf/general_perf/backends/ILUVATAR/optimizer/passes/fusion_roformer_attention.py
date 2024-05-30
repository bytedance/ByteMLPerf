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


class FusionRoformerCrossAttention(Fusion):
    """
    Fuse VideoBertAttention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(
            model,
            "CustomQkvCrossToContext_IxRT",
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
        head_dim = math.floor(1.0 / (mul_value * mul_value))
        hidden_size = tensor_shape[0]
        num_heads = hidden_size // head_dim

        return num_heads, hidden_size

    def create_attention_node(
        self,
        num_heads: int,
        hidden_size: int,
        input_q: str,
        input_k: str,
        input_v: str,
        input_mask: str,
        output: str,
        matmul_qk_add: NodeProto,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input_q: str,
            input_k: str,
            input_v: str,
            input_mask: str,
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

        attention_node_name = self.model.create_node_name("CrossAttention")

        attention_inputs = [input_q, input_k, input_v, input_mask]

        attention_node = helper.make_node(
            "CustomQkvCrossToContext_IxRT",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.iluvatar"
        attention_node.attribute.extend([helper.make_attribute("type_id", 2)])
        attention_node.attribute.extend([helper.make_attribute("has_mask", 1)])
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])

        return attention_node

    def get_shape(self, edge_name):
        for info in self.model.graph().value_info:
            if info.name == edge_name:
                return info.type.tensor_type.shape.dim
        return None

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_paths = {
            "path1": (
                [
                    "CustomFCPluginDynamic_IxRT",
                    "Reshape",
                    "Transpose",
                    "Reshape",
                    "MatMul",
                ],
                [0, 0, 0, 0, 0],
            ),
            "path2": (
                [
                    "CustomFCPluginDynamic_IxRT",
                    "Reshape",
                    "Transpose",
                    "Reshape",
                    "MatMul",
                ],
                [1, 0, 0, 0, 0],
            ),
        }
        # print('start_nodes:', start_node.name)
        qkv_nodes, qkv_path = self.match_parent_path_from_dict(start_node, qkv_paths)

        if qkv_nodes is None:
            logger.debug("fuse_attention: failed to match qkv path")
            return

        fc_after_atten = None
        if qkv_path in ["path1", "path2"]:
            (
                fc_after_atten,
                reshape_qkv_2,
                transpose_qkv,
                reshape_qkv_1,
                matmul_qkv,
            ) = qkv_nodes

        """
        Match
        Add --> LayerNormalization -->  Attention -->     Add --> LayerNormalization
         |                                                        |
         |                                                        |
         +---------------------------------------------------------
        """
        add_before_layernorm = self.model.match_parent(start_node, "Add", None)
        if add_before_layernorm is not None:
            node_children = input_name_to_nodes[add_before_layernorm.output[0]]
            for child in node_children:
                if child is not None and child.op_type == "LayerNormalization":
                    root_input = child.output[0]

        v_paths = {"path1": (["Reshape", "Transpose", "Reshape"], [1, 0, 0])}

        v_nodes, v_path = self.match_parent_path_from_dict(matmul_qkv, v_paths)
        if v_path == "path1":
            (reshape_v, transpose_v, v_reshape) = v_nodes

        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return

        qk_paths = {
            "path1": (
                ["Softmax", "Add", "Mul", "Mul", "Reshape", "MatMul"],
                [0, 0, None, None, None, 0],
            )
        }

        qk_nodes, qk_path = self.match_parent_path_from_dict(matmul_qkv, qk_paths)

        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return
        # print('qk_nodes', qk_nodes[0].name)
        matmul_qk_add = None
        if qk_path == "path1":
            (_, add_mask, mul_mask, mul_qk, reshape_qk, matmul_qk) = qk_nodes

        q_paths = {
            "path1": (["Transpose", "Add"], [0, 0]),
        }
        q_nodes, q_path = self.match_parent_path_from_dict(matmul_qk, q_paths)
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        # print('q_nodes', q_nodes[0].name)
        if q_path == "path1":
            (q_tranpose, q_add) = q_nodes

        k_paths = {
            "path1": (["Reshape", "Transpose", "Add"], [1, 0, 0]),
        }
        k_nodes, k_path = self.match_parent_path_from_dict(matmul_qk, k_paths)

        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        # print('k_nodes', k_nodes[0].name)
        if k_path == "path1":
            (_, k_transpose, k_add) = k_nodes
        # print('add_mask', add_mask.name)
        mask_paths = {
            "path1": (
                ["Mul", "Sub", "Unsqueeze", "Cast", "Greater"],
                [1, None, 1, 0, 0],
            )
        }
        mask_nodes, mask_path = self.match_parent_path_from_dict(add_mask, mask_paths)

        if mask_nodes is None:
            logger.debug("fuse_attention: failed to match mask path")
            return
        # print('mask_nodes', mask_nodes[0].name)
        (_, mask_sub, mask_unsqueeze, mask_cast, mask_greater) = mask_nodes

        if (
            self.get_shape(q_add.output[0]) == self.get_shape(k_add.output[0])
            and self.get_shape(k_add.output[0]) == self.get_shape(v_reshape.output[0])
            and mul_mask.input[1] in mask_unsqueeze.output
        ):
            attention_last_node = reshape_qkv_1

            num_heads, hidden_size = self.get_num_heads_and_hidden_size(
                fc_after_atten, mul_qk
            )

            q_transpose_type = None
            q_transpose_name = None
            for info in self.model.graph().value_info:
                if info.name == q_tranpose.output[0]:
                    q_transpose_type = info.type
                    q_transpose_name = info.name
                    break

            q_transpose_output = helper.make_value_info(
                q_transpose_name[:-2] + "_fake_q", q_transpose_type
            )
            q_transpose_node = helper.make_node(
                "Transpose",
                inputs=[q_add.output[0]],
                outputs=[q_transpose_output.name],
                name=q_transpose_output.name,
            )
            q_transpose_node.attribute.extend(
                [helper.make_attribute("perm", [0, 2, 1, 3])]
            )

            k_transpose_output = helper.make_value_info(
                q_transpose_name[:-2] + "_fake_k", q_transpose_type
            )
            k_transpose_node = helper.make_node(
                "Transpose",
                inputs=[k_add.output[0]],
                outputs=[k_transpose_output.name],
                name=k_transpose_output.name,
            )
            k_transpose_node.attribute.extend(
                [helper.make_attribute("perm", [0, 2, 1, 3])]
            )

            v_transpose_output = helper.make_value_info(
                q_transpose_name[:-2] + "_fake_v", q_transpose_type
            )
            v_transpose_node = helper.make_node(
                "Transpose",
                inputs=[v_reshape.output[0]],
                outputs=[v_transpose_output.name],
                name=v_transpose_output.name,
            )
            v_transpose_node.attribute.extend(
                [helper.make_attribute("perm", [0, 2, 1, 3])]
            )

            mask_type = None
            for info in self.model.graph().value_info:
                if info.name == mask_sub.output[0]:
                    mask_type = info.type
                    break

            new_mask_type = onnx.TypeProto()
            new_mask_type.tensor_type.elem_type = onnx.TensorProto.INT32
            for dim in mask_type.tensor_type.shape.dim:
                new_dim = new_mask_type.tensor_type.shape.dim.add()
                new_dim.CopyFrom(dim)

            mask_cast_to_int32_output = helper.make_value_info(
                mask_sub.name + "_cast_to_int32", new_mask_type
            )
            mask_cast_to_int32_node = helper.make_node(
                "Cast",
                inputs=[mask_sub.output[0]],
                outputs=[mask_cast_to_int32_output.name],
                name=mask_cast_to_int32_output.name,
            )
            mask_cast_to_int32_node.attribute.extend([helper.make_attribute("to", 6)])

            new_node = self.create_attention_node(
                num_heads,
                hidden_size,
                q_transpose_node.output[0],
                k_transpose_node.output[0],
                v_transpose_node.output[0],
                mask_cast_to_int32_node.output[0],
                attention_last_node.output[0],
                matmul_qk_add,
            )
            if new_node is None:
                return

            self.nodes_to_add.extend(
                [
                    q_transpose_node,
                    k_transpose_node,
                    v_transpose_node,
                    new_node,
                    mask_cast_to_int32_node,
                ]
            )
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name
            self.node_name_to_graph_name[q_transpose_node.name] = self.this_graph_name
            self.node_name_to_graph_name[k_transpose_node.name] = self.this_graph_name
            self.node_name_to_graph_name[v_transpose_node.name] = self.this_graph_name
            self.node_name_to_graph_name[
                mask_cast_to_int32_node.name
            ] = self.this_graph_name

            self.nodes_to_remove.extend(qkv_nodes[3:])
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes[:-1])
            self.nodes_to_remove.extend(k_nodes[:-1])
            self.nodes_to_remove.extend(v_nodes[:-1])
            self.nodes_to_remove.extend([mask_nodes[0]])
