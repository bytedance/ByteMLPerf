# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math
from logging import getLogger
from typing import Tuple, Union

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import FusionUtils, NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionGptAttentionNoPast(Fusion):
    """
    Fuse GPT-2 Attention without past state into one Attention node.
    This does not support attention_mask graph input right now.
    """

    def __init__(self, model: OnnxModel):
        super().__init__(
            model,
            "CustomQKVToContextPluginDynamic_IxRT",
            ["CustomSkipLayerNormPluginDynamic_IxRT", "LayerNormalization"],
            "without past",
        )
        self.where_qk_shared = None

    def get_num_heads_and_hidden_size(
        self, custom_fc: NodeProto, div: NodeProto
    ) -> Tuple[int, int]:
        div_initializer = self.model.get_initializer(div.input[1])

        # 检查float_data是否为空
        if len(div_initializer.float_data) > 0:
            div_value = div_initializer.float_data[0]
        else:
            # 如果float_data为空，尝试其他方式获取数据
            # 例如，如果数据存储在raw_data中
            if len(div_initializer.raw_data) > 0:
                dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[div_initializer.data_type]
                div_value = np.frombuffer(div_initializer.raw_data, dtype=dtype)[0]
            else:
                raise ValueError("Data not found in the div_initializer")

        for attr in custom_fc.attribute:
            if attr.name == "W":
                tensor_value = attr.t
                tensor_shape = [dim for dim in tensor_value.dims]
                break
        head_dim = math.ceil(div_value * div_value)
        hidden_size = tensor_shape[0]
        num_heads = hidden_size // head_dim

        return num_heads, hidden_size

    def create_attention_node(
        self,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        where_qk: NodeProto,
    ) -> Union[NodeProto, None]:

        attention_node_name = self.model.create_node_name("Attention")

        attention_inputs = [input]
        if where_qk is not None:
            has_mask = 1
            has_qk_bias = 1
            attention_inputs.append(where_qk.output[0])

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
        return_indice = []
        add_qkv = normalize_node
        if normalize_node.op_type == "LayerNormalization":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                add_qkv = add_before_layernorm

        qkv_paths = {
            "path1": (
                ["CustomFCPluginDynamic_IxRT", "Reshape", "Transpose", "MatMul"],
                [None, 0, 0, 0],
            ),
            "path2": (
                ["CustomFCPluginDynamic_IxRT", "Transpose", "MatMul"],
                [None, 0, 0],
            ),
        }

        qkv_nodes, qkv_path = self.match_parent_path_from_dict(
            add_qkv,
            qkv_paths,
            output_name_to_node,
            return_indice,
        )  # yapf: disable

        if qkv_nodes is None:
            return
        reshape_2 = None
        if qkv_path == "path1":
            (
                custom_fc_after_attention,
                reshape_2,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes
        else:
            (
                custom_fc_after_attention,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes

        another_input = add_qkv.input[1 - return_indice[0]]

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "Split", "CustomFCPluginDynamic_IxRT"],
            [1, 0, 0, 0],
        )  # yapf: disable
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (
            transpose_v,
            reshape_v,
            split_v,
            custom_fc_before_attention,
        ) = v_nodes

        layernorm_before_attention = self.model.get_parent(
            custom_fc_before_attention, 0, output_name_to_node
        )
        if (
            layernorm_before_attention is None
            or layernorm_before_attention.op_type != "LayerNormalization"
        ):
            if layernorm_before_attention.op_type != "Add":
                logger.debug(
                    f"failed to get layernorm before gemm. Got {layernorm_before_attention.op_type}"
                )
                return

        if not another_input in layernorm_before_attention.input:
            # match openai-gpt
            if not another_input in layernorm_before_attention.output:
                logger.debug("Add and LayerNormalization shall have one same input")
                return

        qk_nodes = self.model.match_parent_path(
            matmul_qkv, ["Softmax", "Add", "Where", "Div", "MatMul"], [0, None, 0, 1, 0]
        )
        where_qk = None
        matmul_qk = None
        mask_return_indices = []
        if qk_nodes is not None:
            (softmax_qk, add_qk, where_qk, div_qk, matmul_qk) = qk_nodes
            mask_nodes = self.model.match_parent_path(
                add_qk,
                ["Mul", "Sub", "Cast", "Unsqueeze"],
                [None, 0, 1, 0],
                return_indice=mask_return_indices,
            )  # yapf: disable
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match mask path")
                return

        q_nodes = self.model.match_parent_path(
            matmul_qk, ["Transpose", "Reshape", "Split"], [0, 0, 0]
        )
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (transpose_q, reshape_q, split_q) = q_nodes
        if split_v != split_q:
            logger.debug("fuse_attention: skip since split_v != split_q")
            return

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Transpose", "Reshape", "Split"], [1, 0, 0]
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        (transpose_k, reshape_k, split_k) = k_nodes
        if split_v != split_k:
            logger.debug("fuse_attention: skip since split_v != split_k")
            return

        if where_qk is None:
            return

        if self.where_qk_shared is None:
            where_qk.input[1] = mask_nodes[0].output[0]
            div_qk.output[0] = where_qk.output[0]
            add_qk.input[1 - mask_return_indices[0]] = div_qk.output[0]
            self.where_qk_shared = where_qk
            self.nodes_to_remove.extend([softmax_qk, add_qk, div_qk, matmul_qk])
        else:
            self.nodes_to_remove.extend(
                [softmax_qk, add_qk, where_qk, div_qk, matmul_qk]
            )

        num_heads, hidden_size = self.get_num_heads_and_hidden_size(
            custom_fc_after_attention, div_qk
        )
        new_node = self.create_attention_node(
            num_heads,
            hidden_size,
            custom_fc_before_attention.output[0],
            transpose_qkv.output[0] if reshape_2 is None else reshape_2.output[0],
            self.where_qk_shared,
        )

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        if reshape_2 is not None:
            self.nodes_to_remove.extend([reshape_2])
        self.nodes_to_remove.extend([transpose_qkv, matmul_qkv])
        self.nodes_to_remove.extend(q_nodes)
        self.nodes_to_remove.extend(k_nodes)
        self.nodes_to_remove.extend(v_nodes[:-1])
