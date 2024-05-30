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


class FusionConformerAttention(Fusion):
    """
    Fuse VideoBertAttention subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int):
        super().__init__(model, "CustomQKVToContextPluginDynamic_IxRT", ["Concat"])

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def get_num_heads_and_hidden_size(
        self, atten_matmul: NodeProto, div: NodeProto
    ) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        atten_matul_initializer = self.model.get_initializer(atten_matmul.input[1])
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

        atten_matul_shape_value = NumpyHelper.to_array(atten_matul_initializer).shape
        head_dim = math.ceil(div_value * div_value)
        hidden_size = atten_matul_shape_value[0]
        num_heads = hidden_size // head_dim

        return num_heads, hidden_size

    def create_attention_node(
        self, num_heads: int, hidden_size: int, inputs: str, outputs: str
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

        attention_node = helper.make_node(
            "CustomQKVToContextPluginDynamic_IxRT",
            inputs=inputs,
            outputs=outputs,
            name=attention_node_name,
        )
        attention_node.domain = "com.iluvatar"
        attention_node.attribute.extend([helper.make_attribute("type_id", 2)])
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        attention_node.attribute.extend(
            [helper.make_attribute("hidden_size", hidden_size)]
        )
        attention_node.attribute.extend([helper.make_attribute("has_mask", 1)])
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        attention_node.attribute.extend([helper.make_attribute("has_qk_bias", 1)])

        return attention_node

    def fuse_reshape(self, shape_data_name):

        shape_tensor = helper.make_tensor(
            name=shape_data_name,
            data_type=TensorProto.INT64,
            dims=[3],
            vals=np.int64([128, -1, self.hidden_size // self.num_heads]).tobytes(),
            raw=True,
        )
        self.model.add_initializer(shape_tensor, self.this_graph_name)

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node

        paths = {
            "path": (
                ["Unsqueeze", "Mul", "Gather", "Shape", "LayerNormalization"],
                [None, None, None, None, None],
            ),
        }

        reshape_nodes, reshape_path = self.match_parent_path_from_dict(
            start_node, paths
        )
        if reshape_nodes is None:
            return

        self.nodes_to_remove.append(start_node)

        self.nodes_to_remove.extend(reshape_nodes[:-1])
        self.fuse_reshape(start_node.output[0])
