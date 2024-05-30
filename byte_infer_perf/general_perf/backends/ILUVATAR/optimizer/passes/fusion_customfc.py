# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionCustomFCGPT2(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "CustomFCPluginDynamic_IxRT", ["Reshape"], "gpt2")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        nodes = self.model.match_parent_path(node, ["Gemm", "Reshape"], [0, 0])

        if nodes is None:
            return False

        (matmul, reshape_before_matmul) = nodes

        matmul_weight = self.model.get_initializer(matmul.input[1])
        matmul_bias = self.model.get_initializer(matmul.input[2])

        if matmul_weight is None or matmul_bias is None:
            return False

        w = NumpyHelper.to_array(matmul_weight)
        b = NumpyHelper.to_array(matmul_bias)

        transB = 0
        for attr in matmul.attribute:
            if attr.name == "transB":
                transB = attr.i
                break

        trans_matmul_weight = w
        if transB == 0:
            trans_matmul_weight = w.transpose(1, 0)
        if matmul_weight.name not in self.model.initializer_visited.keys():
            self.model.initializer_visited[matmul_weight.name] = True
            if matmul_weight.data_type == 10:
                matmul_weight.CopyFrom(
                    numpy_helper.from_array(
                        trans_matmul_weight.astype(np.float16), matmul_weight.name
                    )
                )
            else:
                matmul_weight.CopyFrom(
                    numpy_helper.from_array(trans_matmul_weight, matmul_weight.name)
                )

        if matmul_bias.data_type == 10:
            matmul_bias.CopyFrom(
                numpy_helper.from_array(b.astype(np.float16), matmul_bias.name)
            )
        else:
            matmul_bias.CopyFrom(numpy_helper.from_array(b, matmul_bias.name))

        fused_node = helper.make_node(
            "CustomFCPluginDynamic_IxRT",
            inputs=[reshape_before_matmul.input[0]],
            outputs=node.output,
            name=self.model.create_node_name("CustomFC", "MatMul_AddBias_"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("out_dims", b.shape[0])])
        fused_node.attribute.extend([helper.make_attribute("type_id", 2)])
        fused_node.attribute.extend([helper.make_attribute("W", matmul_weight)])
        fused_node.attribute.extend([helper.make_attribute("B", matmul_bias)])
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("act_type", -1)])
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)
        self.nodes_to_remove.extend([matmul, node, reshape_before_matmul])


class FusionCustomFcRoformer(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "CustomFCPluginDynamic_IxRT", ["Add"], "roformer fc")

        # For model Roformer.

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if len(node.input) != 2:
            return False

        fc_paths = {
            "path1": (["Reshape", "MatMul", "Reshape"], [0, 0, 0]),
            "path2": (["Reshape", "MatMul", "Reshape"], [1, 0, 0]),
        }

        nodes, paths = self.match_parent_path_from_dict(node, fc_paths)
        if nodes is None:
            return False

        reshape_after_matmul = nodes[0]
        matmul = nodes[1]
        reshape_before_matmul = nodes[2]

        reshape_before_shape = None
        reshape_after_shape = None
        for value_info in self.model.graph().value_info:
            if value_info.name == reshape_before_matmul.input[0]:
                reshape_before_shape = len(value_info.type.tensor_type.shape.dim)
                break
        for value_info in self.model.graph().value_info:
            if value_info.name == reshape_after_matmul.output[0]:
                reshape_after_shape = len(value_info.type.tensor_type.shape.dim)
                break
        if reshape_before_shape != reshape_after_shape:
            return False

        weight = self.model.get_initializer(matmul.input[1])
        bias = self.model.get_initializer(node.input[1]) or self.model.get_initializer(
            node.input[0]
        )

        if weight is None or bias is None:
            return False

        w = NumpyHelper.to_array(weight)
        w_in_size = w.shape[0]
        weight_dim = np.prod(w.shape[1:])

        b = NumpyHelper.to_array(bias)
        bias_dim = np.prod(b.shape)
        trans_matmul_weight = w.transpose(1, 0)
        weight.CopyFrom(onnx.numpy_helper.from_array(trans_matmul_weight, weight.name))
        # Sometimes weights and bias are stored in fp16
        if weight.data_type == 10:
            weight.CopyFrom(
                numpy_helper.from_array(
                    trans_matmul_weight.astype(np.float16), weight.name
                )
            )
        bias_arr = onnx.numpy_helper.to_array(bias).flatten()
        bias.CopyFrom(onnx.numpy_helper.from_array(bias_arr, bias.name))
        if bias.data_type == 10:
            bias.CopyFrom(
                numpy_helper.from_array(
                    NumpyHelper.to_array(bias).astype(np.float16), bias.name
                )
            )

        fused_node = helper.make_node(
            "CustomFCPluginDynamic_IxRT",
            inputs=[reshape_before_matmul.input[0]],
            outputs=node.output,
            name=self.model.create_node_name("CustomFC", "MatMul_AddBias_"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("out_dims", b.shape[0])])
        fused_node.attribute.extend([helper.make_attribute("type_id", 2)])
        fused_node.attribute.extend([helper.make_attribute("W", weight)])
        fused_node.attribute.extend([helper.make_attribute("B", bias)])
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("act_type", -1)])
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)

        self.nodes_to_remove.extend([node])
        self.nodes_to_remove.extend(nodes)
        return True


class FusionCustomFC(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "CustomFCPluginDynamic_IxRT", ["Add"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if self.fuse_1(node, input_name_to_nodes, output_name_to_node):
            return

    def fuse_1(self, node, input_name_to_nodes, output_name_to_node):
        if len(node.input) != 2:
            return False
        nodes = self.model.match_parent_path(node, ["MatMul"], [None])

        if nodes is None:
            return False
        matmul = nodes[0]

        matmul_weight = self.model.get_initializer(matmul.input[1])
        matmul_bias = self.model.get_initializer(
            node.input[1]
        ) or self.model.get_initializer(node.input[0])

        if matmul_weight is None or matmul_bias is None:
            return False

        w = NumpyHelper.to_array(matmul_weight)
        b = NumpyHelper.to_array(matmul_bias)

        trans_matmul_weight = w.transpose(1, 0)
        if matmul_weight.name not in self.model.initializer_visited.keys():
            self.model.initializer_visited[matmul_weight.name] = True
            if matmul_weight.data_type == 10:
                matmul_weight.CopyFrom(
                    numpy_helper.from_array(
                        trans_matmul_weight.astype(np.float16), matmul_weight.name
                    )
                )
            else:
                matmul_weight.CopyFrom(
                    numpy_helper.from_array(trans_matmul_weight, matmul_weight.name)
                )

        if matmul_bias.data_type == 10:
            matmul_bias.CopyFrom(
                numpy_helper.from_array(b.astype(np.float16), matmul_bias.name)
            )
        else:
            matmul_bias.CopyFrom(numpy_helper.from_array(b, matmul_bias.name))

        fused_node = helper.make_node(
            "CustomFCPluginDynamic_IxRT",
            inputs=[matmul.input[0]],
            outputs=node.output,
            name=self.model.create_node_name("CustomFC", "MatMul_AddBias_"),
        )
        fused_node.domain = "com.iluvatar"
        fused_node.attribute.extend([helper.make_attribute("out_dims", b.shape[0])])
        fused_node.attribute.extend([helper.make_attribute("type_id", 2)])
        fused_node.attribute.extend([helper.make_attribute("W", matmul_weight)])
        fused_node.attribute.extend([helper.make_attribute("B", matmul_bias)])
        fused_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        fused_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        fused_node.attribute.extend([helper.make_attribute("act_type", -1)])
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)
        self.nodes_to_remove.extend([matmul, node])
        return True


class FusionCustomFCActivation(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(
            model,
            "CustomFCPluginDynamic_IxRT",
            ["Gelu", "Relu", "CustomGeluPluginDynamic_IxRT", "Mul"],
            "with activation",
        )

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if node.op_type == "Mul":
            return_indice = []
            nodes = self.model.match_parent_path(
                node,
                ["Sigmoid", "Mul", "CustomFCPluginDynamic_IxRT"],
                [None, 0, 0],
                return_indice=return_indice,
            )
            if nodes is None:
                return

            (sigmoid_node, mul_node, custom_fc_node) = nodes
            if output_name_to_node[node.input[1 - return_indice[0]]] != custom_fc_node:
                return

            activation_type = 20
            for attr in custom_fc_node.attribute:
                if attr.name == "act_type":
                    attr.i = activation_type
                    break

            custom_fc_node.output[0] = node.output[0]
            self.nodes_to_add.append(custom_fc_node)
            self.nodes_to_remove.extend([node, sigmoid_node, mul_node, custom_fc_node])
            self.node_name_to_graph_name[custom_fc_node.name] = self.this_graph_name
        else:
            nodes = self.model.match_parent_path(
                node, ["CustomFCPluginDynamic_IxRT"], [0]
            )

            if nodes is None:
                logger.debug("CustomFCActivation: failed to match fc+gelu/relu path")
                return

            fc_node = nodes[0]
            activation_type = 3
            if node.op_type == "Gelu":
                activation_type = 21
            if node.op_type == "Relu":
                activation_type = 4

            for attr in fc_node.attribute:
                if attr.name == "act_type":
                    attr.i = activation_type
                    break

            fc_node.output[0] = node.output[0]
            self.nodes_to_add.append(fc_node)
            self.nodes_to_remove.extend([node, fc_node])
            self.node_name_to_graph_name[fc_node.name] = self.this_graph_name


class FusionConformerCustomFCActivation(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(
            model,
            "CustomFCPluginDynamic_IxRT",
            ["Mul"],
            "with activation",
        )

    def fuse(self, node, input_name_to_nodes, output_name_to_node):

        # return_indice = []
        nodes = self.model.match_parent_path(
            node,
            ["Sigmoid", "CustomFCPluginDynamic_IxRT"],
            [
                None,
                0,
            ],
            # return_indice=return_indice,
        )
        if nodes is None:
            return
        (sigmoid_node, custom_fc_node) = nodes
        # if output_name_to_node[node.input[1 - return_indice[0]]] != custom_fc_node:
        #     return
        activation_type = 20
        for attr in custom_fc_node.attribute:
            if attr.name == "act_type":
                attr.i = activation_type
                break
        custom_fc_node.attribute.extend([helper.make_attribute("swish_alpha", 1.0)])
        custom_fc_node.output[0] = node.output[0]
        self.nodes_to_add.append(custom_fc_node)
        self.nodes_to_remove.extend([node, sigmoid_node, custom_fc_node])
        self.node_name_to_graph_name[custom_fc_node.name] = self.this_graph_name
