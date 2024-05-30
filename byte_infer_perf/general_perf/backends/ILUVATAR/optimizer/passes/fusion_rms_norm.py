import logging
from typing import Dict

from onnx import helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class FusionRMSNorm(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "RMSNorm", "Mul")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        if node.op_type != "Mul":
            return

        sim_ln_nodes = None
        # SimplifiedLayerNorm calculation (notation from https://onnx.ai/onnx/operators/onnx__LayerNormalization.html#summary):
        # DD = Pow(D, 2)
        # Var = ReduceMean(DD)
        # VarEps = Add(Var, epsilon)
        # StdDev = Sqrt(VarEps)
        # InvStdDev = Div(1, StdDev)
        # Normalized = Mul(D, InvStdDev)
        # NormalizedScaled = Mul(Normalized, Scale)

        #                              RMSNorm
        #          +-------------------------------------------------------+
        #          |                                                       |
        # Add --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Mul
        #                                                                  |
        #                                                                 node
        sim_ln_nodes_1 = self.model.match_parent_path(
            node,
            ["Mul", "Div", "Sqrt", "Add", "ReduceMean", "Pow", "Add"],
            [1, 1, 1, 0, 0, 0, 0],
        )
        #                                RMSNorm
        #             +-------------------------------------------------------+
        #             |                                                       |
        # Gather --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Mul
        #                                                                     |
        #                                                                    node
        sim_ln_nodes_2 = self.model.match_parent_path(
            node,
            ["Mul", "Div", "Sqrt", "Add", "ReduceMean", "Pow", "Gather"],
            [1, 1, 1, 0, 0, 0, 0],
        )

        # For LLaMA from Microsoft custom export:
        # sim_ln_nodes_3 uses a different start parent index than sim_ln_nodes_1
        #
        #                              RMSNorm
        #          +-------------------------------------------------------+
        #          |                                                       |
        # Add --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Mul
        #                                                                  |
        #                                                                 node
        sim_ln_nodes_3 = self.model.match_parent_path(
            node,
            ["Mul", "Div", "Sqrt", "Add", "ReduceMean", "Pow", "Add"],
            [0, 1, 1, 0, 0, 0, 0],
        )

        # sim_ln_nodes_4 starts with a graph input instead of an Add node like sim_ln_nodes_3
        #
        #                                  RMSNorm
        #                  +-----------------------------------------------+
        #                  |                                               |
        # graph_input --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul
        #                                                                  |
        #                                                                 node
        sim_ln_nodes_4 = self.model.match_parent_path(
            node,
            ["Mul", "Div", "Sqrt", "Add", "ReduceMean", "Pow"],
            [0, 1, 1, 0, 0, 0],
        )

        add_node, pow_node = None, None
        if sim_ln_nodes_1 is not None:
            sim_ln_nodes = sim_ln_nodes_1
            add_node = sim_ln_nodes[3]
            pow_node = sim_ln_nodes[-2]
        elif sim_ln_nodes_2 is not None:
            sim_ln_nodes = sim_ln_nodes_2
            add_node = sim_ln_nodes[3]
            pow_node = sim_ln_nodes[-2]
        elif sim_ln_nodes_3 is not None:
            sim_ln_nodes = sim_ln_nodes_3
            add_node = sim_ln_nodes[3]
            pow_node = sim_ln_nodes[-2]
        elif sim_ln_nodes_4 is not None:
            sim_ln_nodes = sim_ln_nodes_4
            add_node = sim_ln_nodes[3]
            pow_node = sim_ln_nodes[-1]
            # Verify that parent input to Pow node is graph_input
            if pow_node.input[0] not in self.model.get_graphs_input_names():
                return
        else:
            return

        layernorm_weight_index = (
            1 if sim_ln_nodes in (sim_ln_nodes_3, sim_ln_nodes_4) else 0
        )
        starts_with_graph_input = sim_ln_nodes == sim_ln_nodes_4

        if self.model.find_constant_input(pow_node, 2.0) != 1:
            return

        root_input = pow_node.input[0]
        if root_input != sim_ln_nodes[0].input[0]:
            return

        i, add_weight = self.model.get_constant_input(add_node)
        if add_weight is None or add_weight <= 0 or add_weight > 1.0e-4:
            logger.warning(f"epsilon value is not expected: {add_weight}")
            return

        self.nodes_to_remove.extend(
            sim_ln_nodes[:-1] if not starts_with_graph_input else sim_ln_nodes
        )
        self.nodes_to_remove.append(node)

        normalize_node = helper.make_node(
            "RMSNormPluginDynamic_IxRT",
            inputs=[root_input, node.input[layernorm_weight_index]],
            outputs=[node.output[0]],
            name=self.model.create_node_name(
                "RMSNormPluginDynamic_IxRT", name_prefix="RMSNorm_"
            ),
        )

        normalize_node.domain = "com.iluvatar"
        normalize_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        normalize_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        normalize_node.attribute.extend(
            [helper.make_attribute("epsilon", float(add_weight))]
        )
        normalize_node.attribute.extend([helper.make_attribute("axis", -1)])
        normalize_node.attribute.extend([helper.make_attribute("stash_type", 1)])
        gamma_data = self.model.get_initializer(normalize_node.input[1])
        gamma_data_np = NumpyHelper.to_array(gamma_data)
        normalize_node.attribute.extend(
            [helper.make_attribute("hidden_size", int(gamma_data_np.shape[0]))]
        )

        normalize_node.attribute.extend([helper.make_attribute("gamma", gamma_data)])

        self.nodes_to_add.append(normalize_node)
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name
        return True
