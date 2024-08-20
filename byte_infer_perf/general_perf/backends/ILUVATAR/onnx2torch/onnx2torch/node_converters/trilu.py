__all__ = [
    'OnnxTrilu',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxTrilu(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, upper: int):
        super().__init__()
        self.upper = upper

    def forward(self, *input_tensors) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        diagonal = 0
        if len(input_tensors) > 1:
            diagonal = input_tensors[1]

        if self.upper:
            return torch.triu(input_tensors[0], diagonal)
        else:
            return torch.tril(input_tensors[0], diagonal)


@add_converter(operation_type='Trilu', version=4)
@add_converter(operation_type='Trilu', version=11)
@add_converter(operation_type='Trilu', version=13)
@add_converter(operation_type='Trilu', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    upper = node.attributes.get('k', 1)

    torch_module = OnnxTrilu(
        upper=upper,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
