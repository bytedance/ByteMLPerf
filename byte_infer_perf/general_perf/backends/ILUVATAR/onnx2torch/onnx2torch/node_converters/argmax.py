__all__ = [
    'OnnxArgMax',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxArgMax(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis: int, keepdims: int, **kwargs):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims == 1

    def forward(self, *input_tensors) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if len(input_tensors) != 1:
            raise RuntimeError(f"Invalid input tensors, expect one tensor, but got {len(input_tensors)}.")
        return torch.argmax(input_tensors[0], dim=self.axis, keepdim=self.keepdims)


@add_converter(operation_type='ArgMax', version=4)
@add_converter(operation_type='ArgMax', version=11)
@add_converter(operation_type='ArgMax', version=13)
@add_converter(operation_type='ArgMax', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    keepdims = node.attributes.get("keepdims", 1)

    torch_module = OnnxArgMax(
        axis=axis,
        keepdims=keepdims
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
