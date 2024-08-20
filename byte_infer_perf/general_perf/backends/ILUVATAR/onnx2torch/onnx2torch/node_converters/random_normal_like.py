__all__ = [
    'OnnxRandomNormalLike',
]

from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
import onnx2torch.utils.dtype as dtype_utils


class OnnxRandomNormalLike(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, dtype: Optional[int], mean: Optional[float], scale: Optional[float], seed: Optional[int]):
        super().__init__()
        if dtype is not None:
            dtype = dtype_utils.onnx_dtype_to_torch_dtype(dtype)

        self.dtype = dtype
        self.mean = mean
        self.scale = scale
        self.seed = seed

    def forward(self, *input_tensors) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if self.seed is not None:
            raise RuntimeError("The argument `seed` is not supported now.")

        dtype = input_tensors[0].dtype if self.dtype is None else self.dtype
        return torch.normal(self.mean, self.scale, input_tensors[0].shape, dtype=dtype, device=input_tensors[0].device)


@add_converter(operation_type='RandomNormalLike', version=1)
@add_converter(operation_type='RandomNormalLike', version=4)
@add_converter(operation_type='RandomNormalLike', version=11)
@add_converter(operation_type='RandomNormalLike', version=13)
@add_converter(operation_type='RandomNormalLike', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    dtype = node.attributes.get('dtype', None)
    mean = node.attributes.get("mean", 0.0)
    scale = node.attributes.get("scale", 1.0)
    seed = node.attributes.get("seed", None)

    torch_module = OnnxRandomNormalLike(
        dtype=dtype,
        mean=mean,
        scale=scale,
        seed=seed
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
