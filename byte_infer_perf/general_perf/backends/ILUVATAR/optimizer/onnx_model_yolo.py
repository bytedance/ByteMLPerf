# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import List, Optional

from onnx import ModelProto
from passes.fuse_series_bias_add import FusionSerialBiasAdd
from passes.fusion_customfc import FusionCustomFC, FusionCustomFCActivation
from passes.fusion_fastgelu import FusionFastGelu
from passes.fusion_format_roformer import (
    FusionFormatInvalidMask,
    FusionRemoveUselessElementwise,
)
from passes.fusion_gelu import FusionGelu
from passes.fusion_gelu_approximation import FusionGeluApproximation
from passes.fusion_layernorm import FusionLayerNormalization, FusionLayerNormalizationTF
from passes.fusion_options import FusionOptions
from passes.fusion_qordered_attention import FusionQOrderedAttention
from passes.fusion_qordered_gelu import FusionQOrderedGelu
from passes.fusion_qordered_layernorm import FusionQOrderedLayerNormalization
from passes.fusion_reshape import FusionReshape
from passes.fusion_shape import FusionShape
from passes.fusion_utils import FusionUtils
from passes.fusion_yolov5_decoder import FusionYoloV5Decoder
from passes.onnx_model import OnnxModel

logger = getLogger(__name__)


class YoloOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize BERT ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (
            num_heads > 0 and hidden_size % num_heads == 0
        )
        super().__init__(model)
        self.utils = FusionUtils(self)

    def fuse_format_roformer(self):
        FusionRemoveUselessElementwise(self).apply()
        fusion = FusionFormatInvalidMask(self)
        fusion.apply()

    def fuse_custom_fc(self):
        fusion = FusionCustomFC(self)
        fusion.apply()

    def fuse_custom_fc_activation(self):
        fusion = FusionCustomFCActivation(self)
        fusion.apply()

    def fuse_swinT_serial_bias_add(self):
        fusion = FusionSerialBiasAdd(self)
        fusion.apply()

    def fuse_gelu(self):
        fusion = FusionGelu(self)
        fusion.apply()
        fusion = FusionFastGelu(self)
        fusion.apply()
        # Only relevant in models with Q-DQ nodes
        fusion = FusionQOrderedGelu(self)
        fusion.apply()

    def fuse_reshape(self):
        fusion = FusionReshape(self)
        fusion.apply()

    def fuse_shape(self):
        fusion = FusionShape(self)
        fusion.apply()

    def fuse_layer_norm(self):
        fusion = FusionLayerNormalization(self, 0)
        fusion.apply()

        fusion = FusionLayerNormalizationTF(self)
        fusion.apply()

        # Only relevant in models with Q-DQ nodes
        fusion = FusionQOrderedLayerNormalization(self)
        fusion.apply()

    def optimize(
        self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False
    ):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()

        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()

        self.fuse_reshape()

        FusionYoloV5Decoder(self).apply()
        self.remove_unused_constant()
        logger.info(f"opset version: {self.get_opset_version()}")
