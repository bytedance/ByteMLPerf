import sys
import pathlib
from functools import partial

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


try:
    from flash_attn import flash_attn_func

    # https://github.com/Dao-AILab/flash-attention
    class FA2Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flash_attn_v2"]

        def flash_attention_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            v = tensor_mapping["v"]
            out = flash_attn_func(
                q, k, v, 
                causal=True
            )
            return out
    
    OP_MAPPING["flash_attn_v2"] = FA2Op
except:
    pass


try:
    from flash_attn_interface import flash_attn_func

    # https://github.com/Dao-AILab/flash-attention
    class FA3Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flash_attn_v3"]

        def flash_attention_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            v = tensor_mapping["v"]
            out = flash_attn_func(
                q, k, v, 
                causal=True
            )
            return out
    
    OP_MAPPING["flash_attn_v3"] = FA3Op
except:
    pass