# import torch first to make jit op work without `ImportError of libc10.so`
import torch

from .jit_ops import FastCOCOEvalOp, JitOp

try:
    from .fast_coco_eval_api import COCOeval_opt
except ImportError:  #  exception will be raised when users build yolox from source
    pass
