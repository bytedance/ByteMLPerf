import sys
import pathlib

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.vector_norm_ops import LayerNormOp

OP_MAPPING = {
    "torch": LayerNormOp
}
