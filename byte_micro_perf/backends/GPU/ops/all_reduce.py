import sys
import pathlib
from functools import partial

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.predefined_ops.all_reduce import AllReduceOp

OP_MAPPING = {
    "torch": AllReduceOp
}
