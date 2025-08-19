import sys
import pathlib

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.xccl_ops import Device2DeviceOp

OP_MAPPING = {
    "torch": Device2DeviceOp
}
