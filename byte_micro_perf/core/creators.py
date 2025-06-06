import sys
import pathlib
import importlib


FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent
BACKEND_DIR = MICRO_PERF_DIR.joinpath("backends")

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger


def get_backend_cls(backend_type: str):
    backend_module = importlib.import_module(
        "backends." + backend_type + ".backend_" + backend_type.lower())
    backend_cls = getattr(backend_module, "Backend" + backend_type)
    return backend_cls

def get_op_cls(backend_type: str, op_type: str):
    backend_module = importlib.import_module(
        "backends." + backend_type + ".backend_" + backend_type.lower())
    op_mapping = getattr(backend_module, "OP_MAPPING")
    if op_type not in op_mapping:
        raise ValueError("op_type {} not supported".format(op_type))
    op_cls = op_mapping[op_type]
    return op_cls



def create_backend(backend_type: str):    
    backend_cls = get_backend_cls(backend_type)
    backend_instance = backend_cls()
    backend_instance.backend_type = backend_type
    backend_instance.backend_cls = backend_cls
    backend_instance.torch_device_name = backend_instance.get_torch_device_name()
    backend_instance.device_name = backend_instance.get_device_name(0)
    backend_instance.device_count, backend_instance.avail_devices = backend_instance.get_device_count()
    return backend_instance


def create_op(op_type: str, args_dict: dict, backend, op_group=None, group_size=1):
    backend_type = backend.backend_type
    op_cls = get_op_cls(backend_type, op_type)
    op_instance = op_cls(args_dict, backend, op_group=op_group, group_size=group_size)
    return op_instance
