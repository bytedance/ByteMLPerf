import sys
import pathlib
import importlib

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[1])
)

def create_backend_instance(backend_type: str):    
    backend_module = importlib.import_module(
        "backends." + backend_type + ".backend_" + backend_type.lower())
    backend_cls = getattr(backend_module, "Backend" + backend_type)

    backend_instance = backend_cls()
    backend_instance.backend_type = backend_type
    backend_instance.backend_cls = backend_cls
    backend_instance.torch_device_name = backend_instance.get_torch_device_name()
    backend_instance.device_name = backend_instance.get_device_name(0)
    backend_instance.device_count, backend_instance.avail_devices = backend_instance.get_device_count()
    backend_instance.env_dict = backend_instance.get_backend_env()
    return backend_instance




"""
add: {
    "is_concurrent": False,
    "op_mapping: {}
"""

# collect all backends OP_MAPPING
TARGET_BACKEND_OP_MAPPING = {
    "rms_norm": {"is_concurrent": False, "op_mapping": {}},

    # ccl_ops
    "all_reduce": {"is_concurrent": True, "op_mapping": {}},
}

BACKEND_DIR = pathlib.Path(__file__).absolute().parents[1].joinpath("backends")
for backend_dir in BACKEND_DIR.iterdir():
    if not backend_dir.is_dir():
        continue
    try:
        backend_name = backend_dir.name
        backend_module = importlib.import_module(
            "backends." + backend_name + ".backend_" + backend_name.lower())
        backend_op_mapping = getattr(backend_module, f"{backend_name}_OP_MAPPING")
        for op_name in TARGET_BACKEND_OP_MAPPING:
            if op_name in backend_op_mapping:
                TARGET_BACKEND_OP_MAPPING[op_name]["op_mapping"][backend_name] = backend_op_mapping[op_name]
    except:
        continue


def get_op_info(backend_type: str, op_type: str):
    if op_type not in TARGET_BACKEND_OP_MAPPING:
        return False, []
    op_item = TARGET_BACKEND_OP_MAPPING[op_type]
    return op_item["is_concurrent"], op_item["op_mapping"].get(backend_type, [])


def create_op_instance(
    op_cls, args_dict, backend_instance, 
    op_group=None, group_size=1
):
    op_instance = op_cls(
        args_dict, backend_instance, 
        op_group=op_group, group_size=group_size
    )
    return op_instance