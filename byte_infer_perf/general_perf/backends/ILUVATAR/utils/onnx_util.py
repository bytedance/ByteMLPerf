import onnx
from collections import defaultdict

import onnx
import os

## FYI
ONNX_DTYPE = {
    0: onnx.TensorProto.FLOAT,
    1: onnx.TensorProto.FLOAT,
    2: onnx.TensorProto.UINT8,
    3: onnx.TensorProto.INT8,
    4: onnx.TensorProto.UINT16,
    5: onnx.TensorProto.INT16,
    6: onnx.TensorProto.INT32,
    7: onnx.TensorProto.INT64,
    8: onnx.TensorProto.STRING,
    9: onnx.TensorProto.BOOL,
}


def rewrite_tensor_dim(tensor, dim_value_dict):
    if isinstance(dim_value_dict, list):
        dim_value_dict = {idx: i for idx, i in enumerate(dim_value_dict)}
    all_dim = tensor.type.tensor_type.shape.dim
    for idx, value in dim_value_dict.items():
        if isinstance(value, str):
            all_dim[idx].dim_param = "batch"
        else:
            all_dim[idx].dim_value = value


def rewrite_tensor_batch_size(tensor, batch_size):

    dim_value_dict = {0: batch_size}
    rewrite_tensor_dim(tensor, dim_value_dict)


def get_tensor_dim(tensor):
    dims = []
    all_dim = tensor.type.tensor_type.shape.dim
    rank = len(all_dim)
    for i in range(rank):
        if all_dim[i].dim_value:
            dims.append(all_dim[i].dim_value)
        else:
            dims.append(all_dim[i].dim_param)
    return dims


def get_tensor_name(tensor):
    return tensor.name


def nchw_dim_to_nhwc_dim(dim_list):
    assert len(dim_list) == 4
    new_dim = [dim_list[0], dim_list[2], dim_list[3], dim_list[1]]
    return new_dim


def get_input_number(model):
    if isinstance(model, str):
        model = onnx.load(model)
    inputs = model.graph.input
    return len(inputs)

def get_batch_size(model):
    if isinstance(model, str):
        model = onnx.load(model)
    inputs = model.graph.input
    return get_tensor_dim(inputs[0])[0]


def count_op_type(model):
    if isinstance(model, str):
        model = onnx.load(model)

    nodes = model.graph.node

    node2count = defaultdict(int)
    for i in nodes:
        node2count[i.op_type] += 1

    return node2count


def contain_qlinear_opearator(onnx_model):
    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    nodes = onnx_model.graph.node

    for i in nodes:
        op_type = i.op_type.lower()
        if op_type.startswith("qlinear") or op_type.startswith("qgemm"):
            return True
    return False


def get_all_node_name(model, exclude_constant=False, pretty_print=False):
    if isinstance(model, str):
        model = onnx.load(model)

    nodes = model.graph.node
    if exclude_constant:
        all_node = [i.name for i in nodes if i.op_type != "Constant"]
    else:
        all_node = [i.name for i in nodes]

    all_node.sort()
    if pretty_print:
        res = [f'"{i}"' for i in all_node]
        res = ",\n".join(res)
        res = f'[\n{res}\n]'
        print(res)

    return all_node

def rewrite_int64_input_to_int32(model):
    inputs = model.graph.input
    
    for i in inputs:
        if i.type.tensor_type.elem_type == 7:
            i.type.tensor_type.elem_type = 6
    
    print(inputs)
    import pdb;pdb.set_trace()
    onnx.checker.check_model(model)

    return model