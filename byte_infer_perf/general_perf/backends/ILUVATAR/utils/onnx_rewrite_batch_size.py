"""
rewrite src onnx model and infer shape if possible, current sypport

1. rewrite batch_size, e.g 1x3x640x640 -> 32x3x640x640

Attention:
1. all inputs/outputs batchszie dim will be modified together, which means some NLP/Audio senquence models will introduce problems


"""
import onnx
from onnx import OperatorSetIdProto
import onnx.numpy_helper

import onnxoptimizer
from onnxsim import simplify

from .onnx_util import get_batch_size, rewrite_tensor_batch_size

def rewrite_batch_size(model,
                       batch_size,
                       modify_reshape_dim=True,
                       save_model_path=None):

    ## rewrite input and output
    if isinstance(model, str):
        model = onnx.load(model)

        
    ## there is a issue that when the onnx model comes from tf,
    ## some shape info is stored as constant node's output instead of initializer
    passes = [
        "extract_constant_to_initializer", "eliminate_unused_initializer"
    ]
    model = onnxoptimizer.optimize(model, passes)
    
    

    # to support qlinear op if the opset_import is not supported
    # if we have some ohter domains need to import, add them here
    ms_opset = OperatorSetIdProto()
    ms_opset.domain = "com.microsoft"
    ms_opset.version = 1

    ori_opset_import = model.opset_import

    if ms_opset not in ori_opset_import:
        ori_opset_import.append(ms_opset)

    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    

    graph = model.graph
    initializer = graph.initializer
    inputs = graph.input
    outputs = graph.output
    nodes = graph.node

    ori_batch_size = get_batch_size(model)

    ## in case that some onnx model inputs contain initializers' shape info, we will remove them to avoid rewriting input failure

    initializer_names = set([i.name for i in initializer])
    import copy
    tmp_inputs = copy.deepcopy(inputs)
    for i in tmp_inputs:
        if i.name in initializer_names:
            inputs.remove(i)

    for i in inputs:
        rewrite_tensor_batch_size(i, batch_size)

    for i in outputs:
        rewrite_tensor_batch_size(i, batch_size)

    ## we may need to modify reshape initializer if we modify input batchsize
    ## this code only works when the target shape is fixed, and occurs as a input initializer in the node
    ## so this may introduce some other problems when the purpose of reshape operations are totally different

    if modify_reshape_dim:
        reshape_input = []
        for idx, i in enumerate(nodes):
            if i.op_type == "Reshape":
                reshape_input.extend(i.input)
            if i.op_type == "Resize" and len(i.input) == 4:
                reshape_input.append(i.input[3])
        for idx, i in enumerate(initializer):
            if i.name in reshape_input:
                shape = onnx.numpy_helper.to_array(i).copy()
                if shape.dtype == "int64":
                    shape[0] = batch_size
                    initializer[idx].CopyFrom(
                        onnx.numpy_helper.from_array(shape, i.name))

    for i in graph.value_info:
        if i.type.tensor_type.shape.dim:
            if i.type.tensor_type.shape.dim[0].dim_value == ori_batch_size:
                i.type.tensor_type.shape.dim[0].dim_value = batch_size

    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    model = onnx.shape_inference.infer_shapes(model,
                                              check_type=True,
                                              strict_mode=True,
                                              data_prop=True)
    onnx.checker.check_model(model)

    if save_model_path:
        onnx.save(model, save_model_path)
    return model

