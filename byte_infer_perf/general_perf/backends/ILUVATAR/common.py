import random
import torch
import time
import ctypes
import argparse
import numpy as np
from os.path import join, dirname, exists

import tensorrt
from tensorrt import Dims
import pycuda.driver as cuda
from cuda import cuda,cudart
from datasets import load_dataset
from torch.utils.data import SequentialSampler, DataLoader
from transformers import DataCollatorForLanguageModeling, BertTokenizer


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def load_ixrt_plugin(logger=tensorrt.Logger(tensorrt.Logger.INFO), namespace="", dynamic_path=""):
    if not dynamic_path:
        dynamic_path = join(dirname(tensorrt.__file__), "lib", "libixrt_plugin.so")

    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    
    ctypes.CDLL(dynamic_path, mode=ctypes.RTLD_GLOBAL)
    tensorrt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")


def build_engine(model_name, onnx_model_path, engine_path, MaxBatchSize):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()

    profile = builder.create_optimization_profile()

    if model_name == 'resnet50':
        profile.set_shape(
                "input", Dims([1, 3,224,224]), Dims([32, 3,224,224]), Dims([64, 3,224,224]))
        
    elif model_name == 'yolov5':
        profile.set_shape(
                "images", Dims([1, 3,640,640]), Dims([32, 3,640,640]), Dims([64, 3,640,640]))
    
    elif model_name == 'bert':
        profile.set_shape("input_ids.1", Dims([1, 384]), Dims([16, 384]), Dims([MaxBatchSize, 384]))
        profile.set_shape("attention_mask.1", Dims([1, 384]), Dims([16, 384]), Dims([MaxBatchSize, 384]))
        profile.set_shape("token_type_ids.1", Dims([1, 384]), Dims([16, 384]), Dims([MaxBatchSize, 384]))
    
    elif model_name == 'widedeep':
        profile.set_shape(
            "new_numeric_placeholder:0", Dims([MaxBatchSize, 13]), Dims([MaxBatchSize, 13]), Dims([MaxBatchSize, 13]))
        profile.set_shape(
            "new_categorical_placeholder:0", Dims([MaxBatchSize * 26, 2]), Dims([MaxBatchSize * 26, 2]), Dims([MaxBatchSize * 26, 2]))
        profile.set_shape(
            "import/head/predictions/zeros_like:0", Dims([MaxBatchSize, 1]), Dims([MaxBatchSize, 1]), Dims([MaxBatchSize, 1]))
    else:
        pass

    build_config.add_optimization_profile(profile)

    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(onnx_model_path)
    build_config.set_flag(tensorrt.BuilderFlag.FP16)

    # set dynamic
    num_inputs = network.num_inputs

    for i in range(num_inputs):
        if model_name == 'resnet50':
            input_tensor = network.get_input(i)
            input_tensor.shape = Dims([-1, 3, 224, 224])

        elif model_name == 'yolov5':
            input_tensor = network.get_input(i)
            input_tensor.shape = Dims([-1, 3, 640, 640])
            network.get_input(i).dtype = tensorrt.float16

        elif model_name == 'bert':        
            input_tensor = network.get_input(i)
            input_tensor.shape = Dims([-1, 384])
        
        elif model_name == 'widedeep':
            input_tensor = network.get_input(i)
            if i == 0:
                input_tensor.shape = Dims([-26, 2])
            elif i == 1:
                input_tensor.shape = Dims([-1, 13])
            else:
                input_tensor.shape = Dims([-1, 1])

        else:
            pass

    plan = builder.build_serialized_network(network, build_config)

    with open(engine_path, "wb") as f:
        f.write(plan)

    print("Build dynamic shape engine done!")


def init_by_tensorrt(engine_path):
    datatype = tensorrt.DataType.FLOAT
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    with open(engine_path, "rb") as f, tensorrt.Runtime(logger) as runtime:
        runtime = tensorrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context
    
    return engine, context


def setup_io_bindings(engine, context):
    # Setup I/O bindings
    inputs = []
    outputs = []
    allocations = []

    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True

        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = context.get_binding_shape(i)

        if is_input:
            batch_size = shape[0]
        size = np.dtype(tensorrt.nptype(dtype)).itemsize

        for s in shape:
            size *= s
        
        # allocation = cuda.mem_alloc(size)
        err, allocation = cudart.cudaMalloc(size)
        assert err == cudart.cudaError_t.cudaSuccess
        
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(tensorrt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
            "nbytes": size
        }
        # print(
        #     f"binding {i}, name : {name}  dtype : {np.dtype(tensorrt.nptype(dtype))}  shape : {list(shape)}"
        # )
        allocations.append(allocation)

        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)

    return inputs, outputs, allocations


def tensorrt_infer_dynamic(engine, context, input_ids, token_type_ids):
    input_names = [
        "input_ids",
        "token_type_ids"
    ]

    # set dynamic shape
    for input_name in input_names:
        if input_name == "input_ids":
            input_shape = input_ids.shape
        elif input_name == "token_type_ids":
            input_shape = token_type_ids.shape

        input_idx = engine.get_binding_index(input_name)
        context.set_binding_shape(input_idx, Dims(input_shape))

    # Setup I/O bindings
    inputs, outputs, allocations = setup_io_bindings(engine, context)

    ### infer
    # Prepare the output data
    output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])

    # Process I/O and execute the network
    cuda.memcpy_htod(inputs[0]["allocation"], input_ids)
    cuda.memcpy_htod(inputs[1]["allocation"], token_type_ids)

    torch.cuda.synchronize()
    time_start = time.time()
    context.execute_v2(allocations)
    torch.cuda.synchronize()
    time_each = time.time() - time_start

    cuda.memcpy_dtoh(output, outputs[0]["allocation"]) 

    return output, time_each