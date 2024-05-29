import os
import random
import torch
import ctypes
import numpy as np
from os.path import join, dirname, exists

import pycuda.driver as cuda
from cuda import cuda,cudart
import threading

import importlib

tensorrt = None      
Dims = None                                                                           
                          
tvm = None  

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def load_ixrt_plugin(logger=None, namespace="", dynamic_path="", model="", precision=""):
    global tensorrt
    global Dims

    if tensorrt is not None:
        return
    
    if precision == 'FP16':
        if model == 'resnet50' or model == 'bert' or model == 'albert' or model == 'deberta' or model == 'yolov5':
            tensorrt = importlib.import_module("tensorrt_legacy")
            Dims = getattr(tensorrt, "Dims")
        else:
            tensorrt = importlib.import_module("tensorrt")
            Dims = getattr(tensorrt, "Dims")
    
    if precision == 'INT8':
        tensorrt = importlib.import_module("tensorrt")
        Dims = getattr(tensorrt, "Dims")
    
    if not dynamic_path:
        dynamic_path = join(dirname(tensorrt.__file__), "lib", "libixrt_plugin.so")

    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    
    ctypes.CDLL(dynamic_path, mode=ctypes.RTLD_GLOBAL)
    tensorrt.init_libnvinfer_plugins(tensorrt.Logger(tensorrt.Logger.INFO), namespace)
    print(f"Loaded plugin from {dynamic_path}")


def build_engine(model_name, onnx_model_path, engine_path, MaxBatchSize, BuildFlag):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()

    profile = builder.create_optimization_profile()

    if model_name == 'resnet50':
        profile.set_shape(
                "input", Dims([1, 3, 224, 224]), Dims([32, 3, 224, 224]), Dims([MaxBatchSize, 3, 224, 224]))
        
    elif model_name == 'videobert':
        profile.set_shape(
            "image", Dims([1, 3, 224, 224]), Dims([32, 3, 224, 224]), Dims([MaxBatchSize, 3, 224, 224]))
        profile.set_shape(
            "text", Dims([100, 77]), Dims([100, 77]), Dims([100, 77]))
        
    elif model_name == 'yolov5':
        profile.set_shape(
                "images", Dims([1, 3, 640, 640]), Dims([32, 3, 640, 640]), Dims([MaxBatchSize, 3, 640, 640]))
    
    elif model_name == 'bert' or model_name == 'albert' or model_name == 'roberta':
        profile.set_shape(
            "input_ids.1", Dims([1, 384]), Dims([16, 384]), Dims([MaxBatchSize, 384]))
        profile.set_shape(
            "attention_mask.1", Dims([1, 384]), Dims([16, 384]), Dims([MaxBatchSize, 384]))
        profile.set_shape(
            "token_type_ids.1", Dims([1, 384]), Dims([16, 384]), Dims([MaxBatchSize, 384]))
        
    elif model_name == 'deberta':
        profile.set_shape(
            "input_ids.1", Dims([1, 384]), Dims([16, 384]), Dims([MaxBatchSize, 384]))
        profile.set_shape(
            "attention_mask.1", Dims([1, 384]), Dims([16, 384]), Dims([MaxBatchSize, 384]))
    
    elif model_name == 'widedeep':
        profile.set_shape(
            "new_numeric_placeholder:0", Dims([1, 13]), Dims([16, 13]), Dims([MaxBatchSize, 13]))
        profile.set_shape(
            "new_categorical_placeholder:0", Dims([1 * 26, 2]), Dims([16 * 26, 2]), Dims([MaxBatchSize * 26, 2]))
        profile.set_shape(
            "import/head/predictions/zeros_like:0", Dims([1, 1]), Dims([16, 1]), Dims([MaxBatchSize, 1]))
        
    elif model_name == 'conformer':
        profile.set_shape(
            "src", Dims([1, 3, 64, 512]), Dims([16, 3, 64, 512]), Dims([MaxBatchSize, 3, 64, 512]))
        profile.set_shape(
            "src_pad_mask", Dims([1, 128]), Dims([16, 128]), Dims([MaxBatchSize, 128]))
        
    elif model_name == 'roformer':
        profile.set_shape(
            "input_segment0", Dims([1, 1024]), Dims([16, 1024]), Dims([MaxBatchSize, 1024]))
        profile.set_shape(
            "input_token0", Dims([1, 1024]), Dims([16, 1024]), Dims([MaxBatchSize, 1024]))
        
    elif model_name == 'swin':
        profile.set_shape(
                "pixel_values.1", Dims([1, 3, 384, 384]), Dims([32, 3, 384, 384]), Dims([MaxBatchSize, 3, 384, 384]))

    else:
        pass

    build_config.add_optimization_profile(profile)

    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(onnx_model_path)
    
    if BuildFlag == 'FP16':
        build_config.set_flag(tensorrt.BuilderFlag.FP16)
    
    if BuildFlag == 'INT8':
        build_config.set_flag(tensorrt.BuilderFlag.INT8)

    # set dynamic shape
    num_inputs = network.num_inputs

    for i in range(num_inputs):
        if model_name == 'resnet50':
            input_tensor = network.get_input(i)
            input_tensor.shape = Dims([-1, 3, 224, 224])

        elif model_name == 'videobert':
            input_tensor = network.get_input(i)
            if i == 0:
                input_tensor.shape = Dims([-1, 3, 224, 224])
            else:
                input_tensor.shape = Dims([100, 77])

        elif model_name == 'yolov5':
            input_tensor = network.get_input(i)
            input_tensor.shape = Dims([-1, 3, 640, 640])
            network.get_input(i).dtype = tensorrt.float16

        elif model_name == 'bert' or model_name == 'albert' or model_name == 'roberta' or model_name == 'deberta':        
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

        elif model_name == 'conformer':
            input_tensor = network.get_input(i)
            if i == 0:
                input_tensor.shape = Dims([-1, 3, 64, 512])
            else:
                input_tensor.shape = Dims([-1, 128])
        
        elif model_name == 'roformer':
            input_tensor = network.get_input(i)
            input_tensor.shape = Dims([-1, 1024])

        elif model_name == 'swin':
            input_tensor = network.get_input(i)
            input_tensor.shape = Dims([-1, 3, 384, 384])

        else:
            pass

    plan = builder.build_serialized_network(network, build_config)

    with open(engine_path, "wb") as f:
        f.write(plan)

    print("***Build dynamic shape engine success!***")


def build_igie_engine(model_name, model_path, input_dict, model_framework, precision, engine_path):
    global tvm

    if tvm is not None:
        return
    
    if not os.path.exists(engine_path):
        tvm = importlib.import_module("tvm")
        from general_perf.backends.ILUVATAR.utils.import_model import import_model_to_igie

        target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
        mod, params = import_model_to_igie(model_path, input_dict, model_framework, backend='igie')
        lib = tvm.relay.build(mod, target=target, params=params, precision=precision, verbose=False)
        lib.export_library(engine_path)
    else:
        pass


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

        allocations.append(allocation)

        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)

    return inputs, outputs, allocations


# multi cores inference codes
class Task:
    def __init__(self, bs, dataset, device_id, load_fun, benchmark_fun, performance_reports, lock, framework) -> None:
        self.dataset = dataset
        self.benchmark_fun = benchmark_fun
        self.device_id = device_id
        self.performance_reports = performance_reports
        checkCudaErrors(cudart.cudaSetDevice(device_id))
        if framework != 'gpt2':
            load_fun(bs)

        self.lock = lock
        self.module = None
        

    def run(self):
        checkCudaErrors(cudart.cudaSetDevice(self.device_id))
        batch_reports = self.benchmark_fun(self.dataset)
        self.performance_reports.append(batch_reports)


class TaskThread(threading.Thread):
   def __init__(self, func, args):
      threading.Thread.__init__(self)
      self.func = func
      self.args = args
      
   def run(self):
      self.func(*self.args)


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
