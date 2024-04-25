import os
import tvm

from .import_model import import_model_to_igie
from .target import get_target


# a simple wrapper for compile engine and get module
def compile_engine_from_args(args):
    target, device = get_target(args.target)
    
    if not os.path.exists(args.engine_path):
        mod, params = import_model_to_igie(args.model_path, args.input_dict, args.model_framework)
        lib = tvm.relay.build(mod, target=target, params=params, precision=args.precision, verbose=args.verbose, required_pass=args.required_pass)
        lib.export_library(args.engine_path)
    else:
        lib = tvm.runtime.load_module(args.engine_path)   
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))
    return module