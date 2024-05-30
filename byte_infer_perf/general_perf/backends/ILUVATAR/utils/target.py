import tvm

def get_target(target_name):
    
    target = None
    if target_name == "llvm":
        target = tvm.target.Target(target_name)
    
    elif target_name == "iluvatar":
        target = tvm.target.iluvatar(model="MR")
    
    elif target_name == "iluvatar_with_cudnn_cublas":
        target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas")
    elif target_name == "iluvatar_with_ixinfer":
        target = tvm.target.iluvatar(model="MR", options="-libs=ixinfer")
    elif target_name == "iluvatar_with_all_libs":
        target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")

    else:
        raise Exception(f"Unsupport Target name: {target_name}!")
    
    device = tvm.device(target.kind.name, 0)
    
    return target, device
