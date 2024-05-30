import os
import shutil
import onnx
import torch
import torchvision
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

import tvm
from tvm import relay
import tvm.relay.testing.tf as tf_testing
from .onnx_util import get_batch_size, rewrite_int64_input_to_int32
from .onnx_rewrite_batch_size import rewrite_batch_size
from .argument import to_bool
from tvm.relay.transform.iluvatar import SimplifyGraph

def import_model_to_igie(model_path_or_name, input_dict, model_framework):
    
    base_name = os.path.splitext(os.path.split(model_path_or_name)[1])[0]
    cache_hash = f"{base_name}_cache_dir"
    mod_path = os.path.join(cache_hash, "mod.cache")
    params_path = os.path.join(cache_hash, "params.cache")
    
    # find cached mod and params
    if os.path.exists(cache_hash) and to_bool(os.environ.get("IGIE_USE_CACHE", False)):
        with open(mod_path, "r") as mod_file:
            mod = tvm.parser.fromtext(mod_file.read())
        
        with open(params_path, "rb") as params_file:
            params = relay.load_param_dict(params_file.read())

        return mod, params
    
    paddle_dir_path = os.path.split(model_path_or_name)[0]
    if os.path.exists(model_path_or_name) or os.path.exists(paddle_dir_path):
        if model_framework == "onnx":
            batch_size = list(input_dict.values())[0][0]
            model_path = model_path_or_name
            
                  
            # we don't want to handle multi_input case here,
            # e.g. input_ids:1000,22 pixel_values:32,3,224,224 attention_mask:1000,22 for clip model
            if len(input_dict) == 1:
                batch_size_from_model = get_batch_size(model_path_or_name)
                if isinstance(batch_size_from_model, int) and batch_size_from_model != batch_size:
                    model_path = f"{model_path[:-5]}_rewrite_b{batch_size}.onnx"
                    rewrite_batch_size(model_path_or_name, batch_size, save_model_path=model_path)

            model = onnx.load(model_path)
            # model = rewrite_int64_input_to_int32(model)
            mod, params = relay.frontend.from_onnx(model, input_dict, freeze_params=True)
    
        elif model_framework == "pytorch":
            scripted_model = torch.jit.load(model_path_or_name).eval()
            input_infos = [(k, v) for k, v in input_dict.items()]
            mod, params = relay.frontend.from_pytorch(scripted_model, input_infos=input_infos)
    
        elif model_framework == "tensorflow":
            with tf_compat_v1.gfile.GFile(model_path_or_name, "rb") as f:
                graph_def = tf_compat_v1.GraphDef()
                graph_def.ParseFromString(f.read())
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)
            mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_dict)
            
        elif model_framework == "paddle":
                import paddle
                model = paddle.jit.load(model_path_or_name)
                mod, params = relay.frontend.from_paddle(model, input_dict)
        else:
            raise ValueError(f"framwork {model_framework} is not supported yet")
        
    else:
        # In this case we will try to find from tochvision
        # e.g. model_path_or_name="resnet18"

        try:
            import ssl 
            ssl._create_default_https_context = ssl._create_unverified_context
            model = getattr(torchvision.models, model_path_or_name)(pretrained=True).eval()
        except:
            raise ValueError(f"can not find model {model_path_or_name} from torchvision and current working directory")
        
        
        input_datas = []
        for shape in input_dict.values():
            # currently torchvision model should always use float32 input
            input_datas.append(torch.randn(shape))
        
        scripted_model = torch.jit.trace(model, tuple(input_datas)).eval()
        input_infos = [(k, v) for k, v in input_dict.items()]
        mod, params = relay.frontend.from_pytorch(scripted_model, input_infos=input_infos) 

    # save cache
    if to_bool(os.environ.get("IGIE_USE_CACHE", False)):
        if os.path.exists(cache_hash):
            shutil.rmtree(cache_hash)
        os.makedirs(cache_hash)
        
        mod_path = os.path.join(cache_hash, "mod.cache")
        with open(mod_path, "w") as mod_file:
            mod_file.write(mod.astext())

        params_path = os.path.join(cache_hash, "params.cache")
        with open(params_path, "wb") as params_file:
            params_file.write(relay.save_param_dict(params))
    
    # need SimlifyGraph mod when importing onnx models, especially the model contains Q/DQ node
    mod = SimplifyGraph(mod, params)   
    
    return mod, params