import argparse
import os
import sys
import json
from numbers import Number

def to_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return value.lower() in ("yes", "true", "t", "1")
    elif isinstance(value, Number):
        return value != 0
    else:
        return False


def get_args_parser():

    parser = argparse.ArgumentParser()

    # always required
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="model path or model name in torchviso")

    parser.add_argument("--input",
                        type=str,
                        required=True,
                        dest="input",
                        nargs='+',
                        help="""
                            input name and shape/dtype, format shoul be input_name:input_shape or input_name:input_shape/dtype,
                            and use space to connect multiple inputs,
                            if dtype is not given, we assuem the dtype is float32
                            single input case: --input input1:1,3,224,224
                            multiple inputs case: --input input1:32,3,224,224 input2:32,100
                            miltiple inputs with differnet dtype case: --input input1:32,3,224,224/float32 input2:32,100/int64
                            """)
                        
    parser.add_argument("--precision",
                        type=str,
                        choices=["fp32", "fp16", "int8"],
                        required=True,
                        help="model inference precision")
    
    ## common optional
    parser.add_argument("--target",
                        type=str,
                        choices=["llvm", "iluvatar", "iluvatar_with_cudnn_cublas",  "iluvatar_with_ixinfer", "iluvatar_with_all_libs"],
                        default="iluvatar_with_all_libs",
                        help="""IGIE compile target
                            llvm: cpu only
                            iluvatar: gpu without any other accerelate library
                            iluvatar_with_cudnn_cublas: gpu with all accerelate library cudnn/cublas
                            iluvatar_with_ixinfer: gpu with all accerelate library ixinfer
                            iluvatar_with_all_libs: gpu with all accerelate library cudnn/cublas/ixinfer
                            """)
    
    parser.add_argument("--engine_path",
                        type=str,
                        default=None,
                        help="save path of engine, save in pwd if not provided")

    parser.add_argument("--warmup",
                        type=int,
                        default=3,
                        help="numbe of warmup before test")
    
    # parser.add_argument("--test_count",
    #                     type=int,
    #                     default=None,
    #                     help="number of batch to test, test all batch if not specified")

    parser.add_argument("--verbose",
                        type=to_bool,
                        default=False,
                        help="dump igie mod to file if is True")
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help="number of workers used in pytorch dataloader")
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=None,
                        help="""model batch size for datalodaer,
                            use the first dimension of the first input when not specified 
                            this argument will be useful for multi-input case:
                            e.g. input_ids:1000,22 pixel_values:32,3,224,224 attention_mask:1000,22
                            """)
    
    ## dataset
    parser.add_argument("--use_imagenet",
                        type=to_bool,
                        default=False,
                        help="use imagenet val dataet for calibration and test")
    
    parser.add_argument("--use_coco2017",
                        type=to_bool,
                        default=False,
                        help="use coco2017 val datatset for calibration and test")

    # parser.add_argument("--custom_data_path",
    #                     type=str,
    #                     default=None,
    #                     help="user-provided custom data path to define user's datalodaer"
    #                     )

    parser.add_argument("--input_layout",
                        type=str,
                        choices=["NHWC", "NCHW"],
                        default="NCHW",
                        help="model input layout, only works for cv model")

    parser.add_argument("--calibration_file_path",
                        type=str,
                        default=None,
                        help="user-provided calibration npy data path, only used for calibration")
    
    ## custom quantization config
    parser.add_argument("--automatic_yolo_quantization",
                        type=to_bool,
                        default=False,
                        help="automaticlly find the best strategy for yolo by skipping the yolo detect node quantization")    
    
    parser.add_argument("--quantization_config_path",
                        type=str,
                        default=None,
                        help="quantization config path for onnxruntime, should be a json file, refer to igie-doc for more infomation")    
    
    
    
    parser.add_argument("--acc_target",
                        type=float,
                        default=None,
                        help="Model inference Accuracy target.")
    
    parser.add_argument("--fps_target",
                        type=float,
                        default=None,
                        help="Model inference FPS target.")

    parser.add_argument("--perf_only",
                        type=to_bool,
                        default=False,
                        help="run performance test only")
    
    parser.add_argument('--just_export',
                        type=to_bool,
                        default=False,
                        help="just export engine and return")
    
    ## other custom option
    
    parser.add_argument("--custom_option",
                        type=str,
                        default=None,
                        dest="custom_option",
                        nargs='+',
                        help="""
                            user-provided custom key:value option, use space to connect multiple option,
                            bool value will be cast to Python bool type automaticaly,
                            single option case: --custom_option my_data_path:/local/data
                            multiple option case: --custom_option my_data_path:/local/data use_optionA:True
                            """)
    
    
    return parser



def _parse_framework(args_dict):
    model_path_or_name = args_dict["model_path"]
    framework = None
 
    # NOTE(chen.chen):
    # We rely on the suffix to distinguish the source framework of the model,
    # e.g. model.onnx, model.pb, etc. 
    
    # But if the model_path is_not exists, we will try to find it from torchvision and raise except when not found
    # e.g. resnet18, resnet50
    
    if os.path.exists(model_path_or_name):
        ext = os.path.splitext(model_path_or_name)[1]
        
        if ext == ".onnx":
            framework = "onnx"
        elif ext == ".pb":
            framework = "tensorflow"
        elif ext == ".pt":
            framework = "pytorch"
        else:
            raise ValueError(f"{ext} is not supported yet")
    else:            
        # NOTE(chen.chen)
        # paddle model saved as a directory
        # so we need check if it is a paddle model here
        paddle_model = f"{model_path_or_name}.pdmodel"
        if os.path.exists(paddle_model):
            framework = "paddle"
        else:        
            # NOTE(chen.chen):
            # we support use torchvision pretrained model
            # when model_path has no extension, we will try to find it from torchvision
            # e.g. --model_path resnet50
            framework = "pytorch"

    args_dict["model_framework"] = framework

        

def _parse_input(args_dict):
    input_list = args_dict.pop("input")    
    
    input_dict = {}
    input_name_list = []
    input_shape_list = []
    input_dtype_list = []
    batch_size = None
    for i in input_list:
        name, shape_dtype = i.rsplit(":", 1)
        if "/" in shape_dtype:
            shape, dtype = shape_dtype.split("/")
            dtype = dtype.replace("fp", "float")
            input_dtype_list.append(dtype)
        else:
            shape = shape_dtype
            input_dtype_list.append("float32")
        shape = tuple([int(j) for j in shape.split(",")])
        input_dict[name] = shape
        input_name_list.append(name)
        input_shape_list.append(shape)
        
        if batch_size is None:
            batch_size = shape[0]
    
    args_dict["input_dict"] = input_dict
    args_dict["input_name_list"] = input_name_list
    args_dict["input_shape_list"] = input_shape_list
    args_dict["input_dtype_list"] = input_dtype_list
    if args_dict["batch_size"] is None:
        args_dict["batch_size"] = batch_size


def _parse_engine_path(args_dict):
    if args_dict["engine_path"] is None:
        model_base_name = os.path.splitext(os.path.split(args_dict["model_path"])[1])[0]
        args_dict["engine_path"] = f"{model_base_name}_batchsize_{args_dict['batch_size']}_{args_dict['precision']}.so"
    assert args_dict["engine_path"].endswith("so")

   
def _parse_custom_option(args_dict):
    custom_option_dict = {}
    if args_dict["custom_option"] is not None :
        custom_option = args_dict.pop("custom_option")
        
        for option in custom_option:
            key, value = option.split(":", 1)
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif "," in value:
                value = value.split(",")
            custom_option_dict[key] = value
    
    required_pass = custom_option_dict.get("required_pass", [])
    if not isinstance(required_pass, list):
        required_pass = [required_pass]
    
    args_dict["required_pass"] = required_pass
    args_dict["custom_option"] = custom_option_dict


def _parse_dataset(args_dict):
    args_dict["use_builtin_data"] = args_dict["use_imagenet"] or args_dict["use_coco2017"]
    if not args_dict["use_builtin_data"]:
        args_dict["perf_only"] = True

def _parse_quantization_config(args_dict):
    
    quantization_config_path = args_dict["quantization_config_path"]
    if quantization_config_path is not None:
        assert os.path.exists(quantization_config_path)
        
        with open(quantization_config_path, "r") as f:
            data = json.load(f)
        args_dict["quantization_config"] = data
    else:
        args_dict["quantization_config"] = {}



def get_args(return_dict=False):   
    if sys.version_info.major != 3 and sys.version_info.minor < 7:
        raise ValueError(f"need at least python3.7, got {sys.version}")
    
    args_dict = vars(get_args_parser().parse_args())

    _parse_framework(args_dict)
    _parse_input(args_dict)
    _parse_engine_path(args_dict)
    _parse_quantization_config(args_dict)
    _parse_dataset(args_dict)
    _parse_custom_option(args_dict)
    
    from pprint import pprint
    pprint(args_dict, indent=2)  

    if return_dict:
        return args_dict
    
    return argparse.Namespace(**args_dict)
    


if __name__ == "__main__":
    # python3 argument.py --model_path=a/b/c.onnx --input input1:32,3,224,224 --precision=int8
    # python3 argument.py --model_path=a/b/c.onnx --input input1:32,3,224,224,44444 input2:32,100 --precision=int8
    # python3 argument.py --model_path=a/b/c.onnx --input input1:32,3,224,224,44444/float32 input2:32,100/int64 --precision=int8
    # python3 argument.py --model_path=a/b/c.onnx --input input1:32,3,224,224,44444/float32 input2:32,100/fp16 --precision=int8
    # python3 argument.py --model_path=a/b/c.onnx --input input1:32,3,224,224,44444 input2:32,100 --precision=int8 --custom_option my_data_path:/local/data use_optionA:True
    # python3 argument.py --model_path=a/b/c.onnx --input input1:32,3,224,224,44444 input2:32,100 --precision=int8 --custom_option my_data_path:/local/data use_optionA:True required_pass:pass1,pass2,pass3
    args = get_args(return_dict=True)
    
    from pprint import pprint
    pprint(args)
    