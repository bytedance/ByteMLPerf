import os
import psutil
from itertools import permutations
import numpy as np

import tvm
from tvm import relay

import onnx
import onnx.helper as onnx_helper
import onnxoptimizer
from onnxsim import simplify
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      quantize_static, QuantType,
                                      CalibrationMethod)

from .onnx_util import contain_qlinear_opearator, rewrite_tensor_dim
from .onnx_rewrite_batch_size import rewrite_batch_size
from .dataloader import get_dataloader_from_args

class Node:
    def __init__(self, name, op_type, input, output):
        self.name = name
        self.op_type = op_type
        self.input = input
        self.output = output
        
        
        self.from_node = []
        self.to_node = []

    def __repr__(self) -> str:
        return f"{self.name} [{self.op_type}], input = {self.input}, output = {self.output}"

    
    @staticmethod
    def connect(node_list):
        perm = permutations(node_list, 2)
        for (i, j) in perm:
            i._connect(j)    
    
    def _connect(self, node):
        if node in self.from_node or node in self.to_node:
            return
        for output in node.output:
            if output in set(self.input):
                node.to_node.append(self)
                self.from_node.append(node)

class Model:
    @staticmethod
    def add_ms_opset_domain(model,
                            ms_opset_domain="com.microsoft",
                            ms_opset_version=1):
        found = False
        for i in model.opset_import:
            if i.domain == ms_opset_domain:
                found = True
                break

        if not found:
            ms_opset = onnx_helper.make_operatorsetid(ms_opset_domain,
                                                        ms_opset_version)
            model.opset_import.append(ms_opset)

        return model

    @staticmethod
    def preprocess_onnx(model):
        model = Model.add_ms_opset_domain(model)

        passes = onnxoptimizer.get_available_passes()

        no_need = [
            # NOTE(chen.chen): the following passes cause some error, need to debug
            "lift_lexical_references",
            "split_init",
            "split_predict",

            # we do not want to rename anything
            "rename_input_output",
            "set_unique_name_for_nodes"
        ]
        passes = [i for i in passes if i not in no_need]       
        model = onnxoptimizer.optimize(model, passes)

        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"

        # model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
        return model
    
    def __init__(self, model):
        if isinstance(model, str):
            model = onnx.load(model)
        self.model = Model.preprocess_onnx(model)
        
        self.graph = self.model.graph
        self.nodes = self.graph.node
        self.node_list = []
        for i in self.nodes:
            self.node_list.append(Node(i.name, i.op_type, i.input, i.output))
        Node.connect(self.node_list)
        
    
    
def find_detect_node(model):
    if isinstance(model, str):
        model = Model(model)
    assert isinstance(model, Model)
    
    node_list = model.node_list
    
    
    last_conv = []
    # find last conv nodes before detect
    for i in range(len(node_list) - 1, -1,  -1):
        node = node_list[i]
        if not node.op_type == "Conv":
            continue
        
        after_node = node.to_node[:]
        find_conv = False
        while after_node:
            last = after_node.pop()
            after_node.extend(last.to_node)
            
            if last.op_type == "Conv":
                find_conv = True
                break

        if not find_conv:
            last_conv.append(node)
    
    
    
    exclude_detect_node_type = [
        "Add", "Mul", "Concat",  
        # "Reshape", "Exp", "Power", "Slice", "Split" ## these node will not be quantized actually
        ]
    exclude_detect_node_name = []
    for i in last_conv:
        after_node = i.to_node[:]
        while after_node:
            last = after_node.pop()
            after_node.extend(last.to_node)
            
            if last.op_type in exclude_detect_node_type:
                exclude_detect_node_name.append(last.name)
    
    exclude_detect_node_name = sorted(list(set(exclude_detect_node_name)))
    return exclude_detect_node_name


def find_unsupported_node(model):
    if isinstance(model, str):
        model = Model(model)
    assert isinstance(model, Model)
    
    node_list = model.node_list
    
    
    igie_not_supported_node_type = [
        "Softmax",
        "Gemm", # igie onnx frontend error for mobilenetv2
    ]
    exclude_node_name = []
    for i in node_list:
        if i.op_type in igie_not_supported_node_type:
            exclude_node_name.append(i.name)
       
    return exclude_node_name


def find_group_conv_node(model):
    if isinstance(model, str):
        model = Model(model)
    assert isinstance(model, Model)
    
    nodes = model.graph.node

    exclude_node_name = []
    for node in nodes:
        if node.op_type == "Conv":
            attrs = node.attribute
            for j in attrs:
                if j.name == "group" and j.i != 1:
                    exclude_node_name.append(node.name)
       
    return exclude_node_name

class BaseDataReader(CalibrationDataReader):

    def __init__(self, dataloader, cnt_limit=500):
        # pytorch-like dataloader
        self.dataloader = dataloader
        self.cnt = 0
        self.cnt_limit = cnt_limit
        self.rewind()

    def get_next(self):
        raise NotImplementedError

    def reset_dataloader(self):
        self.dataloader_iter = iter(self.dataloader)

    def rewind(self):
        self.reset_dataloader()
        self.cnt = 0

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader
        self.rewind()

    def should_stop(self, memory_upper_bound=80):
        # avoid oom
        if BaseDataReader._exceed_memory_upper_bound(
                upper_bound=memory_upper_bound
        ) or self.cnt + 1 > self.cnt_limit:
            return True
        self.cnt += 1
        return False

    def get_next_data(self):
        data = next(self.dataloader_iter, None)
        if data is None:
            self.reset_dataloader()
            data = next(self.dataloader_iter, None)
        return data

    @staticmethod
    def _exceed_memory_upper_bound(upper_bound=90):
        # upper_bound in [0, 100]

        info = psutil.virtual_memory()
        total_percent = info.percent
        if total_percent >= upper_bound:
            return True
        return False

class ONNXDataReader(BaseDataReader):
    def __init__(self, input_name_list, dataloader, cnt_limit=500):
        self.input_name_list = input_name_list
        super().__init__(dataloader, cnt_limit)
    
    def get_next(self):
        if self.should_stop(memory_upper_bound=90):
            return None
        print(f"onnx calibration data count: {self.cnt}")
        all_input = self.get_next_data()
        
        #NOTE(chen.chen)
        # we assumen the all_input contains each input tensorin input_name_list with the same order
        assert len(all_input) >= len(self.input_name_list)
        ort_input = {k: np.array(v) for k, v in zip(self.input_name_list, all_input)}
        return ort_input
            

def fill_onnx_input_shape(model_path, input_shape_list, model_save_path=None):
    model = onnx.load(model_path)
    inputs = model.graph.input

    assert len(inputs) == len(input_shape_list), f"input number error, should be {len(inputs)}, got {len(input_shape_list)}"
    for tensor, shape in zip(inputs, input_shape_list):
        rewrite_tensor_dim(tensor, shape)
        
    model = Model.preprocess_onnx(model)
    
    if model_save_path is None:
        model_save_path = f"{model_path[:-5]}_fill_input.onnx"
    onnx.save(model, model_save_path)
    
    return model_save_path


def onnx_quantize_model_from_args(args):
    ori_model_path = args.model_path
    assert ori_model_path.endswith(".onnx")
    
    # NOTE(chen.chen)
    # we should just rewrite input_shape here since some batch_size dim of reshape op is fixed
    # ori_model_path = fill_onnx_input_shape(ori_model_path, args.input_shape_list)
    
    # skip model which has been quantized
    if contain_qlinear_opearator(ori_model_path):
        return ori_model_path
    
    # check if quantization_config is valid
    # NOTE(chen.chen)
    # if user has not specified the quantization_config
    # we should have a default config here

    config = args.quantization_config.get("onnx", {})  
    quant_format = config.get("quant_format", "qoperator").lower()
    if quant_format == "qdq":   
        quant_format = QuantFormat.QDQ
    elif quant_format == "qoperator":
        quant_format = QuantFormat.QOperator
    else:
        raise ValueError(f"invalid quant_format: {quant_format}")
    
    
    
    op_types_to_quantize = config.get("op_types_to_quantize", [])
    per_channel = config.get("per_channel", False)
    reduce_range = config.get("reduce_range", False)
    nodes_to_quantize = config.get("nodes_to_quantize", [])
    nodes_to_exclude = config.get("nodes_to_exclude", [])
    skip_group_conv_layer = config.get("skip_group_conv_layer", False)
    
    if args.automatic_yolo_quantization:
        yolo_detect_nodes = find_detect_node(ori_model_path)
        nodes_to_exclude.extend([i for i in yolo_detect_nodes if i not in nodes_to_exclude])
        
    if skip_group_conv_layer:
        group_conv_node = find_group_conv_node(ori_model_path)
        print(group_conv_node)
        nodes_to_exclude.extend([i for i in group_conv_node if i not in nodes_to_exclude])
    
    unsupport_node = find_unsupported_node(ori_model_path)
    nodes_to_exclude.extend([i for i in unsupport_node if i not in nodes_to_exclude])
    
    calibrate_method = config.get("calibrate_method", "percentile").lower()
    if calibrate_method == "minmax":
        calibrate_method=CalibrationMethod.MinMax
    elif calibrate_method == "entropy":
        calibrate_method=CalibrationMethod.Entropy
    elif calibrate_method == "percentile":
        calibrate_method=CalibrationMethod.Percentile
    else:
        raise ValueError(f"invalid calibrate_method: {calibrate_method}")
    
    quant_model_path = f"{os.path.split(ori_model_path)[1][:-5]}_quant.onnx"
    
    
    ## NOTE(chen.chen)
    ## for memory issue, we will try to change the batchsize of model to 1 during quantization
    ## but it only works for simple cv model
    ## we reserve a field for user to control this behavior to avoid some strange batch-rewriting result 
    memory_efficient_quant = config.get("memory_efficient_quant", True)
    batch_size =  args.batch_size
    if memory_efficient_quant:
        model_input = ori_model_path[:-5] + "_b1.onnx"
        rewrite_batch_size(ori_model_path, 
                           batch_size=1,
                           save_model_path=model_input)
        args.batch_size = 1
    else:
        model_input = ori_model_path
        
    dataloader = get_dataloader_from_args(args)
    
    calibrate_data_count = config.get("calibrate_data_count", 20)
    datareader = ONNXDataReader(args.input_name_list, dataloader, calibrate_data_count)
    
    args.batch_size = batch_size    
    
    if args.verbose:
        print("onnx quanziation config:")
        print("model_input: ", model_input)
        print("model_output: ", quant_model_path)
        print("quant_format: ", quant_format)
        print("op_types_to_quantize: ", op_types_to_quantize)
        print("per_channel: ", per_channel)
        print("reduce_range: ", reduce_range)
        print("nodes_to_quantize: ", nodes_to_quantize)
        print("nodes_to_exclude: ", nodes_to_exclude)
        print("calibrate_method: ", calibrate_method)
        print("skip_group_conv_layer: ", skip_group_conv_layer)
    
    symmetric_quantize(
        model_input=model_input,
        model_output=quant_model_path, 
        calibration_data_reader=datareader,
        quant_format=quant_format,
        op_types_to_quantize=op_types_to_quantize,
        per_channel=per_channel,
        reduce_range=reduce_range,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=calibrate_method)
    
    ## NOTE(chen.chen)
    ## rewrite the batchsize back to the origin batchsize
    if memory_efficient_quant: 
        rewrite_batch_size(quant_model_path, 
                           batch_size=args.batch_size,
                           save_model_path=quant_model_path)
    
    return quant_model_path




def igie_calibrate_dataset(dataloader, input_name_list, calibrate_data_count=3):
    calibration_data_list = []
    for idx, batch in enumerate(dataloader):
        if idx >= calibrate_data_count:
            break
        
        data_dict = {}
        for data, name in zip(batch, input_name_list):
            data_dict[name] = data
        
        calibration_data_list.append(data_dict)
    return calibration_data_list

def igie_quantize_model_from_args(mod, params, args):
    
    # NOTE(chen.chen)
    # we need to remove unused function for tensorflow
    from tvm.relay.transform.iluvatar import SimplifyGraph
    mod = SimplifyGraph(mod, params)
    
    
    config = args.quantization_config.get("igie", {})  
    
    
    base_name = os.path.splitext(os.path.split(args.model_path)[1])[0]
    
    scale_file_path = config.get("scale_file_path", "")
    if scale_file_path == "":
        scale_file_path = f"quantize_scale_file_{base_name}_{args.target}.npy"
    calibrate_mode = config.get("calibrate_mode", "percentile")
    weight_scale = config.get("weight_scale", "max")
    
    
    skip_first_conv_layer = config.get("skip_first_conv_layer", False)
    if args.target != "iluvatar_with_all_libs":
        skip_first_conv_layer = True
        
    skip_conv_layers = None
    if skip_first_conv_layer:
        skip_conv_layers = [0]

    skip_dense_layer = config.get("skip_dense_layer", False)
    calibrate_chunk_by = config.get("calibrate_chunk_by", -1)
    skip_group_conv_layer = config.get("skip_group_conv_layer", False)
    
    global_scale = config.get("global_scale", 0.8)
    calibrate_data_count = config.get("calibrate_data_count", 3)
    
    if args.verbose:
        print("igie quanziation config:")
        print("calibrate_mode: ", calibrate_mode)
        print("weight_scale: ", weight_scale)
        print("scale_file_path: ", scale_file_path)
        print("skip_dense_layer: ", skip_dense_layer)
        print("skip_first_conv_layer: ", skip_first_conv_layer)
        print("skip_group_conv_layer: ", skip_group_conv_layer)
        print("calibrate_chunk_by: ", calibrate_chunk_by)
        print("global_scale: ", global_scale)
        print("calibrate_data_count: ", calibrate_data_count)
    
    
    if calibrate_mode == "global_scale":
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(calibrate_mode=calibrate_mode,
                                        global_scale=global_scale,
                                        skip_conv_layers=skip_conv_layers,
                                        skip_dense_layer=skip_dense_layer):
                mod = relay.quantize.quantize(mod, params)
    
    elif calibrate_mode == "percentile" or calibrate_mode == "kl_divergence":

        dataloader = get_dataloader_from_args(args)
        dataset = igie_calibrate_dataset(dataloader, args.input_name_list, calibrate_data_count)
            
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(calibrate_mode=calibrate_mode,
                                        weight_scale=weight_scale,
                                        skip_conv_layers=skip_conv_layers,
                                        skip_dense_layer=skip_dense_layer,
                                        calibrate_chunk_by=calibrate_chunk_by,
                                        import_scale_file=scale_file_path,
                                        skip_group_conv_layers=skip_group_conv_layer):
                mod = relay.quantize.quantize(mod, params, dataset=dataset)
        
    else:
        raise ValueError(f"unsupported calibrate_mode: {calibrate_mode}")
    

    
    
    return mod, params




def _modify_symmetric(extra_options):
    if extra_options is None:
        extra_options = {"ActivationSymmetric": True, "WeightSymmetric": True}
    else:
        extra_options["ActivationSymmetric"] = True
        extra_options["WeightSymmetric"] = True

    return extra_options



def symmetric_quantize(
    model_input,
    model_output,
    calibration_data_reader: CalibrationDataReader,
    quant_format=QuantFormat.QOperator,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    optimize_model=False,
    calibrate_method=CalibrationMethod.Percentile,
    extra_options=None,
):
    extra_options = _modify_symmetric(extra_options)
    assert quant_format in [QuantFormat.QOperator, QuantFormat.QDQ]
    quantize_static(model_input,
                    model_output,
                    calibration_data_reader=calibration_data_reader,
                    quant_format=quant_format,
                    op_types_to_quantize=op_types_to_quantize,
                    per_channel=per_channel,
                    reduce_range=reduce_range,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    nodes_to_quantize=nodes_to_quantize,
                    nodes_to_exclude=nodes_to_exclude,
                    optimize_model=optimize_model,
                    use_external_data_format=False,
                    calibrate_method=calibrate_method,
                    extra_options=extra_options)
