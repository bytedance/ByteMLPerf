import argparse
import logging
import time
from typing import Dict, Optional

import onnx
from onnx import ModelProto, helper, load_model
from onnx_model_bert import BertOnnxModel
from onnx_model_roformer import RoformerOnnxModel
from onnx_model_conformer import conformerOnnxModel
from onnx_model_t5 import T5OnnxModel
from onnx_model_yolo import YoloOnnxModel
from onnxsim import simplify
from passes.fusion_options import FusionOptions
from passes.symbolic_shape_infer import SymbolicShapeInference

logger = logging.getLogger(__name__)
MODEL_TYPES = {
    "bert": (BertOnnxModel, None, "pytorch", 1),
    "swint": (BertOnnxModel, None, "pytorch", 1),
    "roformer": (RoformerOnnxModel, None, "tf2onnx", 1),
    "gpt2": (BertOnnxModel, None, "pytorch", 1),
    "t5": (T5OnnxModel, None, "tf2onnx", 1),
    "yolo": (YoloOnnxModel, None, "pytorch", 1),
    "vit": (BertOnnxModel, None, "pytorch", 1),
    "conformer": (conformerOnnxModel, None, "pytorch", 1),
}


def optimize_by_fusion(
    model: ModelProto,
    model_type: str = "bert",
    num_heads: int = 0,
    hidden_size: int = 0,
    optimization_options: Optional[FusionOptions] = None,
):
    """Optimize Model by graph fusion logic.

    Note that ONNXRuntime graph optimizations (like constant folding) will not be applied. So it is better to enable
    constant folding during exporting ONNX model, or run optimize_by_onnxruntime on the model first like optimize_model.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need specify these parameters.

    Args:
        model (ModelProto): model object
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically (for model_type "bert" only).
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically (for model_type "bert" only).
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions. Defaults to None.

     Returns:
        object of an optimizer class.
    """
    if model_type != "bert" and (num_heads == 0 or hidden_size == 0):
        logger.warning(
            "Please specify parameters of num_heads and hidden_size when model_type is not 'bert'"
        )

    (optimizer_class, transformer_class, producer, _) = MODEL_TYPES[model_type]

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f'Model producer not matched: Expected "{producer}", Got "{model.producer_name}".'
            "Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = FusionOptions(model_type)

    optimizer = optimizer_class(model, num_heads, hidden_size)

    optimizer.optimize(optimization_options)

    optimizer.topological_sort()

    return optimizer, transformer_class


def optimize_to_ixrt(args):
    onnx_name = args.onnx[:-5]
    model = onnx.load(args.onnx)

    logger.info("simplify..")
    simplified_model, check = simplify(model)
    logger.info("simplify model end...")
    if args.dump_onnx:
        onnx.save(simplified_model, onnx_name + "_sim.onnx")

    # transfer to static shape and optimize it
    static_sim_model = simplified_model
    if args.input_shapes:
        for input_tensor in simplified_model.graph.input:
            if input_tensor.name in args.input_shapes.keys():
                new_shape = args.input_shapes[input_tensor.name]
                dim_list = []
                for dim in new_shape:
                    if isinstance(dim, int):
                        dim_proto = onnx.TensorShapeProto.Dimension()
                        dim_proto.dim_value = dim
                        dim_list.append(dim_proto)
                    elif isinstance(dim, str):
                        dim_proto = onnx.TensorShapeProto.Dimension()
                        dim_proto.dim_param = dim
                        dim_list.append(dim_proto)

                del input_tensor.type.tensor_type.shape.dim[:]
                input_tensor.type.tensor_type.shape.dim.extend(dim_list)

    try:
        auto_merge = False
        if args.model_type in ["roformer"]:
            auto_merge = True
        static_model = SymbolicShapeInference.infer_shapes(
            simplified_model, 2**31 - 1, auto_merge, False, 3
        )
        static_sim_model, check = simplify(static_model)
        if args.dump_onnx:
            onnx.save(static_sim_model, onnx_name + "_sim_static_sim.onnx")
    except Exception as e:
        static_model = static_sim_model = simplified_model

    if args.dump_onnx:
        onnx.save(static_model, onnx_name + "_sim_static.onnx")

    logger.info("start fusion..")
    opt_model, _ = optimize_by_fusion(
        static_sim_model, args.model_type, args.num_heads, args.hidden_size
    )
    opt_model.save_model_to_file(onnx_name + "_end.onnx")
    logger.info("done..")


def parse_params(params_str):
    params = {}
    for item in params_str.replace(" ", "").split(","):
        key, value = item.split(":")
        params[key] = [int(x) if x.isdigit() else x for x in value.split("x")]
    return params


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", type=str, default=None, required=False, help="ONNX model file path"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=0,
        help="Used in model optimization. The num of the head used in the network",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=0,
        help="Used in model optimization. The hidden_size used in the network",
    )
    parser.add_argument(
        "--input_shapes",
        type=parse_params,
        help='Static input_shapes to the inference, format is --input_shapes "input_name1:3x224x224, input_name2:3x224x224"',
    )
    parser.add_argument(
        "--dump_onnx",
        action="store_true",
        help="Whether to dump onnx",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        choices=["bert", "swint", "roformer", "t5", "yolo", "gpt2", "vit", "conformer"],
        help="Which kind of model to optimize",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["debug", "info", "error"],
        help="Which kind of model to optimize",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    if args.log_level == "info":
        logging.basicConfig(level=logging.INFO)
    elif args.log_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)
    optimize_to_ixrt(args)
