import sys
import csv
import json
import pathlib
import argparse
import logging


CUR_DIR = pathlib.Path(__file__).parent.absolute()
PRJ_ROOT_DIR = CUR_DIR.parent

sys.path.insert(0, str(PRJ_ROOT_DIR))


unique_attrs = [
    "op_name",
    "sku_name",
    "owner",
    "perf_mode"
]


def get_unique_key(
    op_name, 
    sku_name, 
    owner, 
    perf_mode, 
    *args,
    **kwargs
):
    return ".".join([
        sku_name,
        owner,
        op_name,
        perf_mode
    ]).replace(" ", "_")



arguments_map = {
    # 单目算子
    # [batch, len] --> [batch, len]
    "sin": ["dtype", "batch", "len"], 
    "cos": ["dtype", "batch", "len"],
    "exp": ["dtype", "batch", "len"],
    "exponential": ["dtype", "batch", "len"], 
    "silu": ["dtype", "batch", "len"],
    "gelu": ["dtype", "batch", "len"],
    "swiglu": ["dtype", "batch", "len"],
    # float32: float32 --> float16/bfloat16
    # float16: float16 --> float32
    # bfloat16: bfloat16 --> float32
    "cast": ["dtype", "batch", "len"],


    # 双目算子
    # [batch, len] (op) [batch, len] --> [batch, len]
    "add": ["dtype", "batch", "len"], 
    "mul": ["dtype", "batch", "len"], 
    "sub": ["dtype", "batch", "len"], 
    "div": ["dtype", "batch", "len"], 


    # 规约算子
    # [batch, len] --> [batch, len]
    "layernorm": ["dtype", "batch", "len"], 
    "softmax": ["dtype", "batch", "len"],
    # [batch, len] --> [batch, 1]
    "reduce_sum": ["dtype", "batch", "len"],
    "reduce_min": ["dtype", "batch", "len"],
    "reduce_max": ["dtype", "batch", "len"],

    # 索引算子
    # [batch, len] (op) [batch] --> [batch, len]
    "index_add": ["dtype", "batch", "len"],
    # [batch, len] --> [batch, len]
    "sort": ["dtype", "batch", "len"], 
    "unique": ["dtype", "batch", "len"], 
    "gather": ["dtype", "batch", "len"],
    "scatter": ["dtype", "batch", "len"],


    # 矩阵算子
    # [M, K] * [K, N] --> [M, N]
    "gemm": ["dtype", "M", "N", "K"], 
    # [batch, M, K] * [batch, K, N] --> [batch, M, N]
    "batch_gemm": ["dtype", "batch", "M", "N", "K"],
    # # group * {[M, K] * [K, N] = [M, N]
    "group_gemm": ["dtype", "batch", "group", "M_str", "N", "K"], 


    # 通信算子    
    # [batch, len] --> [batch, len]
    # tp_size split over batch
    "broadcast": ["dtype", "tp_size", "batch", "len"], 
    "allreduce": ["dtype", "tp_size", "batch", "len"], 
    "allgather": ["dtype", "tp_size", "batch", "len"], 
    "alltoall": ["dtype", "tp_size", "batch", "len"], 
    "reducescatter": ["dtype", "tp_size", "batch", "len"], 
    "p2p": ["dtype", "tp_size", "batch", "len"], 

    "device2host": ["dtype", "batch", "len"],
    "host2device": ["dtype", "batch", "len"]
}


target_attrs = [
    # latency in us
    "latency"
]


def get_csv_headers(op_name):
    return unique_attrs + arguments_map.get(op_name, []) + target_attrs






logger = logging.getLogger("bytemlperf_aeolus")

def setup_logger(loglevel: str):
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(loglevel.upper())
    logger.propagate = False


sku_name_mapping = {
    "MLU590-M9": "MLU590 M9",
    "MLU590-M9D": "MLU590 M9D",
    "MLU590-M9DK": "MLU590 M9D",
    "Iluvatar BI-V150": "BI-V150",
    "NVIDIA A800-SXM4-80GB": "A800 80GB SXM", 
    "NVIDIA H800": "H800 80GB SXM", 
    "NVIDIA H20": "H20 96GB SXM", 
    "Ascend910B2C": "Ascend910B2"
}

dtype_map = {
    "float": "float32", 
    "half": "float16", 
    "int": "int32"
}
















def normal_ops_func(op, sku_name, frame, perf_mode, json_data):
    if not json_data or "Error" in json_data:
        return
    dtype = json_data["Dtype"]
    if dtype in dtype_map:
        dtype = dtype_map[dtype]

    batch = json_data["Tensor Shapes"][0][0]
    len = json_data["Tensor Shapes"][0][1]
    latency = json_data["Avg latency(us)"]

    return [op, sku_name, frame, perf_mode, dtype, batch, len, latency]



def gemm_func(op, sku_name, frame, perf_mode, json_data):
    if not json_data or "Error" in json_data:
        return
    dtype = json_data["Dtype"]
    if dtype in dtype_map:
        dtype = dtype_map[dtype]

    M = json_data["Tensor Shapes"][0][0]
    K = json_data["Tensor Shapes"][0][1]
    N = json_data["Tensor Shapes"][1][1]
    latency = json_data["Avg latency(us)"]

    return [op, sku_name, frame, perf_mode, dtype, M, N, K, latency]


def batch_gemm_func(op, sku_name, frame, perf_mode, json_data):
    if not json_data or "Error" in json_data:
        return
    dtype = json_data["Dtype"]
    if dtype in dtype_map:
        dtype = dtype_map[dtype]

    batch_size = json_data["Tensor Shapes"][0][0]
    M = json_data["Tensor Shapes"][0][1]
    K = json_data["Tensor Shapes"][0][2]
    N = json_data["Tensor Shapes"][1][2]
    latency = json_data["Avg latency(us)"]

    return [op, sku_name, frame, perf_mode, dtype, batch_size, M, N, K, latency]

def group_gemm_func(op, sku_name, frame, perf_mode, json_data):
    if not json_data or "Error" in json_data:
        return
    dtype = json_data["Dtype"]
    if dtype in dtype_map:
        dtype = dtype_map[dtype]

    batch_size = json_data["Tensor Shapes"][0][0][0]
    group = len(json_data["Tensor Shapes"])

    M_list = [int(json_data["Tensor Shapes"][i][0][0]) // batch_size for i in range(group)]
    M_list_str = "/".join([str(m) for m in M_list])
    K = json_data["Tensor Shapes"][0][0][1]
    N = json_data["Tensor Shapes"][0][1][1]
    latency = json_data["Avg latency(us)"]

    return [op, sku_name, frame, perf_mode, dtype, batch_size, group, M_list_str,N, K, latency]



def ccl_ops_func(op, sku_name, frame, perf_mode, json_data):
    if not json_data or "Error" in json_data:
        return
    dtype = json_data["Dtype"]
    if dtype in dtype_map:
        dtype = dtype_map[dtype]

    tp_size = json_data["Group"]
    batch = json_data["Tensor Shapes"][0][0]
    len = json_data["Tensor Shapes"][0][1]
    latency = json_data["Avg latency(us)"]

    return [op, sku_name, frame, perf_mode, dtype, tp_size, batch, len, latency]

def d2h_h2d_func(op, sku_name, frame, perf_mode, json_data):
    if not json_data or "Error" in json_data:
        return
    dtype = json_data["Dtype"]
    if dtype in dtype_map:
        dtype = dtype_map[dtype]

    batch = json_data["Tensor Shapes"][0][0]
    len = json_data["Tensor Shapes"][0][1]
    latency = json_data["Avg latency(us)"]

    return [op, sku_name, frame, perf_mode, dtype, batch, len, latency]


post_func_map = {
    "sin": normal_ops_func,
    "cos": normal_ops_func,
    "exp": normal_ops_func,
    "exponential": normal_ops_func,
    "silu": normal_ops_func,
    "gelu": normal_ops_func,
    "swiglu": normal_ops_func,
    "cast": normal_ops_func,

    "add": normal_ops_func,
    "mul": normal_ops_func,
    "sub": normal_ops_func,
    "div": normal_ops_func,

    "layernorm": normal_ops_func,
    "softmax": normal_ops_func,
    "reduce_sum": normal_ops_func,
    "reduce_min": normal_ops_func,
    "reduce_max": normal_ops_func,

    "index_add": normal_ops_func,
    "sort": normal_ops_func,
    "unique": normal_ops_func,
    "gather": normal_ops_func,
    "scatter": normal_ops_func,

    "gemm": gemm_func,
    "batch_gemm": batch_gemm_func,
    "group_gemm": group_gemm_func,

    "broadcast": ccl_ops_func,
    "allreduce": ccl_ops_func,
    "allgather": ccl_ops_func,
    "alltoall": ccl_ops_func,
    "reducescatter": ccl_ops_func,
    "p2p": ccl_ops_func,

    "device2host": d2h_h2d_func,
    "host2device": d2h_h2d_func
}




def postprocess(op, file_list, dst_dir):
    json_data_list = [json.load(open(file)) for file in file_list]
    if not json_data_list:
        logger.error(f"no data found in {file_list}")
        return
    
    sku_name = json_data_list[0]["Device Info"]
    sku_name = sku_name_mapping.get(sku_name, sku_name)
    perf_datas = []
    for json_data in json_data_list:
        if "Performance" not in json_data:
            logger.error(f"no performance data")
            continue
        perf_data = json_data["Performance"]
        if not perf_datas:
            perf_datas = perf_data
        else:
            perf_datas.extend(perf_data)
    
    unique_name = get_unique_key(op, sku_name, "torch", "host")
    unique_csv_file = f"{unique_name}.csv"
    unique_csv_path = dst_dir / unique_csv_file
    
    with open(unique_csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(get_csv_headers(op))

        for perf_data in perf_datas:
            if op in post_func_map:
                row = post_func_map[op](op, sku_name, "torch", "host", perf_data)
                if row:
                  writer.writerow(row)



def convert_src(src, dst):
    logger.info(f"src: {src}")
    logger.info(f"dst: {dst}")

    op_data_map = {}
    for file in src.rglob("*.json"):
        dir_name = file.parent.name
        if dir_name == "gemv":
            dir_name = "gemm"
        if not dir_name in op_data_map:
            op_data_map[dir_name] = []
        op_data_map[dir_name].append(file)
    
    for op, files in op_data_map.items():
        logger.info(f"op: {op}")
        if op not in arguments_map and op != "gemv":
            logger.error(f"invalid op: {op}")
            continue
        postprocess(op, files, dst)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="./temp")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logger(args.log_level)

    src_dir = pathlib.Path(args.src).absolute()
    if not src_dir.exists():
        logger.error(f"{args.src} does not exist")
        exit(1)
    elif not src_dir.is_dir():
        logger.error(f"{args.src} is not a directory")
        exit(1)

    output_dir = pathlib.Path(args.output_dir).absolute()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        logger.error(f"{args.output_dir} is not a directory")
        exit(1)

    convert_src(src_dir, output_dir)
