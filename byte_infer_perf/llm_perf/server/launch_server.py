import argparse
import asyncio
import json
import os
import sys


# ${prj_root}/byte_infer_perf
BYTE_MLPERF_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)


from llm_perf.server.endpoint import LLMPerfEndpoint
from llm_perf.server.serve import serve
from llm_perf.utils.logger import logger, setup_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", "-t", type=str, required=True)
    parser.add_argument(
        "--hardware_type", "-hw", type=str, required=True)
    parser.add_argument(
        "--port", "-p", type=str, required=True)
    parser.add_argument(
        "--max_batch_size", type=int, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    loglevel = os.environ.get("LOG_LEVEL", "info")
    setup_logger(loglevel)

    # load model config
    config_path = "llm_perf/model_zoo/" + args.task + ".json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
            logger.info(f"load model config: {config_path}")
    else:
        logger.error(
            f"{config_path} not exists! The json file corresponding to the task {args.task} cannot be found. \
            Please confirm whether the file path is correct."
        )
        sys.exit(1)

    # create llm_perf endpoint
    inferencer = LLMPerfEndpoint(
        model_config=model_config, 
        hardware_type=args.hardware_type, 
        max_batch_size=args.max_batch_size
    )

    # start server
    asyncio.run(serve(args, inferencer))

if __name__ == "__main__":
    main()
