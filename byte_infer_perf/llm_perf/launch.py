import argparse
import asyncio
import importlib
import json
import os
import sys
from typing import Any, Dict

BYTE_MLPERF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)

from llm_perf.core.infernecer import CoreInferencer
from llm_perf.server.serve import serve
from llm_perf.utils.logger import logger, setup_logger


def get_model_impl(model_config: Dict[str, Any], hardware_type: str):
    model_inferface = model_config["model_interface"]
    model_name = model_config["model_name"]

    # Get orig model
    spec = importlib.util.spec_from_file_location(
        model_name, f"llm_perf/model_zoo/{model_name.split('-')[0]}.py"
    )
    base_module_impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_module_impl)

    orig_model = getattr(base_module_impl, model_inferface)

    # Get vendor model
    vendor_model_path = f"llm_perf/backends/{hardware_type}/model_impl"
    if not os.path.exists(vendor_model_path):
        logger.info(
            f"{vendor_model_path} not exists, {model_inferface} model select <ByteMLPerf base model>"
        )
        return orig_model

    vendor_model_impl = importlib.import_module(
        ".", package=vendor_model_path.replace("/", ".")
    )
    if not model_name in vendor_model_impl.__all__.keys():
        logger.info(
            f"can't find {model_name} in {vendor_model_path}/__init__, model select <ByteMLPerf base model>"
        )
        return orig_model

    vendor_model = vendor_model_impl.__all__[model_name]
    logger.info(
        f"find {model_inferface} in {vendor_model_path}, model select <Vendor model>"
    )
    return vendor_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, required=True)
    parser.add_argument("--hardware_type", "-hw", type=str, required=True)
    parser.add_argument("--port", "-p", type=str, required=True)
    parser.add_argument("--max_batch_size", type=int, required=True)
    args = parser.parse_args()

    loglevel = os.environ.get("LOG_LEVEL", "info")
    setup_logger(loglevel)

    # Get Model config
    config_path = "llm_perf/model_zoo/" + args.task + ".json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
            logger.info(f"load model config: {config_path}")
    else:
        logger.error(
            f"{config_path} not exists! The json file corresponding to the task {task} cannot be found. \
            Please confirm whether the file path is correct."
        )
        sys.exit(1)

    setup = importlib.import_module(
        ".setup", package=f"llm_perf.backends.{args.hardware_type}"
    )
    logger.info(f"import setup: {setup}")

    model_impl = get_model_impl(model_config, args.hardware_type)

    scheduler = setup.setup_scheduler(
        model_impl, model_config, max_batch_size=args.max_batch_size
    )

    inferencer = CoreInferencer(scheduler)

    asyncio.run(serve(args, inferencer))


if __name__ == "__main__":
    main()
