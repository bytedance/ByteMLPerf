import importlib
import json
import os
import sys
from concurrent import futures
from typing import Any, Dict, Iterable, List

import grpc

# FIXME: If do not import torch here, torch import by `importlib.import_module` will cause torch.libs/libgomp-6e1a1d1b.so.1.0.0: cannot allocate memory in static TLS block
import torch
import transformers

from llm_perf import server_pb2, server_pb2_grpc
from llm_perf.core.infernecer import CoreInferencer
from llm_perf.utils.logger import logger
from llm_perf.utils.pb import deserialize_value, serialize_value


class Inference(server_pb2_grpc.InferenceServicer):
    def __init__(self, task: str, hardware_type: str, max_batch_size: int) -> None:
        super().__init__()

        # Get Model config
        config_path = "llm_perf/model_zoo/" + task + ".json"
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
            ".setup", package=f"llm_perf.backends.{hardware_type}"
        )
        logger.info(f"import setup: {setup}")

        model_impl = self.get_model_impl(model_config, hardware_type)

        scheduler = setup.setup_scheduler(
            model_impl, model_config, max_batch_size=max_batch_size
        )

        self.inferencer = CoreInferencer(scheduler)

    def get_model_impl(self, model_config: Dict[str, Any], hardware_type: str):
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

    def StreamingInference(
        self, request: server_pb2.InferenceRequest, context: grpc.ServicerContext
    ) -> Iterable[server_pb2.InferenceResponse]:
        logger.debug(f"StreamingInference request id {request.req_id}")

        req = {k: deserialize_value(v) for k, v in request.inputs.items()}
        prompt = req["input_messages"]
        generate_config = {
            "min_new_tokens": req["min_new_tokens"],
            "max_new_tokens": req["max_new_tokens"],
            "top_p": req["top_p"],
            "top_k": req["top_k"],
            "get_input_logits": req["get_input_logits"],
        }

        # Generating
        for result in self.inferencer.streaming_inference(
            prompt=prompt, generate_config=generate_config
        ):
            yield server_pb2.InferenceResponse(
                req_id=request.req_id,
                outputs={
                    "output_messages": serialize_value(result["token"]),
                    "ppl": serialize_value(result["ppl"]),
                    "dump_file": serialize_value(result["dump_file"]),
                },
            )


def serve(args) -> None:
    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=10))
    server_pb2_grpc.add_InferenceServicer_to_server(
        Inference(args.task, args.hardware_type, args.max_batch_size), server
    )
    server.add_insecure_port(f"[::]:{args.port}")

    server.start()
    logger.info(f"GRPC Server start at {args.port}")

    fifo_name = "./server_fifo"
    with open(fifo_name, "w") as fifo_fd:
        fifo_fd.write("Server Ready")
        fifo_fd.flush()
    server.wait_for_termination()
