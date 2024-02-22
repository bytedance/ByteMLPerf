import importlib
import json
import os
import sys
from concurrent import futures
from typing import Any, AsyncIterable, Dict, Iterable, List

import grpc

# FIXME: If do not import torch here, torch import by `importlib.import_module` will cause torch.libs/libgomp-6e1a1d1b.so.1.0.0: cannot allocate memory in static TLS block
import torch
import transformers

from llm_perf import server_pb2, server_pb2_grpc
from llm_perf.core.infernecer import CoreInferencer
from llm_perf.utils.logger import logger
from llm_perf.utils.pb import deserialize_value, serialize_value


class Inference(server_pb2_grpc.InferenceServicer):
    def __init__(self, inferencer: CoreInferencer) -> None:
        super().__init__()
        self.inferencer = inferencer

    async def StreamingInference(
        self, request: server_pb2.InferenceRequest, context: grpc.ServicerContext
    ) -> AsyncIterable[server_pb2.InferenceResponse]:
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
        async for result in self.inferencer.streaming_inference(
            prompt=prompt, generate_config=generate_config
        ):
            yield server_pb2.InferenceResponse(
                req_id=request.req_id,
                outputs={k: serialize_value(v) for k, v in result.items()},
            )


async def serve(args, inferencer: CoreInferencer) -> None:
    workers = min(os.cpu_count(), 100)
    server = grpc.aio.server(
        migration_thread_pool=futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="server"
        )
    )
    server_pb2_grpc.add_InferenceServicer_to_server(Inference(inferencer), server)
    server.add_insecure_port(f"[::]:{args.port}")

    await server.start()
    logger.info(f"GRPC Server start at {args.port}")

    fifo_name = "./server_fifo"
    with open(fifo_name, "w") as fifo_fd:
        fifo_fd.write("Server Ready")
        fifo_fd.flush()
    await server.wait_for_termination()
