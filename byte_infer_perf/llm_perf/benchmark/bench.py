import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Iterable, List

import grpc
import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_perf.utils.logger import logger, setup_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from llm_perf import server_pb2, server_pb2_grpc
from llm_perf.utils.pb import deserialize_value, serialize_value
from llm_perf.utils.reporter import Reporter


def gen_request(
    stub: server_pb2_grpc.InferenceStub,
    prompt: str,
    min_new_tokens: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    get_input_logits: int,
) -> Iterable[server_pb2.InferenceResponse]:
    req = server_pb2.InferenceRequest(
        req_id=str(os.getpid()) + "_" + str(int(time.time())),
        inputs={
            "input_messages": serialize_value(prompt),
            "min_new_tokens": serialize_value(min_new_tokens),
            "max_new_tokens": serialize_value(max_new_tokens),
            "top_p": serialize_value(top_p),
            "top_k": serialize_value(top_k),
            "get_input_logits": serialize_value(get_input_logits),
        },
    )
    for res in stub.StreamingInference(req):
        yield res


def format_question(line) -> str:
    question = line["question"]
    for choice in ["A", "B", "C", "D"]:
        question += f"\n{choice}. {line[f'{choice}']}"
    question += "\n答案："
    return question


def bench(
    backend: str, workload: dict, model_config: dict, result_queue: mp.Queue, args
):
    dataset: pd.DataFrame = pd.read_csv(model_config["dataset"])
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = server_pb2_grpc.InferenceStub(channel)
    buffer_dump_file: List[str] = []
    for row_index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = format_question(row)
        st = time.time()
        generate_tokens_len = 0
        first_token_latency = 0
        output_messages: str = ""
        ppl: List[float] = []
        for res in gen_request(
            stub,
            question,
            workload["min_new_tokens"],
            workload["max_new_tokens"],
            args.top_p,
            args.top_k,
        ):
            if not first_token_latency:
                first_token_latency = time.time() - st

            res = {k: deserialize_value(v) for k, v in res.outputs.items()}
            output_messages += res["output_messages"]
            _ppl = res["ppl"]
            logger.debug(f"get ppl: {_ppl}")
            ppl.append(_ppl)

        dump_file = res["dump_file"]
        if dump_file != "":
            buffer_dump_file.append(dump_file)
        else:
            logger.debug(f"Dump logits diff disabled")

        generate_tokens_len = len(output_messages)

        use_time = time.time() - st
        per_token_latency = use_time / generate_tokens_len

        result = {
            "prompt_tokens": question,
            "prompt_tokens_len": len(question),
            "output_message": output_messages,
            "generate_tokens_len": generate_tokens_len,
            "first_token_latency": first_token_latency,
            "per_token_latency": per_token_latency,
            "ppl": ppl,
            "dump_file": dump_file,
        }
        logger.debug(f"prompt response: {result}")
        result_queue.put(result)

    result_queue.put(None)
    logger.debug(f"buffer numpy files: {buffer_dump_file}")


def bench_performance(
    backend: str, workload: dict, model_config: dict, result_queue: mp.Queue, args
):
    dataset: pd.DataFrame = pd.read_csv(model_config["dataset"])
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = server_pb2_grpc.InferenceStub(channel)
    for row_index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = format_question(row)
        st = time.time()
        generate_tokens_len = 0
        first_token_latency = 0
        output_messages: str = ""
        for res in gen_request(
            stub,
            question,
            workload["min_new_tokens"],
            workload["max_new_tokens"],
            args.top_p,
            args.top_k,
            0,
        ):
            if not first_token_latency:
                first_token_latency = time.time() - st

            res = {k: deserialize_value(v) for k, v in res.outputs.items()}
            output_messages += res["output_messages"]

        generate_tokens_len = len(output_messages)

        use_time = time.time() - st
        per_token_latency = use_time / generate_tokens_len

        result = {
            "prompt_tokens": question,
            "prompt_tokens_len": len(question),
            "output_message": output_messages,
            "generate_tokens_len": generate_tokens_len,
            "first_token_latency": first_token_latency,
            "per_token_latency": per_token_latency,
        }
        logger.debug(f"prompt response: {result}")
        result_queue.put(result)

    result_queue.put(None)


def bench_accuracy(
    backend: str, workload: dict, model_config: dict, result_queue: mp.Queue, args
):
    dataset: pd.DataFrame = pd.read_csv(model_config["dataset"])
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = server_pb2_grpc.InferenceStub(channel)
    for row_index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = format_question(row)
        prompt_ppl: List[float] = []
        for res in gen_request(
            stub,
            question,
            workload["min_new_tokens"],
            workload["max_new_tokens"],
            args.top_p,
            args.top_k,
            1,
        ):
            res = {k: deserialize_value(v) for k, v in res.outputs.items()}
            ppl = res["ppl"]
            prompt_ppl.append(ppl)

        dump_file = res["dump_file"]

        result = {"ppl": prompt_ppl, "dump_file": dump_file}
        logger.debug(f"prompt response: {result}")
        result_queue.put(result)

    result_queue.put(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--min-new-tokens", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--hardware_type", "-hw", type=str, required=True)
    parser.add_argument("--device_count", type=int, required=True)
    args = parser.parse_args()

    loglevel = os.environ.get("LOG_LEVEL", "debug")
    setup_logger(loglevel)

    dataset: pd.DataFrame = pd.read_csv(args.dataset)

    result_queue = mp.Queue()
    jobs: List[mp.Process] = []

    for i in range(args.batch_size):
        p = mp.Process(
            target=bench,
            args=(
                args.dataset,
                args.hardware_type,
                args.task,
                dataset,
                result_queue,
                args,
            ),
        )
        jobs.append(p)
        p.start()

    reporter = Reporter(
        args.batch_size,
        args.task,
        args.hardware_type,
        args.device_count,
        args.min_new_tokens,
        args.max_new_tokens,
    )
    reporter.start()
    while True:
        result = result_queue.get()
        if result is None:
            break

        reporter.submit(result)

    for p in jobs:
        p.join()

    reporter.stop()
    reporter.summary()


if __name__ == "__main__":
    main()
