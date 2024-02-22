import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any, AsyncIterable, Callable, Dict, Iterable, List

import backoff
import grpc
import pandas as pd
from tqdm import tqdm

from llm_perf.utils.logger import logger
from llm_perf.utils.reporter import ReportType

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from llm_perf import server_pb2, server_pb2_grpc
from llm_perf.utils.pb import deserialize_value, serialize_value


@backoff.on_exception(backoff.expo, Exception, factor=0.1, max_value=1, max_tries=3)
def gen_stream_request(
    stub: server_pb2_grpc.InferenceStub,
    index: int,
    prompt: str,
    min_new_tokens: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    get_input_logits: int,
) -> Iterable[server_pb2.InferenceResponse]:
    req = server_pb2.InferenceRequest(
        req_id=str(index) + "_" + str(int(time.time())),
        inputs={
            "input_messages": serialize_value(prompt),
            "min_new_tokens": serialize_value(min_new_tokens),
            "max_new_tokens": serialize_value(max_new_tokens),
            "top_p": serialize_value(top_p),
            "top_k": serialize_value(top_k),
            "get_input_logits": serialize_value(get_input_logits),
        },
    )
    for res in stub.StreamingInference(req, wait_for_ready=False):
        yield res


def format_question(line) -> str:
    question = line["question"]
    for choice in ["A", "B", "C", "D"]:
        question += f"\n{choice}. {line[f'{choice}']}"
    question += "\n答案："
    return question


def bench_accuracy(stub, workload: Dict[str, Any], result_queue: mp.Queue):
    dataset: pd.DataFrame = pd.read_csv(workload["dataset"])
    result_queue.put("@start")
    for row_index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = format_question(row)
        perplexity_list: List[float] = []
        for res in gen_stream_request(
            stub,
            index=1,
            prompt=question,
            min_new_tokens=workload["min_new_tokens"],
            max_new_tokens=workload["max_new_tokens"],
            top_p=0,
            top_k=1,  # use greedy search for accuracy bench
            get_input_logits=1,
        ):
            res = {k: deserialize_value(v) for k, v in res.outputs.items()}
            perplexity = res["choice"]["perplexity"]
            logits_dump = res["choice"]["logits_dump"]
            if not logits_dump:
                perplexity_list.append(perplexity)

        result = {"perplexity": perplexity_list, "logits_dump": logits_dump}
        logger.debug(f"prompt response: {result}")
        result_queue.put(result)

    result_queue.put(None)


def bench_performance(
    stub,
    index: int,
    workload: Dict[str, Any],
    input_tokens: int,
    result_queue: mp.Queue,
):
    result_queue.put("@start")
    perf_start = time.time()
    perf_time: int = workload["perf_time"]
    bar = tqdm(total=perf_time, unit="s")
    while perf_start + perf_time > time.time():
        prompt = "我" * input_tokens
        st = time.time()
        first_token_latency = 0
        output_messages: str = ""
        for res in gen_stream_request(
            stub,
            index=index,
            prompt=prompt,
            min_new_tokens=workload["min_new_tokens"],
            max_new_tokens=workload["max_new_tokens"],
            top_p=0,
            top_k=1,
            get_input_logits=0,
        ):
            res = {k: deserialize_value(v) for k, v in res.outputs.items()}
            output_messages += res["choice"]["message"]

            if not first_token_latency:
                first_token_latency = time.time() - st

        use_time = time.time() - st
        prompt_tokens = res["usage"]["prompt_tokens"]
        completion_tokens = res["usage"]["completion_tokens"]
        per_token_latency = use_time / completion_tokens

        result = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output_message": output_messages,
            "first_token_latency": first_token_latency,
            "per_token_latency": per_token_latency,
        }
        logger.debug(f"bench_{index} prompt response: {result}")
        result_queue.put(result)
        bar.update(use_time)

    bar.close()
    result_queue.put(None)


def benchmark(
    index: int,
    workload: Dict[str, Any],
    report_type: ReportType,
    input_tokens: int,
    result_queue: mp.Queue,
    args,
):
    logger.debug(f"{report_type.name} bench_{index} start")

    with grpc.insecure_channel(f"{args.host}:{args.port}") as channel:
        stub = server_pb2_grpc.InferenceStub(channel)
        time.sleep(random.randint(0, 10))

        try:
            if report_type == ReportType.ACCURACY:
                bench_accuracy(stub, workload, result_queue)
            elif report_type == ReportType.PERFORMANCE:
                bench_performance(stub, index, workload, input_tokens, result_queue)
        except Exception as e:
            logger.error(f"{report_type.name} bench_{index} error: {e}")
            raise e

    logger.debug(f"{report_type.name} bench_{index} finish")
