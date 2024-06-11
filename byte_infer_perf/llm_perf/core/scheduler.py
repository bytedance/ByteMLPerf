import os
import time
import random
import threading
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any, AsyncIterable, Dict, Iterable, List, Tuple, Union

import torch
import torch.multiprocessing as mp
from torch import distributed as dist

from llm_perf.core.generation import (
    GenerateConfig,
    GenerateRequest,
    GenerateResult
)
from llm_perf.core.inferencer import CoreInferencer
from llm_perf.core.sampler import CoreSampler
from llm_perf.utils.logger import logger
from llm_perf.utils.reporter import calc_perplexity


class CoreScheduler(ABC):
    def __init__(
        self,
        inferencer: CoreInferencer,
        sampler: CoreSampler,
        task_cls=CoreInferencer.Task
    ) -> None:
        super().__init__()

        self.inferencer: CoreInferencer = inferencer
        self.sampler: CoreSampler = sampler

        self.Task = task_cls
        self.task_queue: Queue[self.Task] = Queue()

        self.started = False
        self.scheduler_thread = None


    def start(self):
        if not self.started:
            logger.info("start scheduler thread")
            self.started = True
            self.scheduler_thread = threading.Thread(target=self.scheduler_loop)
            self.scheduler_thread.start()

    def stop(self):
        if self.started:
            logger.info("stop scheduler thread")
            self.started = False
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=1.)


    @abstractmethod
    @torch.no_grad()
    def scheduler_loop(self):
        raise NotImplementedError

    async def generate(
        self, 
        req: GenerateRequest
    ) -> Union[
        AsyncIterable[GenerateResult], Tuple[AsyncIterable[GenerateResult], float, str]
    ]:
        task = self.Task(request=req)
        self.submit(task)

        async for result in self.get_packet_results(
            req.generate_config.get_input_logits, task
        ):
            yield result


    def submit(self, task):
        self.task_queue.put_nowait(task)


    async def get_packet_results(
        self, 
        get_input_logits: bool, 
        task: CoreInferencer.Task
    ) -> Union[
        AsyncIterable[GenerateResult], Tuple[AsyncIterable[GenerateResult], float, str]
    ]:
        # Save last generate token index minus 1, cann't use len(generate_ids) because may generate
        #  many times, but generate thread only schedule once, which means length bigger than last generate token index.
        _gen_get_id = 0
        while True:
            result = await task.get_result()
            if result is None:
                if task.exception:
                    raise task.exception
                break

            if get_input_logits:
                gen_ids = task.generate_ids[:_gen_get_id]
                _gen_get_id += 1
                # 1. label = input_ids + generate_ids[:-1], [:-1] is remove just generate token
                labels = torch.LongTensor(task.request.input_ids + gen_ids)
                logger.debug(
                    f"label shape: {labels.shape}, input_logits shape: {len(result.input_logits)}"
                )
                # 2. .view convert List to Tensor view
                input_logits = torch.FloatTensor(result.input_logits).view(
                    1, labels.size(-1) - 1, -1
                )
                perplexity = calc_perplexity(input_logits=input_logits, labels=labels)
                dump_file = await self.dump_last_logits(result, task)
                yield result, perplexity, dump_file
            else:
                yield result

        if get_input_logits:
            dump_file = await self.dump_last_logits(result, task)
            yield None, -1, dump_file
        return



    async def dump_last_logits(self, result, task: CoreInferencer.Task):
        """Dump prompt logits

        Args:
            result: prompt generate result
            task: prompt relate task

        Return:
            dump_file: prompt output logits numpy saved file, logits shape: [1, generate_token_len, vocab_size]
        """
        if result is None:
            tmp_dir = ".tmp_logits"
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            import numpy as np

            dump_file = (
                tmp_dir
                + "/"
                + str(random.randint(0, 100))
                + "_"
                + str(int(time.time()))
                + ".npy"
            )
            # np_logits shape is [1, seq_len, vocab_size]
            np_logits = np.expand_dims(np.array(task.all_last_logits), axis=0)
            np.save(dump_file, np_logits)
            return dump_file
        else:
            # last_logits shape is [vocab_size]
            # all_last_logits shape is [seq_len, vocab_size]
            if not hasattr(task, "all_last_logits"):
                task.all_last_logits = [result.last_logits]
            else:
                task.all_last_logits.append(result.last_logits)
            return ""


