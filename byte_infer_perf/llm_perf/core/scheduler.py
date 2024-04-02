import os
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any, AsyncIterable, Dict, Iterable, List, Tuple, Union

import torch
import torch.multiprocessing as mp
from torch import distributed as dist

import llm_perf.core.common as core_comm
from llm_perf.core.common import (
    GenerateConfig,
    GenerateRequest,
    GenerateResult,
    Packet,
    PacketStatus,
)
from llm_perf.core.engine import CoreEngine
from llm_perf.core.sampler import CoreSampler
from llm_perf.utils.logger import logger
from llm_perf.utils.reporter import calc_perplexity


class CoreScheduler(ABC):
    def __init__(
        self,
        engine: CoreEngine,
        sampler: CoreSampler,
        comm=core_comm,
        **kwargs,
    ) -> None:
        super().__init__()
        self.Packet = comm.Packet
        self.packet_queue: Queue[self.Packet] = Queue()
        self.engine: CoreEngine = engine
        self.sampler: CoreSampler = sampler

    def start(self):
        if dist.is_initialized():
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.local_rank = 0
            self.world_size = 1

        if self.local_rank > 0:
            self.worker_loop()
        else:
            self.scheduler_thread = threading.Thread(target=self.scheduler_loop)
            self.scheduler_thread.setDaemon(True)
            self.scheduler_thread.start()

    def stop(self):
        self.scheduler_thread.join()

    @abstractmethod
    @torch.inference_mode()
    def scheduler_loop(self):
        raise NotImplementedError

    @torch.inference_mode()
    def worker_loop(self):
        try:
            while True:
                self.engine.do_inference([])
        except Exception as e:
            logger.info(f"worker {self.local_rank} exit: {e}")

    def submit(self, packet):
        self.packet_queue.put_nowait(packet)

    async def dump_last_logits(self, result, packet: Packet):
        """Dump prompt logits

        Args:
            result: prompt generate result
            packet: prompt relate packet

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
            np_logits = np.expand_dims(np.array(packet.all_last_logits), axis=0)
            np.save(dump_file, np_logits)
            return dump_file
        else:
            # last_logits shape is [vocab_size]
            # all_last_logits shape is [seq_len, vocab_size]
            if not hasattr(packet, "all_last_logits"):
                packet.all_last_logits = [result.last_logits]
            else:
                packet.all_last_logits.append(result.last_logits)
            return ""

    async def get_packet_results(
        self, 
        get_input_logits: bool, 
        packet: Packet
    ) -> Union[
        AsyncIterable[GenerateResult], Tuple[AsyncIterable[GenerateResult], float, str]
    ]:
        # Save last generate token index minus 1, cann't use len(generate_ids) because may generate
        #  many times, but generate thread only schedule once, which means length bigger than last generate token index.
        _gen_get_id = 0
        while True:
            result = await packet.get_result()
            if result is None:
                if packet.exception:
                    raise packet.exception
                break

            if get_input_logits:
                gen_ids = packet.generate_ids[:_gen_get_id]
                _gen_get_id += 1
                # 1. label = input_ids + generate_ids[:-1], [:-1] is remove just generate token
                labels = torch.LongTensor(packet.request.input_ids + gen_ids)
                logger.debug(
                    f"label shape: {labels.shape}, input_logits shape: {len(result.input_logits)}"
                )
                # 2. .view convert List to Tensor view
                input_logits = torch.FloatTensor(result.input_logits).view(
                    1, labels.size(-1) - 1, -1
                )
                perplexity = calc_perplexity(input_logits=input_logits, labels=labels)
                dump_file = await self.dump_last_logits(result, packet)
                yield result, perplexity, dump_file
            else:
                yield result

        if get_input_logits:
            dump_file = await self.dump_last_logits(result, packet)
            yield None, -1, dump_file
        return

    async def generate(
        self, 
        req: GenerateRequest
    ) -> Union[
        AsyncIterable[GenerateResult], Tuple[AsyncIterable[GenerateResult], float, str]
    ]:
        packet = self.Packet(request=req)
        self.submit(packet)

        async for result in self.get_packet_results(
            req.generate_config.get_input_logits, packet
        ):
            yield result
