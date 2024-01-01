import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any, Dict, Iterable, List, Tuple, Union

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from transformers import PreTrainedTokenizer

import llm_perf.core.common as core_comm
from llm_perf.core.common import (
    GenerateConfig,
    GenerateRequest,
    GenerateResult,
    PacketStatus,
)
from llm_perf.core.engine import CoreEngine
from llm_perf.core.sampler import CoreSampler
from llm_perf.utils.logger import logger
from llm_perf.utils.reporter import calc_ppl


class CoreScheduler(ABC):
    def __init__(
        self,
        engine: CoreEngine,
        sampler: CoreSampler,
        tokenizer: PreTrainedTokenizer,
        comm=core_comm,
        **kwargs,
    ) -> None:
        super().__init__()
        self.Packet = comm.Packet
        self.packet_queue: Queue[self.Packet] = Queue()
        self.engine: CoreEngine = engine
        self.sampler: CoreSampler = sampler
        self.tokenizer = tokenizer
        self.add_sep_token = kwargs.get("add_sep_token", False)

        # self.dump_logits = int(os.environ.get("DUMP_LOGITS", 0))

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

    def dump_last_logits(self, result, packet):
        """Dump prompt logits

        Args:
            result: prompt generate result
            packet: prompt relate packet

        Return:
            dump_file: prompt output logits numpy saved file, logits shape: [1, generate_token_len, vocab_size]
        """
        if packet.result_q_empty() and packet.state == PacketStatus.FINISH:
            tmp_dir = ".tmp_logits"
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            import numpy as np

            dump_file = tmp_dir + "/" + str(int(time.time())) + ".npy"
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

    def get_results(self, packet) -> Iterable[GenerateResult]:
        while True:
            # Both empty and FINISH, empty means we got all generate result, FINISH means generate done,
            # cann't use FINISH alone because may be we haven't get generate result, but generate done
            if packet.result_q_empty() and packet.state == PacketStatus.FINISH:
                break
            result = packet.get_result()
            yield result

    def generate(
        self, req: GenerateRequest
    ) -> Union[Iterable[GenerateResult], Tuple[Iterable[GenerateResult], float, str]]:
        packet = self.Packet(request=req)
        self.submit(packet)

        # Save last generate token index minus 1, cann't use len(generate_ids) because may generate
        #  many times, but generate thread only schedule once, which means length bigger than last generate token index.
        _gen_get_id = 0

        for result in self.get_results(packet):
            if req.generate_config.get_input_logits:
                gen_ids = packet.generate_ids[:_gen_get_id]
                _gen_get_id += 1
                # 1. label = input_ids + generate_ids[:-1], [:-1] is remove just generate token
                labels = torch.LongTensor(packet.request.input_ids + gen_ids)
                logger.debug(f"label shape: {labels.shape}")
                # 2. .view convert List to Tensor view
                logger.debug(f"input_logits shape: {len(result.input_logits)}")
                input_logits = torch.FloatTensor(result.input_logits).view(
                    1, labels.size(-1) - 1, -1
                )
                ppl = calc_ppl(input_logits=input_logits, labels=labels)

                # Dump logits
                dump_file = self.dump_last_logits(result, packet)
                yield result, ppl, dump_file
            else:
                yield result
